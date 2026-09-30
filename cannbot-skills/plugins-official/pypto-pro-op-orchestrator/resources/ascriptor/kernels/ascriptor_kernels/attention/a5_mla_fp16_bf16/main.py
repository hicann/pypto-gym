# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run standard FP16/BF16 MLA through OpExec and check it against the FP64 oracle.

    python main.py                                # every model-scale case, functional simulator
    python main.py --list                         # the case ids, scale and purpose
    python main.py --variant online_paired        # one schedule, every case that uses it
    python main.py --launcher aclnn --case all    # cce backend on this machine's card
    python main.py --launcher pypto --case model_zero

Six schedules share one 4-D ABI and are selected per case by `variant`. They differ in how the
single-head KV cache moves on chip and where the online softmax state lives:

  serial_nd         the straightforward one, and the control the others are compared against
  prefill_resident  KV stays resident for a causal prefill
  decode_preload    a deeper KV pipeline for multi-query decode
  decode_splitkv    the key range is split across cores and merged afterwards
  online_prefetch   online softmax with the next KV tile prefetched
  online_paired     two items share a core, so their KV tiles and contexts interleave

Case scale: `full_shape` cases are real model shapes and only run on hardware — the functional
simulator would need hours for one. `model` cases are the same schedules shrunk until a
simulator can execute them, each around one named behaviour. Under sim and pipesim the
full-shape cases are skipped with a printed reason rather than silently dropped. The four
`serial_*` cases are full-shape controls: the same shape as a teaching case, run through
`serial_nd`, so a disagreement is the schedule's and not the shape's.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import SCHEDULES, make_kernel
from reference import NAMES, make_inputs, reference

# Bounded [-1, 1] inputs, one low-precision P materialisation and a final FP16/BF16 store,
# against an FP64 formula that rounds only once. The relative L2 residual is what rejects a
# vacuous output: a kernel that writes zeros everywhere passes atol on its own.
TOLERANCE = {"atol": 0.004, "rtol": 0.025, "max_relative_l2": 0.01}

CASES = [
    {"id": "decode_single_query", "seed": 20260907, "block_dim": 16, "scale": "full_shape",
     "purpose": "Single-query decode over a 2048-key cache, split-KV schedule",
     "parameters": {"B": 1, "SQ": 1, "SKV": 2048, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "decode_splitkv", "input_pattern": "uniform"}},
    {"id": "prefill_causal", "seed": 20260907, "block_dim": 28, "scale": "full_shape",
     "purpose": "Causal prefill, KV resident on chip",
     "parameters": {"B": 2, "SQ": 128, "SKV": 128, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "prefill_resident", "input_pattern": "uniform"}},
    {"id": "prefill_bf16_dn448", "seed": 20260907, "block_dim": 28, "scale": "full_shape",
     "purpose": "Causal prefill in BF16 at Dn=448, the non-power-of-two nope width",
     "parameters": {"B": 2, "SQ": 128, "SKV": 128, "Nq": 64, "Dn": 448, "Dr": 64, "dtype": "bfloat16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "prefill_resident", "input_pattern": "uniform"}},
    {"id": "decode_bnsd_two_queries", "seed": 20260907, "block_dim": 28, "scale": "full_shape",
     "purpose": "Two-query decode in BNSD, preload schedule",
     "parameters": {"B": 16, "SQ": 2, "SKV": 2048, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BNSD", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "decode_preload", "input_pattern": "uniform"}},
    {"id": "model_decode_single_query", "seed": 20260910, "block_dim": 1, "scale": "model",
     "purpose": "decode_single_query shrunk to B=1, SQ=1, SKV=192, Nq=8",
     "parameters": {"B": 1, "SQ": 1, "SKV": 192, "Nq": 8, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "model_prefill_causal", "seed": 20260911, "block_dim": 1, "scale": "model",
     "purpose": "prefill_causal shrunk to B=1, SQ=3, SKV=64, Nq=8",
     "parameters": {"B": 1, "SQ": 3, "SKV": 64, "Nq": 8, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "model_prefill_bf16_dn448", "seed": 20260912, "block_dim": 1, "scale": "model",
     "purpose": "prefill_bf16_dn448 shrunk to B=1, SQ=2, SKV=128, Nq=8",
     "parameters": {"B": 1, "SQ": 2, "SKV": 128, "Nq": 8, "Dn": 448, "Dr": 64, "dtype": "bfloat16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "model_decode_bnsd_two_queries", "seed": 20260913, "block_dim": 1, "scale": "model",
     "purpose": "decode_bnsd_two_queries shrunk to B=1, SQ=2, SKV=64, Nq=16",
     "parameters": {"B": 1, "SQ": 2, "SKV": 64, "Nq": 16, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BNSD", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "model_zero", "seed": 20260914, "block_dim": 1, "scale": "model",
     "purpose": "decode_single_query shrunk to B=1, SQ=1, SKV=64, Nq=8, input_pattern=zero",
     "parameters": {"B": 1, "SQ": 1, "SKV": 64, "Nq": 8, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "zero"}},
    {"id": "model_head_tail", "seed": 20260915, "block_dim": 1, "scale": "model",
     "purpose": "decode_single_query shrunk to B=1, SQ=1, SKV=64, Nq=3",
     "parameters": {"B": 1, "SQ": 1, "SKV": 64, "Nq": 3, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "model_slot_reuse", "seed": 20260916, "block_dim": 1, "scale": "model",
     "purpose": "decode_single_query shrunk to B=1, SQ=1, SKV=320, Nq=24",
     "parameters": {"B": 1, "SQ": 1, "SKV": 320, "Nq": 24, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "model_idle_core", "seed": 20260917, "block_dim": 2, "scale": "model",
     "purpose": "decode_single_query shrunk to B=1, SQ=1, SKV=64, Nq=8",
     "parameters": {"B": 1, "SQ": 1, "SKV": 64, "Nq": 8, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "model_causal_boundary", "seed": 20260918, "block_dim": 1, "scale": "model",
     "purpose": "prefill_causal shrunk to B=1, SQ=65, SKV=128, Nq=1",
     "parameters": {"B": 1, "SQ": 65, "SKV": 128, "Nq": 1, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "serial_decode_single_query", "seed": 20260907, "block_dim": 28, "scale": "full_shape",
     "purpose": "serial_nd control at the same shape as decode_single_query: the schedules must agree",
     "parameters": {"B": 1, "SQ": 1, "SKV": 2048, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "serial_prefill_causal", "seed": 20260907, "block_dim": 28, "scale": "full_shape",
     "purpose": "serial_nd control at the same shape as prefill_causal: the schedules must agree",
     "parameters": {"B": 2, "SQ": 128, "SKV": 128, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "serial_prefill_bf16_dn448", "seed": 20260907, "block_dim": 28, "scale": "full_shape",
     "purpose": "serial_nd control at the same shape as prefill_bf16_dn448: the schedules must agree",
     "parameters": {"B": 2, "SQ": 128, "SKV": 128, "Nq": 64, "Dn": 448, "Dr": 64, "dtype": "bfloat16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "serial_decode_bnsd_two_queries", "seed": 20260907, "block_dim": 28, "scale": "full_shape",
     "purpose": "serial_nd control at the same shape as decode_bnsd_two_queries: the schedules must"
                "agree",
     "parameters": {"B": 16, "SQ": 2, "SKV": 2048, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BNSD", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "serial_nd", "input_pattern": "uniform"}},
    {"id": "model_resident_fp16", "seed": 20260908, "block_dim": 3, "scale": "model",
     "purpose": "prefill_causal shrunk to B=2, SQ=3",
     "parameters": {"B": 2, "SQ": 3, "SKV": 128, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "prefill_resident", "input_pattern": "uniform"}},
    {"id": "model_resident_bf16", "seed": 20260909, "block_dim": 3, "scale": "model",
     "purpose": "prefill_bf16_dn448 shrunk to B=2, SQ=3",
     "parameters": {"B": 2, "SQ": 3, "SKV": 128, "Nq": 64, "Dn": 448, "Dr": 64, "dtype": "bfloat16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "prefill_resident", "input_pattern": "uniform"}},
    {"id": "model_optimized_decode_reuse", "seed": 20260910, "block_dim": 1, "scale": "model",
     "purpose": "decode_bnsd_two_queries shrunk to B=1, Nq=64, SKV=640",
     "parameters": {"B": 1, "SQ": 2, "SKV": 640, "Nq": 64, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BNSD", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "decode_preload", "input_pattern": "uniform"}},
    {"id": "model_optimized_decode_tail", "seed": 20260911, "block_dim": 3, "scale": "model",
     "purpose": "decode_bnsd_two_queries shrunk to B=1, SQ=1, Nq=3, SKV=128",
     "parameters": {"B": 1, "SQ": 1, "SKV": 128, "Nq": 3, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BNSD", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "decode_preload", "input_pattern": "uniform"}},
    {"id": "model_split_merge_tail", "seed": 20260912, "block_dim": 3, "scale": "model",
     "purpose": "decode_single_query shrunk to Nq=70, SKV=512",
     "parameters": {"B": 1, "SQ": 1, "SKV": 512, "Nq": 70, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "decode_splitkv", "input_pattern": "uniform"}},
    {"id": "model_split_zero_idle", "seed": 20260913, "block_dim": 28, "scale": "model",
     "purpose": "decode_single_query shrunk to Nq=3, SKV=256",
     "parameters": {"B": 1, "SQ": 1, "SKV": 256, "Nq": 3, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "decode_splitkv", "input_pattern": "zero"}},
    {"id": "model_split_many_tiles", "seed": 20260914, "block_dim": 3, "scale": "model",
     "purpose": "decode_single_query shrunk to Nq=9, SKV=4096",
     "parameters": {"B": 1, "SQ": 1, "SKV": 4096, "Nq": 9, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "decode_splitkv", "input_pattern": "uniform"}},
    {"id": "decode_large_batch", "seed": 20260908, "block_dim": 28, "scale": "full_shape",
     "purpose": "Large-batch decode, paired-item schedule",
     "parameters": {"B": 60, "SQ": 1, "SKV": 2048, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_paired", "input_pattern": "uniform"}},
    {"id": "prefill_bf16_bnsd_long", "seed": 20260908, "block_dim": 28, "scale": "full_shape",
     "purpose": "Long BF16 BNSD prefill, prefetch schedule",
     "parameters": {"B": 2, "SQ": 512, "SKV": 512, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "bfloat16", "layout": "BNSD", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_prefetch", "input_pattern": "uniform"}},
    {"id": "decode_bf16_causal_large_batch", "seed": 20260908, "block_dim": 28, "scale": "full_shape",
     "purpose": "Large-batch causal BF16 decode, paired schedule",
     "parameters": {"B": 96, "SQ": 2, "SKV": 2048, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "bfloat16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_paired", "input_pattern": "uniform"}},
    {"id": "model_prefetch_query_wrap", "seed": 20260930, "block_dim": 1, "scale": "model",
     "purpose": "Repeated M128 items, a BNSD query wrap, partial output and product-alias retirement"
                "(B=1, SQ=129, Nq=2)",
     "parameters": {"B": 1, "SQ": 129, "SKV": 512, "Nq": 2, "Dn": 512, "Dr": 64, "dtype": "bfloat16", "layout": "BNSD", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_prefetch", "input_pattern": "uniform"}},
    {"id": "model_prefetch_causal_boundary", "seed": 20260931, "block_dim": 2, "scale": "model",
     "purpose": "Two and three KV prefixes, a one-row final item and an idle Vector (B=1, SQ=257,"
                "SKV=384, Nq=1)",
     "parameters": {"B": 1, "SQ": 257, "SKV": 384, "Nq": 1, "Dn": 512, "Dr": 64, "dtype": "bfloat16", "layout": "BNSD", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_prefetch", "input_pattern": "uniform"}},
    {"id": "model_prefetch_zero_idle", "seed": 20260932, "block_dim": 28, "scale": "model",
     "purpose": "Zero state, one KV tile, padded physical rows and idle cores (B=1, SQ=2, SKV=128,"
                "Nq=3, input_pattern=zero)",
     "parameters": {"B": 1, "SQ": 2, "SKV": 128, "Nq": 3, "Dn": 512, "Dr": 64, "dtype": "bfloat16", "layout": "BNSD", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_prefetch", "input_pattern": "zero"}},
    {"id": "model_paired_odd_partition", "seed": 20260933, "block_dim": 2, "scale": "model",
     "purpose": "Odd item ceiling, both single/pair orders and repeated six-tile K-slot reuse (B=3,"
                "SKV=768)",
     "parameters": {"B": 3, "SQ": 1, "SKV": 768, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_paired", "input_pattern": "uniform"}},
    {"id": "model_paired_even_partition", "seed": 20260934, "block_dim": 3, "scale": "model",
     "purpose": "Even item ceiling, paired ownership and repeated packs across batches (B=5, SKV=384)",
     "parameters": {"B": 5, "SQ": 1, "SKV": 384, "Nq": 128, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_paired", "input_pattern": "uniform"}},
    {"id": "model_paired_tail", "seed": 20260935, "block_dim": 2, "scale": "model",
     "purpose": "Clipped odd final item, one KV tile, one real row and an idle Vector (B=1, SKV=128,"
                "Nq=129)",
     "parameters": {"B": 1, "SQ": 1, "SKV": 128, "Nq": 129, "Dn": 512, "Dr": 64, "dtype": "float16", "layout": "BSND", "is_causal": False, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_paired", "input_pattern": "uniform"}},
    {"id": "model_paired_causal_decode", "seed": 20260936, "block_dim": 2, "scale": "model",
     "purpose": "BSND physical rows cross query positions, with BF16, partial items and a one-key"
                "causal boundary (B=1, SKV=128, Nq=67)",
     "parameters": {"B": 1, "SQ": 2, "SKV": 128, "Nq": 67, "Dn": 512, "Dr": 64, "dtype": "bfloat16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_paired", "input_pattern": "uniform"}},
    {"id": "model_paired_causal_lifetime", "seed": 20260937, "block_dim": 3, "scale": "model",
     "purpose": "Even-ceiling paired partition with causal prefixes [1,2], [2,1], [2,2]; either"
                "context retires first (B=1, SQ=192, SKV=256, Nq=2, layout=BNSD)",
     "parameters": {"B": 1, "SQ": 192, "SKV": 256, "Nq": 2, "Dn": 512, "Dr": 64, "dtype": "bfloat16", "layout": "BNSD", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_paired", "input_pattern": "uniform"}},
    {"id": "model_paired_zero_idle", "seed": 20260938, "block_dim": 28, "scale": "model",
     "purpose": "Zero inputs, an inactive context, one KV tile and idle cores (B=1, SKV=128, Nq=3,"
                "input_pattern=zero)",
     "parameters": {"B": 1, "SQ": 2, "SKV": 128, "Nq": 3, "Dn": 512, "Dr": 64, "dtype": "bfloat16", "layout": "BSND", "is_causal": True, "scale_value": -1.0, "num_kv_heads": 1, "variant": "online_paired", "input_pattern": "zero"}},
]


def execute(case, inputs, launcher, backend):
    """Build the schedule this case names, then launch it. The output is passed in poisoned
    with NaN and seeded into the launch, so a row the kernel never writes reads back as NaN
    rather than as the zero the reference also produces for a zero input."""
    p = case["parameters"]
    entry = make_kernel(p["dtype"], p["layout"], p["B"], p["SQ"], p["SKV"], p["Nq"],
                        p["Dn"], p["is_causal"], variant=p["variant"])
    out = torch.full_like(inputs["q_nope"], float("nan"))
    op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(*(inputs[name] for name in NAMES), out)}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail = torch.equal(got, want), "  bitwise"
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (bounds.get("atol", 0.0) + bounds.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok, detail = torch.allclose(got.float(), want.float(), **bounds), f"  allclose={margin:.2f}x ({bounds})"
        if "max_relative_l2" in tolerance:
            norm = torch.linalg.vector_norm(want.double().flatten())
            residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
            relative = (residual / norm).item() if norm > 0 else residual.item()
            ok = ok and relative <= tolerance["max_relative_l2"]
            detail += f"  rel_l2={relative:.3e}/{tolerance['max_relative_l2']}"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:6s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok:
        # Destinations arrive NaN-poisoned, so an element still NaN was never written.
        outside = (got != want) if tolerance is None else ~torch.isclose(got.float(), want.float(), **bounds)
        if outside.any():
            idx = outside.nonzero()
            poison = int((outside & torch.isnan(got.float())).sum())
            note = f", {poison} still NaN-poisoned (never written)" if poison else ""
            print(f"      {len(idx)}/{outside.numel()} elements outside{note}; "
                  f"first {' '.join(str(tuple(i.tolist())) for i in idx[:3])}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--variant", choices=sorted(SCHEDULES), help="only this schedule")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()

    selected = [c for c in CASES if args.case in ("all", c["id"])
                and (args.variant is None or c["parameters"]["variant"] == args.variant)]
    if args.list:
        for case in selected:
            print(f"{case['id']:32s} {case['scale']:10s} "
                  f"{case['parameters']['variant']:16s} {case['purpose']}")
        print(f"\n{len(selected)} of {len(CASES)} cases")
        return 0
    if not selected:
        parser.error("no case matches; --list prints them")

    failed, skipped = [], []
    for case in selected:
        if case["scale"] == "full_shape" and args.launcher in ("sim", "pipesim"):
            skipped.append(case["id"])
            continue
        p = case["parameters"]
        print(f"{case['id']}  ({case['scale']}, {p['variant']}, B={p['B']} SQ={p['SQ']} "
              f"SKV={p['SKV']} Nq={p['Nq']}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    ran = len(selected) - len(skipped)
    print(f"\n{ran - len({f.split('/')[0] for f in failed})}/{ran} cases passed")
    if skipped:
        print(f"skipped under {args.launcher} (model-shape cases cover these schedules): "
              + ", ".join(skipped))
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
