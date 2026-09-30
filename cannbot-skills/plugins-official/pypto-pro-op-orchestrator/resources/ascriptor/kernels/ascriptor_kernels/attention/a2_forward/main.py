# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the nine A2/A3 forward-attention sources through OpExec and check them.

    python main.py                          # every case, functional simulator, a2 facade
    python main.py --device a3              # the same sources through the a3 facade
    python main.py --list
    python main.py --case mha_d256          # every case of one variant
    python main.py --case pj_bf16.causal_tail --launcher pipesim
    python main.py --launcher aclnn --case gqa_bf16.ring_reuse

These are nine sources, not nine schedules for one kernel: they differ in ABI as well as in
precision, and a case names its variant the way it names its shape.

    dense_fp16      FP16 in, FP32 out, 512-key groups, no row statistics
    mha_bf16        BF16 probabilities, FP32 out, publishes rowmax and rowsum
    gqa_bf16        the same, with B/HQ/HKV -- several query heads share one KV head
    mha_d256        D = 256, block32 causality switched at run time, FP32 out
    mha_d256_bf16   the same at BF16 output precision
    pj_fp16_lag1    HiFloat8 probability store, 128-key groups, FP16 in, FP32 out
    pj_bf16_lag2    the same at BF16, deeper lag
    pj_bf16_causal  the same with a true per-token causal mask
    pv_stage        not attention: the unnormalized partial PV of each 128-key tile

reference.py's MODELS table is what makes one reference formula serve all nine, and it is
the place to read first if you want to know what a variant actually computes.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import build_kernel
from reference import MODELS, make_inputs, reference

# The FP32 row statistics are captured before the probabilities are rounded, so rowmax and
# rowsum are held to one tight rule whichever variant published them -- and to no residual
# bound, because that is the ceiling each variant's own output rule owns.
ROW_STATISTICS = {"atol": 3e-05, "rtol": 2e-05}

# `out` (or `pv`) gets the rule its own variant's precision needs, so TOLERANCE is keyed by
# variant first and by output name second. The relative L2 is the binding criterion in every
# one of them and the elementwise bound is deliberately loose, for a reason that is the same
# everywhere: the contract stores the probabilities at the source's own precision, and a
# probability landing near a rounding midpoint of that format is taken to opposite
# neighbours by the last bit of any two independent exp() implementations. One such straddle
# is derived to cost up to 1.5e-02 at D=256 -- no pointwise bound can be promised against it
# -- while it moves the relative L2 by about 2e-05 against the 1e-03 to 4e-03 declared here.
# That is why the bf16-output variants carry rtol 1e-02: it is load-bearing, not slack.
TOLERANCE = {
    "dense_fp16": {"default": {"atol": 0.001, "rtol": 0.001, "max_relative_l2": 0.001}},
    "gqa_bf16": {"default": {"atol": 0.001, "rtol": 0.001, "max_relative_l2": 0.001},
                 "rowmax": ROW_STATISTICS, "rowsum": ROW_STATISTICS},
    "mha_bf16": {"default": {"atol": 0.001, "rtol": 0.001, "max_relative_l2": 0.001},
                 "rowmax": ROW_STATISTICS, "rowsum": ROW_STATISTICS},
    "mha_d256": {"default": {"atol": 0.001, "rtol": 0.001, "max_relative_l2": 0.001}},
    "mha_d256_bf16": {"default": {"atol": 0.001, "rtol": 0.01, "max_relative_l2": 0.004}},
    "pj_bf16_lag2": {"default": {"atol": 0.001, "rtol": 0.01, "max_relative_l2": 0.004},
                     "rowmax": ROW_STATISTICS, "rowsum": ROW_STATISTICS},
    "pj_bf16_causal": {"default": {"atol": 0.001, "rtol": 0.01, "max_relative_l2": 0.004},
                       "rowmax": ROW_STATISTICS, "rowsum": ROW_STATISTICS},
    "pj_fp16_lag1": {"default": {"atol": 5e-05, "rtol": 0.001, "max_relative_l2": 0.001},
                     "rowmax": ROW_STATISTICS, "rowsum": ROW_STATISTICS},
    # pv_stage's atol is the source's own 2e-2 and is about unnormalized PV, not about a
    # loose kernel: FP32 QK and exp roundoff crosses FP16 probability midpoints, and the
    # result has not been divided by a denominator that would scale it back down.
    "pv_stage": {"default": {"atol": 0.02, "rtol": 0.001, "max_relative_l2": 0.001}},
}

CASES = [
    {"id": "dense_fp16.group512_aligned", "seed": 0, "block_dim": 1,
     "purpose": "One query tile against two 128-key tiles inside a single 512-key group",
     "parameters": {"variant": "dense_fp16", "BH": 1, "S1": 128, "S2": 256, "D": 128}},
    {"id": "dense_fp16.group512_heads_idle", "seed": 1, "block_dim": 3,
     "purpose": "Two heads over three cores: the third core owns no head",
     "parameters": {"variant": "dense_fp16", "BH": 2, "S1": 128, "S2": 256, "D": 128}},
    {"id": "dense_fp16.group512_group_reuse", "seed": 2, "block_dim": 2,
     "purpose": "Multiple query tiles per core: key-tile reuse, and the group workspace ring "
                "crosses a head boundary",
     "parameters": {"variant": "dense_fp16", "BH": 3, "S1": 256, "S2": 1152, "D": 128}},

    {"id": "gqa_bf16.mqa_tail", "seed": 0, "block_dim": 2,
     "purpose": "MQA: two query heads on one KV head, with tails on both axes",
     "parameters": {"variant": "gqa_bf16", "B": 1, "HQ": 2, "HKV": 1, "S1": 129, "S2": 133,
                    "D": 128}},
    {"id": "gqa_bf16.gqa_batches", "seed": 1, "block_dim": 3,
     "purpose": "Two batches of four query heads over two KV heads: the head map must not "
                "cross a batch",
     "parameters": {"variant": "gqa_bf16", "B": 2, "HQ": 4, "HKV": 2, "S1": 65, "S2": 73,
                    "D": 128}},
    {"id": "gqa_bf16.source_smallest", "seed": 42, "block_dim": 4,
     "purpose": "The smallest original source shape: 384 aligned tokens on four cores",
     "parameters": {"variant": "gqa_bf16", "B": 1, "HQ": 4, "HKV": 2, "S1": 384, "S2": 384,
                    "D": 128}},
    {"id": "gqa_bf16.ring_reuse", "seed": 51, "block_dim": 2,
     "purpose": "Seven query tiles per core wrap the five row-state slots while both query "
                "heads share one KV head",
     "parameters": {"variant": "gqa_bf16", "B": 1, "HQ": 2, "HKV": 1, "S1": 769, "S2": 385,
                    "D": 128}},

    {"id": "mha_bf16.aligned", "seed": 0, "block_dim": 1,
     "purpose": "One query tile, two key tiles, no tail: the BF16 probability path alone",
     "parameters": {"variant": "mha_bf16", "BH": 1, "S1": 128, "S2": 256, "D": 128}},
    {"id": "mha_bf16.tail_heads", "seed": 1, "block_dim": 3,
     "purpose": "Two heads of 65 queries over three cores, with a 73-key tail and an idle core",
     "parameters": {"variant": "mha_bf16", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "mha_bf16.source_tail", "seed": 42, "block_dim": 20,
     "purpose": "The original 257-token boundary at the source's own 20-core launch",
     "parameters": {"variant": "mha_bf16", "BH": 1, "S1": 257, "S2": 257, "D": 128}},
    {"id": "mha_bf16.group_reuse", "seed": 2, "block_dim": 1,
     "purpose": "Seven query tiles on one core wrap both the five-slot row state and the "
                "group workspace ring",
     "parameters": {"variant": "mha_bf16", "BH": 1, "S1": 769, "S2": 641, "D": 128}},

    {"id": "mha_d256.full_aligned", "seed": 0, "block_dim": 1,
     "purpose": "D=256 dense: one 64-row M tile against two key tiles",
     "parameters": {"variant": "mha_d256", "BH": 1, "S1": 64, "S2": 256, "D": 256,
                    "is_causal": 0}},
    {"id": "mha_d256.full_tail", "seed": 1, "block_dim": 2,
     "purpose": "D=256 dense with one row past the 64-row tile and a 73-key tail",
     "parameters": {"variant": "mha_d256", "BH": 1, "S1": 65, "S2": 73, "D": 256,
                    "is_causal": 0}},
    {"id": "mha_d256.full_heads_idle", "seed": 2, "block_dim": 3,
     "purpose": "D=256 dense: two half-empty M tiles over three cores, one idle",
     "parameters": {"variant": "mha_d256", "BH": 2, "S1": 33, "S2": 129, "D": 256,
                    "is_causal": 0}},
    {"id": "mha_d256.full_group_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Three M tiles per active core and two key groups per tile cross the "
                "four-slot workspace ring",
     "parameters": {"variant": "mha_d256", "BH": 1, "S1": 321, "S2": 641, "D": 256,
                    "is_causal": 0}},
    {"id": "mha_d256.block32_aligned", "seed": 0, "block_dim": 1,
     "purpose": "The same shape with is_causal=1: the block32 staircase switched on at run time",
     "parameters": {"variant": "mha_d256", "BH": 1, "S1": 64, "S2": 256, "D": 256,
                    "is_causal": 1}},
    {"id": "mha_d256.block32_tail", "seed": 1, "block_dim": 2,
     "purpose": "Block32 causality where the tail row and the diagonal block coincide",
     "parameters": {"variant": "mha_d256", "BH": 1, "S1": 65, "S2": 73, "D": 256,
                    "is_causal": 1}},
    {"id": "mha_d256.block32_heads_idle", "seed": 2, "block_dim": 3,
     "purpose": "Block32 causality with two partial M tiles and an idle core",
     "parameters": {"variant": "mha_d256", "BH": 2, "S1": 33, "S2": 129, "D": 256,
                    "is_causal": 1}},
    {"id": "mha_d256.block32_group_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Block32 causality across three M tiles and a wrapping four-slot ring",
     "parameters": {"variant": "mha_d256", "BH": 1, "S1": 321, "S2": 641, "D": 256,
                    "is_causal": 1}},

    {"id": "mha_d256_bf16.full_aligned", "seed": 0, "block_dim": 1,
     "purpose": "The BF16-output control for mha_d256.full_aligned: identical shape, one "
                "more rounding on the way out",
     "parameters": {"variant": "mha_d256_bf16", "BH": 1, "S1": 64, "S2": 256, "D": 256,
                    "is_causal": 0}},
    {"id": "mha_d256_bf16.full_tail", "seed": 1, "block_dim": 2,
     "purpose": "BF16 output with a one-row M tail and a 73-key tail",
     "parameters": {"variant": "mha_d256_bf16", "BH": 1, "S1": 65, "S2": 73, "D": 256,
                    "is_causal": 0}},
    {"id": "mha_d256_bf16.full_heads_idle", "seed": 2, "block_dim": 3,
     "purpose": "BF16 output over three cores with one idle",
     "parameters": {"variant": "mha_d256_bf16", "BH": 2, "S1": 33, "S2": 129, "D": 256,
                    "is_causal": 0}},
    {"id": "mha_d256_bf16.full_group_reuse", "seed": 3, "block_dim": 2,
     "purpose": "BF16 output across three M tiles and a wrapping four-slot ring",
     "parameters": {"variant": "mha_d256_bf16", "BH": 1, "S1": 321, "S2": 641, "D": 256,
                    "is_causal": 0}},
    {"id": "mha_d256_bf16.block32_aligned", "seed": 0, "block_dim": 1,
     "purpose": "BF16 output with the block32 staircase switched on",
     "parameters": {"variant": "mha_d256_bf16", "BH": 1, "S1": 64, "S2": 256, "D": 256,
                    "is_causal": 1}},
    {"id": "mha_d256_bf16.block32_tail", "seed": 1, "block_dim": 2,
     "purpose": "BF16 output where the tail row and the diagonal block coincide",
     "parameters": {"variant": "mha_d256_bf16", "BH": 1, "S1": 65, "S2": 73, "D": 256,
                    "is_causal": 1}},
    {"id": "mha_d256_bf16.block32_heads_idle", "seed": 2, "block_dim": 3,
     "purpose": "BF16 output, block32 causality, two partial M tiles and an idle core",
     "parameters": {"variant": "mha_d256_bf16", "BH": 2, "S1": 33, "S2": 129, "D": 256,
                    "is_causal": 1}},
    {"id": "mha_d256_bf16.block32_group_reuse", "seed": 3, "block_dim": 2,
     "purpose": "BF16 output, block32 causality, three M tiles and a wrapping ring",
     "parameters": {"variant": "mha_d256_bf16", "BH": 1, "S1": 321, "S2": 641, "D": 256,
                    "is_causal": 1}},

    {"id": "pj_bf16.lag2_aligned", "seed": 0, "block_dim": 1,
     "purpose": "One 128-key group, no tail: the HiFloat8 probability store on its own",
     "parameters": {"variant": "pj_bf16_lag2", "BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "pj_bf16.lag2_tail", "seed": 1, "block_dim": 2,
     "purpose": "One row past the tile and five keys past the group, at lag two",
     "parameters": {"variant": "pj_bf16_lag2", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "pj_bf16.lag2_heads_idle", "seed": 2, "block_dim": 3,
     "purpose": "Two heads over three cores with one idle, at lag two",
     "parameters": {"variant": "pj_bf16_lag2", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "pj_bf16.lag2_source_shape", "seed": 42, "block_dim": 20,
     "purpose": "The same small double tail at the original 20-core launch count -- a launch "
                "width, not an archived source shape",
     "parameters": {"variant": "pj_bf16_lag2", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "pj_bf16.lag2_ring_reuse", "seed": 51, "block_dim": 2,
     "purpose": "Three query tiles per core, key tails, and repeated workspace-slot reuse "
                "across heads",
     "parameters": {"variant": "pj_bf16_lag2", "BH": 2, "S1": 257, "S2": 385, "D": 128}},

    {"id": "pj_bf16.causal_aligned", "seed": 0, "block_dim": 1,
     "purpose": "The causal control for lag2_aligned: same precision, a true per-token mask",
     "parameters": {"variant": "pj_bf16_causal", "BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "pj_bf16.causal_tail", "seed": 1, "block_dim": 2,
     "purpose": "A per-token causal mask where the diagonal crosses the tail",
     "parameters": {"variant": "pj_bf16_causal", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "pj_bf16.causal_heads_idle", "seed": 2, "block_dim": 3,
     "purpose": "Causal, two heads over three cores with one idle",
     "parameters": {"variant": "pj_bf16_causal", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "pj_bf16.causal_source_shape", "seed": 42, "block_dim": 20,
     "purpose": "The causal double tail at the original 20-core launch count",
     "parameters": {"variant": "pj_bf16_causal", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "pj_bf16.causal_ring_reuse", "seed": 52, "block_dim": 2,
     "purpose": "Causal, three query tiles per core and repeated workspace-slot reuse across heads",
     "parameters": {"variant": "pj_bf16_causal", "BH": 2, "S1": 257, "S2": 385, "D": 128}},

    {"id": "pj_fp16.lag1_aligned", "seed": 0, "block_dim": 1,
     "purpose": "The FP16 control for the HiFloat8 store: one group, lag one",
     "parameters": {"variant": "pj_fp16_lag1", "BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "pj_fp16.lag1_tail", "seed": 1, "block_dim": 2,
     "purpose": "FP16, one row past the tile and five keys past the group, at lag one",
     "parameters": {"variant": "pj_fp16_lag1", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "pj_fp16.lag1_heads_idle", "seed": 2, "block_dim": 3,
     "purpose": "FP16, two heads over three cores with one idle",
     "parameters": {"variant": "pj_fp16_lag1", "BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "pj_fp16.lag1_source_shape", "seed": 42, "block_dim": 20,
     "purpose": "The FP16 double tail at the original 20-core launch count",
     "parameters": {"variant": "pj_fp16_lag1", "BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "pj_fp16.lag1_ring_reuse", "seed": 51, "block_dim": 2,
     "purpose": "FP16, three query tiles per core and repeated workspace-slot reuse across heads",
     "parameters": {"variant": "pj_fp16_lag1", "BH": 2, "S1": 257, "S2": 385, "D": 128}},

    {"id": "pv_stage.stage_aligned", "seed": 0, "block_dim": 1,
     "purpose": "Unnormalized partial PV: two 128-key tiles publish two separate results",
     "parameters": {"variant": "pv_stage", "BH": 1, "S1": 128, "S2": 256, "D": 128}},
    {"id": "pv_stage.stage_heads_idle", "seed": 1, "block_dim": 3,
     "purpose": "Partial PV for two heads over three cores, one idle",
     "parameters": {"variant": "pv_stage", "BH": 2, "S1": 128, "S2": 256, "D": 128}},
    {"id": "pv_stage.stage_group_reuse", "seed": 2, "block_dim": 2,
     "purpose": "Partial PV with multiple query tiles per core: nine key tiles per head, "
                "and the ring crosses a head boundary",
     "parameters": {"variant": "pv_stage", "BH": 3, "S1": 256, "S2": 1152, "D": 128}},
]


def execute(case, inputs, launcher, device):
    """Launch one variant. Every destination is handed in poisoned with NaN and seeded into
    the launch, so a row no core claimed reads back as NaN rather than as a zero.

    Three ABIs share this one function. The plain one passes (S1, S2, D, BH, scale); `d256`
    additionally passes the flattened row counts and the run-time `is_causal` switch; `gqa`
    passes HQ and HKV so the kernel can map query heads onto KV heads itself. `pv_stage`
    publishes S2 // 128 partial results per query row instead of one, which is why the
    destination has that many times as many rows."""
    p = case["parameters"]
    model = MODELS[p["variant"]]
    bh, s1, s2, dim = inputs["BH"], p["S1"], p["S2"], p["D"]
    rows = bh * s1 * (s2 // 128 if model["partial_pv"] else 1)
    out = torch.full((rows, dim), float("nan"), dtype=getattr(torch, model["output_dtype"]))

    scalars = (s1, s2, dim, bh, dim ** -0.5)
    if model["entry_abi"] == "d256":
        scalars = (bh * s1, bh * s2, dim, s1, s2, bh, dim ** -0.5, p["is_causal"])
    elif model["entry_abi"] == "gqa":
        scalars = (s1, s2, dim, bh, p["HQ"], p["HKV"], dim ** -0.5)

    op = OpExec(build_kernel(p["variant"], device), launcher=launcher, backend="cce",
                device=device, block_dim=case["block_dim"],
                out_dir=f"tmp/{launcher}/{p['variant']}", seed_outputs=True)
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    if not model["stats"]:
        return {"pv" if model["partial_pv"] else "out": op(q, k, v, out, *scalars)}

    # The row statistics are published per head at a stride padded up to a multiple of eight
    # FP32 values, so two heads can never share a 32-byte GM line. Slicing them back to the
    # compact [BH * S1] public layout is host work and stays here.
    stride = (s1 + 7) // 8 * 8
    rowmax = torch.full((bh * stride,), float("nan"))
    rowsum = torch.full_like(rowmax, float("nan"))
    values = op(q, k, v, out, rowmax, rowsum, *scalars)
    compact = {name: value.reshape(bh, stride)[:, :s1].contiguous().reshape(-1)
               for name, value in zip(("rowmax", "rowsum"), values[1:], strict=True)}
    return {"out": values[0], **compact}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail, outside = torch.equal(got, want), "  bitwise", got != want
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        atol, rtol = bounds.get("atol", 0.0), bounds.get("rtol", 0.0)
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for each
        # element. Print the worst element as a fraction of its own allowance, so the number has a
        # bound of 1 and a passing line cannot read as a failing one.
        room = (atol + rtol * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        outside = ~torch.isclose(got.float(), want.float(), **bounds)
        ok = not outside.any().item()
        detail = f"  allclose={margin:.2f}x (atol={atol:g} rtol={rtol:g})"
        if "max_relative_l2" in tolerance:
            norm = torch.linalg.vector_norm(want.double().flatten())
            residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
            relative = (residual / norm).item() if norm > 0 else residual.item()
            ok = ok and relative <= tolerance["max_relative_l2"]
            detail += f"  rel_l2={relative:.3e}/{tolerance['max_relative_l2']}"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok and outside.any():
        # The destinations arrive NaN-poisoned, so an element that is still NaN was never written.
        # Saying which, and where, is the difference between "something is nan" and "row 127 is".
        idx, poison = outside.nonzero(), int((outside & torch.isnan(got.float())).sum())
        note = f", {poison} still NaN-poisoned (never written)" if poison else ""
        print(f"      {len(idx)}/{outside.numel()} elements outside{note}; "
              f"first {' '.join(str(tuple(i.tolist())) for i in idx[:3])}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--device", default="a2", choices=("a2", "a3"))
    parser.add_argument("--case", default="all", help="a case id, a variant name, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:36s} {case['purpose']}")
        return 0

    selected = [case for case in CASES
                if args.case in ("all", case["id"], case["parameters"]["variant"])]
    if not selected:
        parser.error(f"no case or variant named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (S1={p['S1']} S2={p['S2']} D={p['D']}, device={args.device}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.device)
        rules = TOLERANCE[p["variant"]]
        for name in expected:
            if not compare(name, actual[name], expected[name], rules.get(name, rules["default"])):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.rsplit('/', 1)[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
