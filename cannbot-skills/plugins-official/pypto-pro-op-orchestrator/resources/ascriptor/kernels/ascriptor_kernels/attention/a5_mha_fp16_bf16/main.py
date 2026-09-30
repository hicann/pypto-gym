# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run standard FP16/BF16 MHA through OpExec and check it against the FP64 oracle.

    python main.py                              # every model-scale case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --launcher aclnn --case all  # cce backend on this machine's card, every case
    python main.py --launcher pypto --case model_resident_reuse

Three schedules share one kernel signature and are selected per case by `variant`:
`head_resident` keeps one head's K/V on chip, `head_preload` runs a deeper K/V
pipeline for long prefill, and `packed_decode` packs many one-query items per core.

Case scale: `full_shape` cases are the real model shapes and only run on hardware --
the functional simulator would need hours for one of them. `model` cases are the same
schedules shrunk until a simulator can execute them, each shrunk around one specific
behaviour named in its `purpose`. Under sim and pipesim the full-shape cases are
skipped with a printed reason rather than silently dropped.

This folder is self-contained: it imports the installed `ascriptor` and nothing from
the repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernel
from reference import make_inputs, reference

# One low-precision probability materialization and a final rounding separate the kernel
# from an FP64 formula that rounds only its output, so this comparison is a tolerance,
# not bit equality. The relative L2 residual is what rejects a vacuous output: a kernel
# that writes zeros everywhere passes atol on its own.
TOLERANCE = {"atol": 0.004, "rtol": 0.025, "max_relative_l2": 0.01}

CASES = [
    {"id": "cross_bf16_short_kv", "seed": 20260908, "block_dim": 28, "scale": "full_shape",
     "purpose": "Cross attention, short KV, head-resident schedule",
     "parameters": {"dtype": "bfloat16", "layout": "BSND", "B": 2, "SQ": 512, "SKV": 128, "H": 16,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "head_resident",
                    "input_pattern": "uniform"}},
    {"id": "prefill_fp16_causal_long", "seed": 20260908, "block_dim": 28, "scale": "full_shape",
     "purpose": "Long causal prefill, preload schedule",
     "parameters": {"dtype": "float16", "layout": "BSND", "B": 4, "SQ": 1024, "SKV": 1024, "H": 32,
                    "D": 128, "scale_value": -1.0, "is_causal": True, "variant": "head_preload",
                    "input_pattern": "uniform"}},
    {"id": "decode_fp16_large_batch", "seed": 20260908, "block_dim": 28, "scale": "full_shape",
     "purpose": "Single-query decode over a large batch, packed schedule",
     "parameters": {"dtype": "float16", "layout": "BSND", "B": 128, "SQ": 1, "SKV": 128, "H": 32,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "packed_decode",
                    "input_pattern": "uniform"}},

    {"id": "model_resident_reuse", "seed": 20260909, "block_dim": 1, "scale": "model",
     "purpose": "Repeated head-local KV reuse and a one-query M tail",
     "parameters": {"dtype": "bfloat16", "layout": "BSND", "B": 1, "SQ": 257, "SKV": 128, "H": 2,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "head_resident",
                    "input_pattern": "uniform"}},
    {"id": "model_resident_heads", "seed": 20260910, "block_dim": 1, "scale": "model",
     "purpose": "Different head values expose invalid cross-head sharing",
     "parameters": {"dtype": "bfloat16", "layout": "BSND", "B": 1, "SQ": 3, "SKV": 128, "H": 3,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "head_resident",
                    "input_pattern": "head_distinct"}},
    {"id": "model_resident_equal_kv", "seed": 20260911, "block_dim": 3, "scale": "model",
     "purpose": "Equal K/V values in independent storage, and an idle core",
     "parameters": {"dtype": "bfloat16", "layout": "BSND", "B": 1, "SQ": 3, "SKV": 128, "H": 2,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "head_resident",
                    "input_pattern": "equal_kv"}},
    {"id": "model_resident_zero", "seed": 20260912, "block_dim": 3, "scale": "model",
     "purpose": "Zero output with a poisoned destination: every byte must still be written",
     "parameters": {"dtype": "bfloat16", "layout": "BSND", "B": 1, "SQ": 3, "SKV": 128, "H": 1,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "head_resident",
                    "input_pattern": "zero"}},
    {"id": "model_resident_nonzero", "seed": 20260913, "block_dim": 3, "scale": "model",
     "purpose": "The nonzero control for model_resident_zero: same signature, real values",
     "parameters": {"dtype": "bfloat16", "layout": "BSND", "B": 1, "SQ": 3, "SKV": 128, "H": 1,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "head_resident",
                    "input_pattern": "uniform"}},
    {"id": "model_preload_reuse", "seed": 20260914, "block_dim": 1, "scale": "model",
     "purpose": "Four KV256 tiles wrap each three-slot K/V cache, including KV3 to the next "
                "query's KV0; the first query item ends on a 127-key visible prefix",
     "parameters": {"dtype": "float16", "layout": "BSND", "B": 1, "SQ": 257, "SKV": 1024, "H": 2,
                    "D": 128, "scale_value": -1.0, "is_causal": True, "variant": "head_preload",
                    "input_pattern": "uniform"}},
    {"id": "model_preload_explicit_scale", "seed": 20260915, "block_dim": 1, "scale": "model",
     "purpose": "Explicit positive scale, and the right-bottom causal boundary inside one KV256 tile",
     "parameters": {"dtype": "float16", "layout": "BSND", "B": 1, "SQ": 129, "SKV": 256, "H": 1,
                    "D": 128, "scale_value": 0.08838, "is_causal": True, "variant": "head_preload",
                    "input_pattern": "uniform"}},
    {"id": "model_preload_auto_zero_scale", "seed": 20260916, "block_dim": 2, "scale": "model",
     "purpose": "Scale 0.0 is the sentinel for 1/sqrt(D); three real queries inside physical M128",
     "parameters": {"dtype": "float16", "layout": "BSND", "B": 1, "SQ": 3, "SKV": 256, "H": 1,
                    "D": 128, "scale_value": 0.0, "is_causal": True, "variant": "head_preload",
                    "input_pattern": "uniform"}},
    {"id": "model_packed_reuse", "seed": 20260917, "block_dim": 1, "scale": "model",
     "purpose": "Twelve two-head items across two batches on one core; K2/V2/P2 reuse and final drain",
     "parameters": {"dtype": "float16", "layout": "BSND", "B": 2, "SQ": 1, "SKV": 128, "H": 12,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "packed_decode",
                    "input_pattern": "uniform"}},
    {"id": "model_packed_heads", "seed": 20260918, "block_dim": 1, "scale": "model",
     "purpose": "Each real row selects its own 128-key segment; all fourteen padding rows stay zero",
     "parameters": {"dtype": "float16", "layout": "BSND", "B": 1, "SQ": 1, "SKV": 128, "H": 8,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "packed_decode",
                    "input_pattern": "head_distinct"}},
    {"id": "model_packed_equal_kv", "seed": 20260919, "block_dim": 1, "scale": "model",
     "purpose": "Equal but non-aliased K/V preserve independent head ownership",
     "parameters": {"dtype": "float16", "layout": "BSND", "B": 1, "SQ": 1, "SKV": 128, "H": 4,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "packed_decode",
                    "input_pattern": "equal_kv"}},
    {"id": "model_packed_zero_idle", "seed": 20260920, "block_dim": 28, "scale": "model",
     "purpose": "Two items on 28 cores: 26 idle cores, an idle second Vector per active core",
     "parameters": {"dtype": "float16", "layout": "BSND", "B": 1, "SQ": 1, "SKV": 128, "H": 4,
                    "D": 128, "scale_value": -1.0, "is_causal": False, "variant": "packed_decode",
                    "input_pattern": "zero"}},
]


def execute(case, inputs, launcher, backend):
    """Build the schedule this case names, then launch it. The output tensor is passed in
    poisoned with NaN and seeded into the launch, so a row the kernel never writes reads
    back as NaN rather than as the zero the reference also produces for a zero input."""
    p = case["parameters"]
    entry = make_kernel(p["dtype"], p["layout"], p["B"], p["SQ"], p["SKV"], p["H"],
                        p["D"], p["is_causal"], p["scale_value"], variant=p["variant"])
    out = torch.full_like(inputs["query"], float("nan"))
    op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(inputs["query"], inputs["key"], inputs["value"], out)}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail = torch.equal(got, want), ""
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (bounds.get("atol", 0.0) + bounds.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok = torch.allclose(got.float(), want.float(), **bounds)
        detail = f"  allclose={margin:.2f}x ({bounds})"
        if "max_relative_l2" in tolerance:
            residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
            norm = torch.linalg.vector_norm(want.double().flatten())
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
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:32s} {case['scale']:10s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed, skipped = [], []
    for case in selected:
        if case["scale"] == "full_shape" and args.launcher in ("sim", "pipesim"):
            skipped.append(case["id"])
            continue
        print(f"{case['id']}  ({case['scale']}, {case['parameters']['variant']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
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
