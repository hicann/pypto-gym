# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the host-metadata GQA prefill through OpExec and check it against the interval oracle.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case whole_query

The grid is not decided on the device. `reference.schedule` builds the per-core flat intervals
the case names and turns them into the int32 fa_meta/fd_meta tensors, which are passed in beside
Q, K and V; the kernel reads its own interval out of them. That makes the schedule a case
parameter -- `staggered` forces interior boundaries off the m-block edges so the Flash-Decode
merge runs, `whole_query` aligns them so it never does.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import fia_gqa_step3_transpose_kernel
from reference import make_inputs, reference

# The reference is not a dense softmax: it replays the same per-interval partial states, the
# same max-aware merge and the same BF16 rounding of the published probabilities. What is left
# between the two is FP32 reduction order and the final rounding, which is why the relative L2
# bound here is tighter than the one the flat PFA units use.
TOLERANCE = {"atol": 0.001, "rtol": 0.01, "max_relative_l2": 0.004}

# `block_dim` is how many cores are launched; `work_cores` is how many are given an interval.
# The difference is deliberate -- a launched core with no work still joins the vector-wide
# barrier before the merge -- so neither is a free knob.
CASES = [
    {"id": "dev_split", "seed": 0, "block_dim": 3,
     "purpose": "Three cores with the boundaries nudged off the m-block edges, so every query "
                "tile is split and the Flash-Decode merge runs",
     "parameters": {"M": 384, "N": 259, "D": 128, "work_cores": 3, "schedule": "staggered"}},
    {"id": "tail_idle", "seed": 1, "block_dim": 4,
     "purpose": "A 65-row query tail, and a fourth launched core with no interval that still has "
                "to reach the barrier",
     "parameters": {"M": 321, "N": 259, "D": 128, "work_cores": 3, "schedule": "staggered"}},
    {"id": "whole_query", "seed": 2, "block_dim": 4,
     "purpose": "Whole m-blocks per core: nothing is split, no merge runs, two launched cores "
                "are idle, and both the query and key axes end on a one-row tail",
     "parameters": {"M": 257, "N": 385, "D": 128, "work_cores": 2, "schedule": "whole_query"}},
    {"id": "single_core", "seed": 3, "block_dim": 1,
     "purpose": "One core owning one partial query tile and one partial key tile: the tail paths "
                "with no merge at all",
     "parameters": {"M": 65, "N": 73, "D": 128, "work_cores": 1, "schedule": "even"}},
]


def execute(case, inputs, launcher, backend):
    """The output is allocated NaN-poisoned and seeded into the launch, so a query row no drain
    writes reads back as NaN rather than as a plausible zero. block_dim must be the launch core
    count the metadata was built for: fa_meta has one row per launched core."""
    out = torch.full((inputs["M"], 128), float("nan"), dtype=torch.bfloat16)
    op = OpExec(fia_gqa_step3_transpose_kernel, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(inputs["q"], inputs["k"], inputs["v"], inputs["fa_meta"], inputs["fd_meta"],
                      out, inputs["M"], inputs["N"], 128, 128 ** -0.5,
                      (inputs["N"] + 127) // 128)}


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
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:14s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (M={p['M']}, N={p['N']}, schedule={p['schedule']}, "
              f"work_cores={p['work_cores']}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
