# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the mixed C/V pipelines through OpExec and check them against the staged reference.

    python main.py                                  # all 115 cases, functional simulator
    python main.py --pattern CVCVC                  # one stage graph, all four schedules
    python main.py --mode pipeline --pattern CVC    # one kernel, every size
    python main.py --list
    python main.py --launcher pipesim --case cvc_pipeline_7

Five stage graphs, four schedules each. All four schedules of one graph compute exactly the
same function, so they are directly comparable: if `pipeline` disagrees with `serial` the
difference is the schedule, because nothing else changed.

  serial           one item finishes before the next starts
  pipeline         the next item's first stage overlaps this item's later stages
  resident_serial  operands stay on chip across items, serially
  resident         both at once

Sizes are 1, 2, 3, 5 and 7 full 32-row tiles, chosen so a ring of slots wraps zero, one and
several times; `pipeline` additionally runs on two cores at 1, 5 and 7 tiles, where the item
split and the delayed schedule interact.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

import kernel
from reference import MODES, PATTERNS, ROWS, make_inputs, reference

# Independent FP32 matmuls over bounded dyadic inputs, with an explicit FP16 rounding between
# stages. The bound covers that one rounding; it is not a budget for the arithmetic.
TOLERANCE = {"rtol": 0.003, "atol": 0.0001}

# seed = base(number of stages) + tiles. Every one of the 115 declared cases satisfies this,
# so the table is written as the rule rather than as 115 near-identical rows.
SEED_BASE = {3: 91300, 4: 91400, 5: 91500}
# `pipeline` is the only schedule with a two-core case: it is the one whose correctness
# depends on how items are split, so the sizes where a ring wraps are run on two cores too.
TWO_CORE = {"pipeline": (32, 160, 224)}


def _case(pattern, mode, rows, block_dim):
    tiles = rows // 32
    return {"id": f"{pattern.lower()}_{mode}_{tiles}" + ("_cores2" if block_dim == 2 else ""),
            "seed": SEED_BASE[len(pattern)] + tiles, "block_dim": block_dim,
            "purpose": f"{pattern} on the {mode} schedule, {tiles} tile(s)"
                       + (f", split over {block_dim} cores" if block_dim > 1 else ""),
            "parameters": {"pattern": pattern, "mode": mode, "rows": rows}}


CASES = [_case(pattern, mode, rows, block_dim)
         for pattern in PATTERNS for mode in MODES for rows in ROWS
         for block_dim in (1, 2) if block_dim == 1 or rows in TWO_CORE.get(mode, ())]


def execute(case, inputs, launcher, backend):
    """Pick the kernel this case's (graph, schedule) pair names. The output is handed in as
    NaN and seeded into the launch, so a tile the schedule fails to drain reads back as NaN."""
    entry = getattr(kernel, inputs["pattern"].lower() + "_" + inputs["mode"])
    op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"y": op(inputs["x"], inputs["w0"], inputs["w1"], inputs["w2"],
                    inputs["y"], inputs["rows"])}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail = torch.equal(got, want), "  exact"
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (bounds.get("atol", 0.0) + bounds.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok, detail = torch.allclose(got.float(), want.float(), **bounds), f"  allclose={margin:.2f}x ({bounds})"
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
    parser.add_argument("--pattern", choices=PATTERNS, help="only this stage graph")
    parser.add_argument("--mode", choices=MODES, help="only this schedule")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()

    selected = [c for c in CASES
                if args.case in ("all", c["id"])
                and (args.pattern is None or c["parameters"]["pattern"] == args.pattern)
                and (args.mode is None or c["parameters"]["mode"] == args.mode)]
    if args.list:
        for case in selected:
            print(f"{case['id']:28s} {case['purpose']}")
        print(f"\n{len(selected)} of {len(CASES)} cases")
        return 0
    if not selected:
        parser.error("no case matches; --list prints them")

    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['pattern']} {p['mode']}, rows={p['rows']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
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
