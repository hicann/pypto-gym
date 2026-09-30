# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the late-reader CVC residual through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator
    python main.py --launcher pipesim       # where the two schedules actually differ
    python main.py --case pipeline_7

Two kernels compute the same thing on different schedules. `late_reader_serial` finishes one
row tile before starting the next; `late_reader_pipeline` runs a three-slot non-blocking
lookahead so the next tile's first matmul overlaps this tile's vector work.

What the unit is about is the buffer lifetime, not the arithmetic: `p` is read EARLY by the
activation and again LATE by the residual add, so its slot has to stay live across the second
matmul. Give it the same lifetime as `h`, whose last reader is the second matmul, and the
pipelined schedule overwrites `p` before the residual reads it. Under `--launcher sim` both
schedules are simply correct; `--launcher pipesim` is where the difference is visible, because
that is the model that knows when a slot is reused.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

import kernel
from reference import make_inputs, reference

# Exact. The generated inputs are dyadic, so every FP32 product and sum in the formula is
# exact and the single FP16 materialisation of the activation is the only rounding — which
# both sides perform identically. Any difference is a defect, not a precision budget.
TOLERANCE = None

CASES = [
    {"id": "serial_1", "seed": 97032, "block_dim": 1,
     "purpose": "One row tile, serial: no lookahead to get wrong",
     "parameters": {"rows": 32, "mode": "serial"}},
    {"id": "serial_3", "seed": 97096, "block_dim": 1,
     "purpose": "Three tiles, serial: the reference behaviour the pipeline must reproduce",
     "parameters": {"rows": 96, "mode": "serial"}},
    {"id": "serial_7", "seed": 97224, "block_dim": 1,
     "purpose": "Seven tiles, serial",
     "parameters": {"rows": 224, "mode": "serial"}},
    {"id": "pipeline_1", "seed": 97032, "block_dim": 1,
     "purpose": "One tile, pipelined: the lookahead has nothing to overlap with",
     "parameters": {"rows": 32, "mode": "pipeline"}},
    {"id": "pipeline_3", "seed": 97096, "block_dim": 1,
     "purpose": "Three tiles, pipelined: the three-slot ring wraps for the first time",
     "parameters": {"rows": 96, "mode": "pipeline"}},
    {"id": "pipeline_7", "seed": 97224, "block_dim": 1,
     "purpose": "Seven tiles, pipelined: the ring wraps twice, so a p slot is reused twice "
                "while a late reader still needs it",
     "parameters": {"rows": 224, "mode": "pipeline"}},
]


def execute(case, inputs, launcher, backend):
    """The output is handed in as NaN and seeded into the launch, so a row the schedule drops
    reads back as NaN instead of as whatever was in memory."""
    entry = getattr(kernel, "late_reader_" + inputs["mode"])
    op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"y": op(inputs["x"], inputs["w1"], inputs["w2"], inputs["y"], inputs["rows"])}


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
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:12s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (rows={p['rows']}, {p['mode']}, launcher={args.launcher})")
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
