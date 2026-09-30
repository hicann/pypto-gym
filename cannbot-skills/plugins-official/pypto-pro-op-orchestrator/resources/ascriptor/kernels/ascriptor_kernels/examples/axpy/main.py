# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the AXPY kernel through OpExec and check it against the independent reference.

    python main.py                          # every case, functional simulator
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn         # cce backend, on this machine's card
    python main.py --launcher pypto --case zeros

This folder is self-contained: it imports the installed ``ascriptor`` and nothing
from the repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import axpy
from reference import make_inputs, reference

# Each case is the deterministic input recipe reference.make_inputs reads.
# `tolerance` is absent when the kernel must reproduce the reference bit for bit.
CASES = [
    {"id": "random", "seed": 7101, "block_dim": 1,
     "parameters": {"shape": [1, 64], "dtype": "float32", "pattern": "random"}},
    {"id": "cancellation", "seed": 7102, "block_dim": 1,
     "parameters": {"shape": [1, 64], "dtype": "float32", "pattern": "cancellation"}},
    {"id": "zeros", "seed": 7103, "block_dim": 1,
     "parameters": {"shape": [1, 64], "dtype": "float32", "pattern": "zeros"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the kernel. seed_outputs hands the kernel the caller's output tensor, so a
    lane the kernel never writes keeps its NaN poison instead of reading back as a zero."""
    op = OpExec(axpy, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"o": op(inputs["x"], inputs["y"], inputs["o"])}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want) if tolerance is None else torch.allclose(got, want, **tolerance)
    if tolerance is None:
        detail = ""
    else:
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (tolerance.get("atol", 0.0) + tolerance.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        detail = f"  allclose={margin:.2f}x ({tolerance})"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:10s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok:
        # Destinations arrive NaN-poisoned, so an element still NaN was never written.
        outside = (got != want) if tolerance is None else ~torch.isclose(got.float(), want.float(), **tolerance)
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
        print("\n".join(case["id"] for case in CASES))
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name], case.get("tolerance")):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
