# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fifteen rows of predicate logic under one execution mask, including the counter it consumes.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case prefix_70      # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

One 64-lane INT32 register, one prefix gate built by `update_mask(gate, remaining)`, and fifteen
published rows: a compare, a data select, not/and/or/xor/mov/sel over predicates, a pack and its
unpack, a two-register interleave and its deinterleave, and -- the last row -- the value left in
`remaining` after `update_mask` consumed from it.

Every destination mask in the body is deliberately initialised differently (`MaskType.ALL`,
`NONE`, or default) while the reference computes `gate & ...` for all of them. That agreement is
the statement: at b32 a masked predicate operation clears its inactive lanes rather than
preserving what was there, so the initialisation does not survive.

Every result lane is compared, bitwise, from a destination poisoned with -999.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_masks
from reference import make_inputs, reference

DEVICE = "a5"

LANES = 64
POISON = -999

OUTPUTS = ("o",)

# The fifteen rows, so a failure names the operation instead of an index.
ROWS = ("compare", "select(data)", "mask_not", "mask_and", "mask_or", "mask_xor", "mask_mov",
        "mask_sel", "mask_pack", "mask_unpack", "interleave.left", "interleave.right",
        "deinterleave.0", "deinterleave.1", "remaining")

CASES = [
    {"id": "prefix_0", "seed": 9191, "block_dim": 1,
     "purpose": "count 0: the gate is empty, so every masked row is its inactive value and the "
                "counter is never drawn down -- the case where a destination mask's "
                "initialisation would survive if these operations preserved instead of cleared",
     "parameters": {"count": 0}},
    {"id": "prefix_3", "seed": 9194, "block_dim": 1,
     "purpose": "count 3: the smallest non-empty prefix, three active lanes out of 64",
     "parameters": {"count": 3}},
    {"id": "prefix_63", "seed": 9254, "block_dim": 1,
     "purpose": "count 63: one lane short of the register, so an implementation that rounded the "
                "count up to the register width disagrees in exactly one lane -- the cheapest "
                "case to get wrong and the hardest to notice",
     "parameters": {"count": 63}},
    {"id": "prefix_64", "seed": 9255, "block_dim": 1,
     "purpose": "count 64: exactly the register, and the counter lands on zero. The boundary "
                "between having consumed everything and having some left",
     "parameters": {"count": 64}},
    {"id": "prefix_70", "seed": 9261, "block_dim": 1,
     "purpose": "count 70: more than one register, so `update_mask` takes 64 and leaves 6 behind. "
                "The last row reads that 6 back out of the counter instead of inferring it from "
                "the mask bits",
     "parameters": {"count": 70}},
]


def check_domain(inputs, expected):
    """What the rows rest on: the gate is a prefix of the declared count, and the counter row
    states what `update_mask` left rather than what it was given."""
    x, y = inputs["x"], inputs["y"]
    if x.shape != (1, LANES) or x.dtype != torch.int32 or y.shape != (1, LANES):
        raise ValueError(f"the inputs must be int32[1, {LANES}]")
    count = inputs["count"]
    if expected["o"].shape != (len(ROWS), LANES):
        raise ValueError(f"the reference must publish {len(ROWS)} rows")
    if not bool((expected["o"][-1] == max(count - LANES, 0)).all()):
        raise ValueError(f"the counter row must hold max(count - {LANES}, 0)")
    if (expected["o"] == POISON).any():
        raise ValueError("the reference contains the poison value")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives filled with -999 and seeded in, so a row the vector
    function never wrote is distinguishable from a row of legitimate zeros -- and with an empty
    gate, legitimate zeros are most of the output."""
    entry = make_masks(inputs["count"])
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    destination = torch.full((len(ROWS), LANES), POISON, dtype=torch.int32)
    return {"o": op(inputs["x"], inputs["y"], destination)}


def compare(name, got, want):
    """Bitwise: every row is an exact integer or Boolean formula."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {len(ROWS)} rows x {LANES} lanes")
    if not ok:
        for row in (got != want).any(dim=1).nonzero().flatten().tolist():
            lanes = (got[row] != want[row]).nonzero().flatten().tolist()
            unwritten = bool((got[row] == POISON).all())
            print(f"      {ROWS[row]:16s} {len(lanes)} lanes differ at {lanes[:6]}"
                  + ("  (the whole row still holds the poison: never written)" if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:11s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        count = case["parameters"]["count"]
        print(f"{case['id']}  (count={count}, gate covers {min(count, LANES)}/{LANES} lanes, "
              f"counter left {max(count - LANES, 0)}, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
