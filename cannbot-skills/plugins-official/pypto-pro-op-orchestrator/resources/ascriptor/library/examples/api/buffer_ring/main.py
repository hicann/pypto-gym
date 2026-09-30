# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two-slot rings in UB and in GM, with the GM hand-off written out as an event.

    python main.py                      # every case, functional simulator
    python main.py --list               # the case ids, with their purpose
    python main.py --launcher pipesim   # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn     # the cce backend, on this machine's card

Five rows pass through two `DBuff` slots in UB and two `GMBuff` slots in GM. Five beats over two
slots means both rings wrap twice and the fifth beat takes slot 0 for the third time, so a kernel that
forgot a slot was still in use has somewhere to go wrong.

The one thing written by hand is `SEvent(Pipe.MTE3, Pipe.MTE2)`: the GM workspace is written by MTE3
and read back by MTE2, and `auto_sync()` does not order a round trip through GM. Everything on chip
-- both UB rings and their reuse -- it does order.

The answer is a copy, so a wrong slot or a missed event shows up as a whole row in the wrong place.
The rows are ten apart by construction, which is what makes that unmistakable rather than plausible.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import buffer_ring
from reference import BEATS, LANES, SEPARATION, make_inputs, reference

DEVICE = "a5"

POISON = float("nan")     # a row the ring never delivered is a NaN, not a neighbouring row
SLOTS = 2

OUTPUTS = ("o",)

CASES = [
    {"id": "five_beats", "seed": 8301, "block_dim": 1, "parameters": {"beats": BEATS, "slots": SLOTS},
     "purpose": "The original five rows over two slots: both rings wrap twice and the fifth beat "
                "reuses slot 0 for the third time, which is the reuse a missing event would break"},
    {"id": "second_seed", "seed": 8302, "block_dim": 1, "parameters": {"beats": BEATS, "slots": SLOTS},
     "purpose": "The same program on different data, so a row that happened to land correctly is "
                "not the whole evidence"},
]


def check_domain(inputs, expected):
    """The rows have to be distinguishable, or a swapped slot would look like a correct answer."""
    x = inputs["x"]
    means = x.mean(dim=1)
    if not bool((means.diff() > SEPARATION / 2).all()):
        raise ValueError("each row must sit clearly above the one before it, or a wrong slot hides")
    if not torch.equal(expected["o"], x):
        raise ValueError("the reference for a copy is its input")
    if expected["o"].data_ptr() == x.data_ptr():
        raise ValueError("the reference must not alias the input")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives filled with NaN and seeded in, so a row the ring never
    delivered reads back as NaN rather than as whatever was there before."""
    op = OpExec(buffer_ring, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], torch.full((BEATS, LANES), POISON))}


def compare(name, got, want):
    """Bitwise: a copy changes no bit."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:3s} FAIL  {got.dtype}{tuple(got.shape)} != {want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu().contiguous(), want.cpu().contiguous()
    ok = torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8))
    print(f"    {name:3s} {'ok  ' if ok else 'FAIL'}  bitwise over {BEATS} rows of {LANES} lanes")
    if not ok:
        wrong = [beat for beat in range(BEATS) if not torch.equal(got[beat], want[beat])]
        # A wrong slot delivers another beat's row, and the rows are ten apart, so say which.
        delivered = {beat: int((want - got[beat]).abs().sum(dim=1).argmin()) for beat in wrong[:4]}
        unwritten = [beat for beat in wrong if bool(torch.isnan(got[beat]).any())]
        print(f"      rows {wrong} differ; each one's nearest source row is {delivered}"
              + (f"; rows {unwritten} still hold the NaN fill (never delivered)"
                 if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa", "pypto_pro"))
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
        print(f"{case['id']}  ({p['beats']} beats over {p['slots']} slots, launcher={args.launcher})")
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
