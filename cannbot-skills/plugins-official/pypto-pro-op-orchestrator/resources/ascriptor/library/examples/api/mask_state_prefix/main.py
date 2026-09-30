# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read vector mask state set outside a @vf, and replace a register with an exclusive prefix count.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case prefix_last    # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

Two behaviours, one folder, because each explains what the other is not.

`spr`     `set_mask_by_count(n)` and `set_mask(0, pattern)` establish vector mask state at kernel
          scope; `move_mask_spr(mask)` inside the vector function reads it back. Each observation
          is followed by `reset_mask()` before the row is published, and a third observation with
          nothing set confirms the full mask came back.
`prefix`  `unsqueeze(data, mask)` replaces an INT32 register with the **exclusive** prefix count of
          a predicate -- how many earlier lanes were active, not including this one. It is not a
          host reshape and not a data-value decompaction. Both rows run the same predicate over
          different old data, so a result that depended on what was in the register would differ
          between them.

Every declared lane is published and compared bitwise from a destination poisoned with -777.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_spr, prefix
from reference import make_inputs, reference

DEVICE = "a5"

LANES = 64
POISON = -777
MARK = 7  # what an active lane selects in the SPR rows, so a lane's state is a value not a bit

OUTPUTS = ("o",)

CASES = [
    {"id": "spr_0", "seed": 10000, "block_dim": 1,
     "purpose": "Count 0 with pattern 0: both temporary masks are empty, so the first two rows "
                "are legitimately all zero and only the third -- read after reset_mask -- is all "
                "lanes. A row that was never published would also be all zero, which is why the "
                "destination starts at -777 instead",
     "parameters": {"mode": "spr", "count": 0, "pattern": 0, "output_rows": 3}},
    {"id": "spr_1", "seed": 10001, "block_dim": 1,
     "purpose": "Count 1, and a pattern with only bit 0 and bit 63 set: the two ends of the "
                "64-bit word, so a pattern read that dropped the high half or sign-extended it "
                "shows up in one lane at each end",
     "parameters": {"mode": "spr", "count": 1, "pattern": 2**63 + 1, "output_rows": 3}},
    {"id": "spr_20", "seed": 10002, "block_dim": 1,
     "purpose": "Count 20 against pattern 0xF0F -- four lanes set, four clear, four set. No "
                "count-based mask can produce that shape, which is what makes the two SPR "
                "spellings distinguishable rather than two ways of saying the same thing",
     "parameters": {"mode": "spr", "count": 20, "pattern": 3855, "output_rows": 3}},
    {"id": "spr_64", "seed": 10003, "block_dim": 1,
     "purpose": "Count 64 and a pattern of all ones: both temporary masks already are the full "
                "mask, so this is the one case where forgetting reset_mask entirely would still "
                "pass -- it is here to be read alongside spr_20, not on its own",
     "parameters": {"mode": "spr", "count": 64, "pattern": 2**64 - 1, "output_rows": 3}},
    {"id": "prefix_empty", "seed": 10010, "block_dim": 1,
     "purpose": "No active lane: the exclusive prefix is zero everywhere, which is also what an "
                "unsqueeze that did nothing would leave. Zero replacement is therefore not a "
                "useful control for this case; perturbation and the inclusive-prefix control are",
     "parameters": {"mode": "prefix", "active": [], "output_rows": 2}},
    {"id": "prefix_full", "seed": 10011, "block_dim": 1,
     "purpose": "Every lane active: the prefix is 0, 1, 2, ... 63, the one case where the answer "
                "is the lane index and an off-by-one at either end is unmistakable",
     "parameters": {"mode": "prefix", "active": list(range(LANES)), "output_rows": 2}},
    {"id": "prefix_sparse", "seed": 10012, "block_dim": 1,
     "purpose": "Lanes 1, 3 and 4 active: the prefix stalls, steps, steps twice in a row, then "
                "stays -- a shape that separates a running count from a lane index",
     "parameters": {"mode": "prefix", "active": [1, 3, 4], "output_rows": 2}},
    {"id": "prefix_strided", "seed": 10013, "block_dim": 1,
     "purpose": "Every second lane of the first eleven: the prefix increments on exactly half of "
                "them, so a count that advanced on inactive lanes doubles",
     "parameters": {"mode": "prefix", "active": [0, 2, 4, 6, 8, 10], "output_rows": 2}},
    {"id": "prefix_last", "seed": 10014, "block_dim": 1,
     "purpose": "Only lane 63 active: the exclusive prefix is zero in every lane, the active one "
                "included. That is the definition of exclusive, and it is the case that "
                "distinguishes it from an inclusive count",
     "parameters": {"mode": "prefix", "active": [63], "output_rows": 2}},
    {"id": "prefix_mixed", "seed": 10015, "block_dim": 1,
     "purpose": "Ten lanes spread across both ends and the middle, including lane 0 and lane 63: "
                "the boundary pattern, where a prefix that special-cased either edge disagrees",
     "parameters": {"mode": "prefix",
                    "active": [0, 1, 7, 8, 31, 32, 33, 47, 62, 63], "output_rows": 2}},
]


def check_domain(inputs, expected):
    """The assertions that say what each mode's rows mean, checked rather than promised.

    For `spr` the rows are a mask rendered as MARK or zero, so their sums are the set-lane counts
    the case declared -- the count for row 0, the pattern's popcount for row 1, and all lanes for
    row 2, which is the restoration. For `prefix` the two rows must agree (the old data is
    irrelevant), start at zero (exclusive), and never step by more than one.
    """
    values = expected["o"]
    if (values == POISON).any():
        raise ValueError("the reference contains the poison value")
    if inputs["mode"] == "spr":
        if set(values.flatten().tolist()) - {0, MARK}:
            raise ValueError(f"the SPR rows must be {MARK} or 0 in every lane")
        if int(values[0].sum()) != inputs["count"] * MARK:
            raise ValueError("row 0 must have one marked lane per counted lane")
        if int(values[1].sum()) != inputs["pattern"].bit_count() * MARK:
            raise ValueError("row 1 must have one marked lane per set pattern bit")
        if values[2].tolist() != [MARK] * LANES:
            raise ValueError("row 2 must be the restored full mask")
    else:
        if values[0].tolist() != values[1].tolist():
            raise ValueError("both prefix rows must agree; the old register contents are ignored")
        if int(values[0, 0]) != 0:
            raise ValueError("an exclusive prefix starts at zero")
        if set((values[0, 1:] - values[0, :-1]).tolist()) - {0, 1}:
            raise ValueError("an exclusive prefix steps by 0 or 1")


def execute(case, inputs, launcher, backend):
    """One launch. The two modes have different signatures, and both destinations arrive filled
    with -777 and seeded in -- which is what separates a row that was never published from a row of
    legitimate zeros, and legitimate zeros are the expected answer in three of the ten cases."""
    rows = case["parameters"]["output_rows"]
    if inputs["mode"] == "spr":
        entry = make_spr(inputs["count"], inputs["pattern"])
        arguments = (inputs["dummy"],)
    else:
        entry = prefix
        arguments = (inputs["values"], inputs["selector"])
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(*arguments, torch.full((rows, LANES), POISON, dtype=torch.int32))}


def compare(name, got, want):
    """Bitwise: every lane is an exact integer predicate or an exclusive Boolean prefix count."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.shape[0]} rows x "
          f"{got.shape[1]} lanes")
    if not ok:
        for row in (got != want).any(dim=1).nonzero().flatten().tolist():
            lanes = (got[row] != want[row]).nonzero().flatten().tolist()
            unwritten = bool((got[row] == POISON).all())
            print(f"      row {row}: {len(lanes)} lanes differ at {lanes[:6]}, got "
                  f"{got[row][lanes[:4]].tolist()} want {want[row][lanes[:4]].tolist()}"
                  + ("  (the whole row still holds the poison: never published)"
                     if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--mode", default="all", choices=("all", "spr", "prefix"),
                        help="run only one of the two behaviours")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:15s} {case['purpose']}")
        return 0

    selected = [case for case in CASES
                if args.case in ("all", case["id"])
                and args.mode in ("all", case["parameters"]["mode"])]
    if not selected:
        parser.error(f"no case matches --case {args.case!r} --mode {args.mode!r}; "
                     f"--list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        detail = (f"count={p['count']} pattern=0x{p['pattern']:x}" if p["mode"] == "spr"
                  else f"{len(p['active'])} active lanes")
        print(f"{case['id']}  ({p['mode']}: {detail}, launcher={args.launcher})")
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
