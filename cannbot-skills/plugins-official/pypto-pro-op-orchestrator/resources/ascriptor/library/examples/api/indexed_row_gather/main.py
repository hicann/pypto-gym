# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gather whole rows named by a value in memory, at two row widths, and prove the clamp is wrong.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case short_tail     # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

Sixteen slots name rows of two FP16 tables, 64 and 48 columns wide. A row gather is a scalar read
followed by a copy: `Var.GetValueFrom` puts the slot's index in a cell, the cell becomes the row
subscript of a GM view, and one DMA moves that row. This is the addressing only -- there is no
score, mask, softmax or reduction here, and it is not an attention operator.

The four outputs are three separate statements. `k_rows` and `v_rows` record what every slot
fetched, padding slots included, because the copies are unconditional. `guarded` is written under
`if s < live:` and leaves the tail at the fill. `clamped` writes `min(s, live - 1)`
unconditionally, which keeps the address inside the tensor and loses the last real row to the last
padding slot -- and the reference predicts that corruption, so the trap is proved rather than
described.

Every table value is a multiple of 1/64 below 32 in magnitude, so it is exact in FP16 and the
comparison is bitwise. All four destinations arrive filled with 1024.0, which is outside both
tables' range: a lost store stays visible, and the rows the guard withholds are compared against
the fill by name.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import indexed_row_gather
from reference import DK, DV, GUARD, S2, SLOTS, make_inputs, reference

DEVICE = "a5"

OUTPUTS = ("k_rows", "v_rows", "guarded", "clamped")

# Each destination's row width: the two tables are gathered at their own, and DV is the narrower.
WIDTHS = (DK, DV, DV, DV)

CASES = [
    {"id": "full_block", "seed": 9101, "block_dim": 1,
     "purpose": "All sixteen slots live, so no tail question arises and all three DV outputs "
                "agree: the case that establishes the gather itself before the padding paths",
     "parameters": {"live": 16, "padding": "zero"}},
    {"id": "padded_tail", "seed": 9102, "block_dim": 1,
     "purpose": "Twelve live slots with the four padding ones naming row 127, far from any live "
                "row: the padding is copied like anything else, the guard leaves the tail at the "
                "fill, and the clamp drops row 127's data onto row 11",
     "parameters": {"live": 12, "padding": "last"}},
    {"id": "repeated_padding", "seed": 9103, "block_dim": 1,
     "purpose": "The padding repeats the first live row, so the row label in column zero no "
                "longer distinguishes a padding fetch from a real one -- where the data lands is "
                "the only thing left to compare, which is what the whole-output readback does",
     "parameters": {"live": 12, "padding": "repeat"}},
    {"id": "short_tail", "seed": 9104, "block_dim": 1,
     "purpose": "Five live slots out of sixteen, the shortest tail here: twelve clamped writes "
                "pile onto row 4 and only the last one survives, which is the corruption at its "
                "largest",
     "parameters": {"live": 5, "padding": "zero"}},
]


def check_domain(inputs, expected):
    """The two statements the gather rests on. Every slot is dereferenced, so a padding index that
    was not a legal row would be an out-of-range read rather than an unread result; and the live
    count has to be a real slot count, because the guard and the clamp are both derived from it."""
    live = int(inputs["count"][0, 0])
    if not 1 <= live <= SLOTS:
        raise ValueError(f"the live row count must be in [1, {SLOTS}], got {live}")
    if bool(((inputs["index"] < 0) | (inputs["index"] >= S2)).any()):
        raise ValueError(f"every slot, padding included, must name a row in [0, {S2})")
    if expected["guarded"][live:].ne(GUARD).any():
        raise ValueError("the reference must leave the guarded tail at the fill value")


def execute(case, inputs, launcher, backend):
    """One launch for all four destinations, each pre-filled with the out-of-band guard value and
    seeded into the launch -- so a row no store reached reads back as 1024.0, which no table value
    can be, and the guarded tail is compared against it by name rather than being skipped."""
    op = OpExec(indexed_row_gather, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    destinations = [torch.full((SLOTS, width), GUARD, dtype=torch.float16) for width in WIDTHS]
    produced = op(inputs["index"], inputs["count"], inputs["k_table"], inputs["v_table"],
                  *destinations)
    return dict(zip(OUTPUTS, produced, strict=True))


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise. Every table value is exact in FP16 and the kernel only copies rows, so an
    inexact comparison here would only be hiding a wrong index or a lost store."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:8s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:8s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.shape[0]} rows x "
          f"{got.shape[1]} FP16 columns")
    if not ok:
        rows = raw(got).view(got.shape[0], -1).ne(raw(want).view(got.shape[0], -1)).any(dim=1)
        index = rows.nonzero().flatten().tolist()
        # Column zero labels the row a value came from, so it says which row was fetched instead.
        labels = [(slot, got[slot, 0].item(), want[slot, 0].item()) for slot in index[:4]]
        still_filled = [slot for slot in index if bool((got[slot] == GUARD).all())]
        print(f"      rows {index} differ; (slot, got row label, wanted row label) {labels}"
              + (f"; {still_filled} still hold the {GUARD} fill (never written)"
                 if still_filled else ""))
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
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (live={p['live']}/{SLOTS}, padding={p['padding']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
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
