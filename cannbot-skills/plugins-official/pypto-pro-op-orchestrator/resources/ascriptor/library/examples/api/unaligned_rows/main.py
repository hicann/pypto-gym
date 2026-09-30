# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Walk packed rows that do not start on a register boundary, two ways, and compare every cell.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case 7x101_stream      # one of them
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

`rows x width` INT32 cells are packed with no padding, so only row 0 is necessarily 32-byte
aligned and every later row starts at whatever byte phase the width leaves. Two paths read and
write them:

  stream  `unalign_reg_for_load` / `unalign_reg_for_store` state plus a `ub_cursor` per row. The
          load is primed once per row (`ub_to_reg_unalign_pre`), each chunk advances the cursor,
          and the store chain is closed at the end of the row (`reg_to_ub_unalign_post`).
  once    the stateless form: `ub_to_reg_unalign_once` and `reg_to_ub_unalign_once` at an explicit
          element offset, with no cursor to carry between chunks.

The body only adds the row index, so an addressing mistake is not a wrong value -- it is a cell
carrying the wrong row's offset, which the comparison localises by row. The source UB allocation
carries one full register of slack past the payload and rounds its end up, which is what makes a
full-register read legal at a partial final chunk; the destination is exactly the payload, so every
store count has to stay inside its row.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_rows
from reference import make_inputs, reference

DEVICE = "a5"

POISON = 123456  # outside the input's range, so an unwritten cell is not a plausible value

OUTPUTS = ("o",)

# Six geometries, each run through both paths. The `purpose` of a pair says what the geometry is
# for; the mode line says which of the two spellings is walking it.
GEOMETRIES = [
    ((7, 101), "101 cells per row: two register chunks with a 37-wide tail, and six of the seven "
               "rows starting off a register boundary"),
    ((3, 64), "exactly one full register per row, so every row start is 32-byte aligned -- the "
              "control geometry, where the unaligned path has nothing to correct"),
    ((5, 17), "17 cells per row, shorter than a register: every load and store is partial, and "
              "the source's slack is what makes the full-register read legal at all"),
    ((4, 33), "33 cells, just over half a register: one chunk per row, and each row lands at a "
              "different byte phase from the last"),
    ((2, 128), "two full registers per row and no partial chunk anywhere, so a tail-handling bug "
               "cannot hide here -- and neither can it be found here"),
    ((6, 8), "8 cells per row, one 32-byte block, six times: the shortest row, where the cursor "
             "crosses a block boundary on every single row"),
]

CASES = [
    {"id": f"{rows}x{width}_{mode}", "seed": 8951, "block_dim": 1,
     "purpose": f"{why} Walked by the {mode} path.",
     "parameters": {"rows": rows, "width": width, "cells": rows * width, "mode": mode}}
    for (rows, width), why in GEOMETRIES
    for mode in ("stream", "once")
]


def check_domain(inputs, expected):
    """What makes an addressing mistake legible: the poison is outside the input's range, and the
    reference adds only the row index -- so a cell that came from the wrong row is off by the
    difference of two row numbers rather than by an arbitrary amount."""
    p = inputs["parameters"]
    x = inputs["x"]
    if x.shape != (1, p["cells"]) or x.dtype != torch.int32:
        raise ValueError(f"the input must be int32[1, {p['cells']}]")
    if (x == POISON).any():
        raise ValueError(f"the input carries the poison value {POISON}")
    rows = x.reshape(p["rows"], p["width"])
    if not torch.equal(expected["o"].reshape(p["rows"], p["width"]) - rows,
                       torch.arange(p["rows"], dtype=torch.int32).expand(p["width"], -1).T
                       .contiguous()):
        raise ValueError("the reference must add exactly the row index to each row")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives filled with the poison and seeded into the launch, so a
    cell no store covered -- the failure a store count that ran short produces -- reads back as
    123456."""
    p = inputs["parameters"]
    entry = make_rows(p["rows"], p["width"], p["mode"])
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], torch.full_like(inputs["x"], POISON))}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want, geometry=None):
    """Bitwise: integer addition and packed offsets define every stored cell."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} cells")
    if not ok:
        flat_got, flat_want = got.cpu().reshape(-1), want.cpu().reshape(-1)
        index = (flat_got != flat_want).nonzero().flatten().tolist()
        unwritten = [i for i in index if flat_got[i].item() == POISON]
        # The only difference between two rows' values is their row index, so the offset names
        # which row's data arrived in the wrong place.
        offsets = sorted({int(flat_got[i]) - int(flat_want[i]) for i in index
                          if flat_got[i].item() != POISON})[:5]
        where = ([divmod(i, geometry[1]) for i in index[:3]] if geometry else index[:3])
        print(f"      {len(index)}/{got.numel()} cells differ, first at (row, column) {where}; "
              f"row-index offsets seen {offsets}"
              + (f"; {len(unwritten)} cells still hold the poison (never stored)"
                 if unwritten else ""))
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
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['rows']}x{p['width']} cells, mode={p['mode']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name], (p["rows"], p["width"])):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
