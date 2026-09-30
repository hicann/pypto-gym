# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Split one tall input into three separately allocated GMList members, eight rows at a time.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case ragged         # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

The mirror image of `list_concat`: here the `GMList` is the *destination*. One argument holds three
members with different row counts, the caller allocates each one separately, and the device loop
`for t in ys:` walks them with `t.shape[0]` as that member's extent. The source cursor `row`
advances across member boundaries, so the members receive consecutive slices of the input rather
than each starting at row zero.

Every member is compared, none is folded into another, and the comparison is bitwise against plain
host slicing.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import list_split
from reference import make_inputs, reference

DEVICE = "a5"

COLUMNS = 64
TILE_ROWS = 8  # the staging tile, and therefore the granularity every member's rows must divide

OUTPUTS = ("y0", "y1", "y2")

CASES = [
    {"id": "ragged", "seed": 8731, "block_dim": 1,
     "purpose": "16, 32 and 8 rows out of 56: the members are neither equal nor ordered by size, "
                "and the last one is a single staging tile, so a source cursor that reset per "
                "member would hand every member the input's first rows",
     "parameters": {"R0": 16, "R1": 32, "R2": 8, "N": 56}},
    {"id": "different_members", "seed": 8732, "block_dim": 1,
     "purpose": "8, 16 and 24 rows out of 48: the same kernel source over a different partition, "
                "which is what shows the member extents are read from the descriptor",
     "parameters": {"R0": 8, "R1": 16, "R2": 24, "N": 48}},
]


def check_domain(inputs, expected):
    """What the device loop assumes: the staging tile is eight rows, the extents have to sum to
    the input's height or some rows would go nowhere, and the input has to be contiguous for
    `x[row:row+8, :]` to be a copy rather than a strided read."""
    rows, x = inputs["rows"], inputs["x"]
    if len(rows) != 3 or any(r <= 0 or r % TILE_ROWS for r in rows):
        raise ValueError(f"the three member extents must be positive multiples of {TILE_ROWS}, "
                         f"got {rows}")
    if x.shape != (sum(rows), COLUMNS) or x.dtype != torch.float32 or not x.is_contiguous():
        raise ValueError(f"the input must be a contiguous float32[{sum(rows)}, {COLUMNS}]")
    if [expected[name].shape[0] for name in OUTPUTS] != list(rows):
        raise ValueError("each reference member must have its declared extent")


def execute(case, inputs, launcher, backend):
    """One launch. Every member arrives NaN-poisoned and is seeded into the launch, so a member the
    loop never reached -- or rows of it a cursor skipped -- reads back as NaN."""
    op = OpExec(list_split, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    members = [torch.full((rows, COLUMNS), float("nan")) for rows in inputs["rows"]]
    produced = op(inputs["x"], members, inputs["x"].shape[0])
    return dict(zip(OUTPUTS, produced, strict=True))


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: every member byte is an input byte that moved."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.shape[0]} rows x "
          f"{got.shape[1]} columns")
    if not ok:
        rows = raw(got).view(got.shape[0], -1).ne(raw(want).view(got.shape[0], -1)).any(dim=1)
        index = rows.nonzero().flatten().tolist()
        poisoned = [r for r in index if bool(torch.isnan(got[r]).all())]
        print(f"      {len(index)}/{got.shape[0]} rows differ: {index[:8]}"
              + (f"; rows {poisoned[:8]} are still NaN-poisoned (never written)" if poisoned else ""))
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
            print(f"{case['id']:20s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['N']} rows -> members {[p['R0'], p['R1'], p['R2']]}, "
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
