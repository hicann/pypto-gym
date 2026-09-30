# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Concatenate three ragged GMList members on the device, eight rows at a time.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case ragged         # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

`GMList[f32, ("?", 64)]` is one argument holding several separately allocated members whose row
counts differ. The descriptor carries each member's extent, so `for t in xs:` iterates them on the
device and `t.shape[0]` is that member's own row count -- there is no host loop unrolling the
members and no padding to a common shape.

Each member moves through one 8x64 UB staging tile, and the destination row cursor `row` advances
across member boundaries, which is what makes the result a concatenation rather than three
independent copies. The output row count arrives as the runtime scalar `N`.

The comparison is bitwise against `torch.cat`: these are transfers, so a difference would be a
lost or misplaced burst rather than a rounding question.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import list_concat
from reference import make_inputs, reference

DEVICE = "a5"

COLUMNS = 64
TILE_ROWS = 8  # the staging tile, and therefore the granularity every member's rows must divide

OUTPUTS = ("o",)

CASES = [
    {"id": "ragged", "seed": 8721, "block_dim": 1,
     "purpose": "16, 32 and 8 rows: the members are neither equal nor ordered by size, and the "
                "shortest one is exactly one staging tile, so a body that assumed a uniform "
                "member extent lands its second and third members in the wrong rows",
     "parameters": {"R0": 16, "R1": 32, "R2": 8, "N": 56}},
    {"id": "different_members", "seed": 8722, "block_dim": 1,
     "purpose": "8, 16 and 24 rows summing to a different N: the member extents come from the "
                "descriptor rather than from a constant, and this case is what shows the same "
                "kernel source follows them",
     "parameters": {"R0": 8, "R1": 16, "R2": 24, "N": 48}},
]


def check_domain(inputs, expected):
    """What the device loop assumes of the members: the staging tile is eight rows, so an extent
    that is not a multiple of eight would leave a partial burst with no tail path, and a
    non-contiguous member would make `t[r:r+8, :]` a strided read instead of a copy."""
    xs = inputs["xs"]
    if not isinstance(xs, list) or len(xs) != 3:
        raise ValueError("this example declares three members")
    for index, member in enumerate(xs):
        if (member.ndim != 2 or member.shape[1] != COLUMNS or member.shape[0] <= 0
                or member.shape[0] % TILE_ROWS or member.dtype != torch.float32
                or not member.is_contiguous()):
            raise ValueError(f"member {index} must be a contiguous float32[k*{TILE_ROWS}, "
                             f"{COLUMNS}] with k >= 1, got {member.dtype}{tuple(member.shape)}")
    if expected["o"].shape != (sum(m.shape[0] for m in xs), COLUMNS):
        raise ValueError("the reference must be as tall as the members put together")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives NaN-poisoned and is seeded into the launch, so a row no
    burst reached -- the failure a row cursor that stopped advancing across a member boundary
    produces -- reads back as NaN rather than as a plausible value."""
    rows = sum(member.shape[0] for member in inputs["xs"])
    op = OpExec(list_concat, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["xs"], torch.full((rows, COLUMNS), float("nan")), rows)}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: every output byte is a copied input byte."""
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
        print(f"{case['id']}  (members={[p['R0'], p['R1'], p['R2']]} -> {p['N']} rows, "
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
