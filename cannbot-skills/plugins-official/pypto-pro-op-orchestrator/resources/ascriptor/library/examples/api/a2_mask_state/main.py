# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A manual A2/A3 mask, a counted operation over it, and what the count leaves behind.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --device a3           # the same source against the A3 facade
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

Three stages, three outputs, one launch:

  masked    `set_mask(0, 7)` activates three lanes, then `adds(..., repeat=1)` writes only those.
            The other 125 lanes keep the -9.0 the destination was initialized with.
  counted   `muls(..., count=70)` overrides that mask for 70 lanes; the rest keep -7.0.
  normal    an uncounted `adds(..., repeat=2)` after the counted one, which must cover all 128.

The third output is the one that says something surprising: a counted operation returns to the
**normal** all-lane mask, not to the manual three-lane mask that was set before it. If it restored
the manual mask, `normal` would hold its initialization from lane 3 onwards.

Both inactive tails are compared against their explicit initialization, and the arithmetic is
integer-valued FP32, so the comparison is bitwise.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_masks
from reference import make_inputs, reference

DEVICES = ("a2", "a3")

LANES = 128
MASK_LANES = 3      # set_mask(0, 7): bits 0, 1 and 2
COUNTED_LANES = 70
FILLS = {"masked": -9.0, "counted": -7.0}

OUTPUTS = ("masked", "counted", "normal")

CASES = [
    {"id": "count70", "seed": 8961, "block_dim": 1,
     "purpose": f"{MASK_LANES} manually masked lanes, then a {COUNTED_LANES}-lane counted "
                f"operation, then an uncounted repeat-2 that must cover all {LANES}. A counted "
                f"operation that restored the previous three-lane mask instead of the normal one "
                f"leaves the third output at its initialization from lane {MASK_LANES} onwards",
     "parameters": {}},
    {"id": "different_values", "seed": 8962, "block_dim": 1,
     "purpose": "The same three stages over different values, which is what shows the lane counts "
                "belong to the kernel rather than to the data",
     "parameters": {}},
]


def check_domain(inputs, expected):
    """What the three outputs claim, restated from the lane counts rather than copied from the
    reference: each inactive tail holds its own fill, and the third output has no tail at all."""
    x = inputs["x"]
    if x.shape != (1, LANES) or x.dtype != torch.float32:
        raise ValueError(f"the input must be float32[1, {LANES}]")
    if not torch.equal(x, x.round()):
        raise ValueError("integer-valued inputs are what make the comparison bitwise")
    if not bool((expected["masked"][:, MASK_LANES:] == FILLS["masked"]).all()):
        raise ValueError(f"the masked tail must be {FILLS['masked']}")
    if not bool((expected["counted"][:, COUNTED_LANES:] == FILLS["counted"]).all()):
        raise ValueError(f"the counted tail must be {FILLS['counted']}")
    # `normal` has no fill and no inactive tail: its destination is never `dup`ed, so a lane the
    # third operation did not write reads back as the NaN poison. That is what a restored manual
    # mask would look like here -- not a sentinel value, which x + 1 could legitimately produce.
    if not torch.equal(expected["normal"], x + 1.0):
        raise ValueError("the normal output is the whole input plus one, with no inactive tail")


def execute(case, inputs, launcher, backend, device):
    """One launch for all three outputs. Each destination arrives NaN-poisoned and is seeded in --
    note that the kernel then writes its own fills over them with `dup`, so a NaN in the result
    means the fill itself never landed, which is a different failure from a wrong mask."""
    op = OpExec(make_masks(device), launcher=launcher, backend=backend, device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    poisoned = [torch.full_like(inputs["x"], float("nan")) for _ in OUTPUTS]
    return dict(zip(OUTPUTS, op(inputs["x"], *poisoned), strict=True))


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise: integer-valued FP32 arithmetic and explicit sentinels in the inactive lanes."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:8s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:8s} {'ok  ' if ok else 'FAIL'}  bitwise over {LANES} lanes")
    if not ok:
        got, want = got.cpu().flatten(), want.cpu().flatten()
        index = (got != want).nonzero().flatten().tolist()
        # For the two masked outputs a differing lane still holding its fill means a mask kept it
        # inactive. For `normal` there is no fill, so an unwritten lane is the NaN poison instead.
        stale = ([lane for lane in index[:6] if float(got[lane]) == FILLS[name]] if name in FILLS
                 else [lane for lane in index[:6] if got[lane] != got[lane]])
        print(f"      {len(index)}/{LANES} lanes differ, first at {index[:6]}"
              + (f"; lanes {stale} were never written"
                 f"{' (still the fill)' if name in FILLS else ' (still NaN-poisoned)'}"
                 if stale else "")
              + ("; they start at the manual mask's width, which is what a counted operation "
                 "restoring the manual mask instead of the normal one looks like"
                 if index and index[0] == MASK_LANES and name == "normal" else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--device", default=DEVICES[0], choices=DEVICES)
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
        print(f"{case['id']}  (mask={MASK_LANES} lanes, count={COUNTED_LANES}, then all {LANES}, "
              f"device={args.device}, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend, args.device)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
