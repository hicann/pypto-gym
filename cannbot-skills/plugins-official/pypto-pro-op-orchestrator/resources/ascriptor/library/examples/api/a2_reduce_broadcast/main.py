# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A row reduced in two counted stages, then broadcast back to every lane in two more.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case width_256      # one of them
    python main.py --device a3           # the same source against the A3 facade
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

    o[r] = x[r] * sum(x[r])

Four stages, and each pair exists because one call cannot do the job:

  cadd x2   the first reads 64 lanes per repeat and leaves one partial sum per 64-lane group; the
            second reduces those `groups` partials to one total. Its `count_per_rep` is the live
            partial count, not 64.
  brcb x2   `brcb` fills a 32-byte block from a value, so one call turns the total into eight copies
            and the second turns those eight into all 64 lanes of a reusable scale.

The two `cadd` strides count different things, which is the detail worth carrying away: the
destination repeat stride counts result *elements*, the source repeat stride counts 32-byte *blocks*.

Widths 64, 128 and 256 give one, two and four partials. Small integer inputs make the sums and
products exact in FP32, so the comparison is bitwise -- and that is a property of these inputs, not
a reduction tolerance this example is offering for arbitrary floats.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_reduce
from reference import make_inputs, reference

DEVICES = ("a2", "a3")

ROWS = 3
LANES_PER_GROUP = 64

OUTPUTS = ("o", "sums")

CASES = [
    {"id": f"width_{width}", "seed": 8971, "block_dim": 1,
     "purpose": {
         64: "One 64-lane group: the first cadd is a single repeat and the second reduces a single "
             "partial, so this is the width where the two-stage structure is degenerate -- read it "
             "as the baseline the wider cases are compared against",
         128: "Two groups: the first cadd repeats twice with a source stride of 8 blocks and the "
              "second has two live partials, so both stages' counts finally matter. The output "
              "takes two muls against the same broadcast scale, which a scale valid only for the "
              "first group would fail",
         256: "Four groups: the longest reduction here and the largest intermediate partial count, "
              "with four muls sharing one scale",
     }[width],
     "parameters": {"width": width}}
    for width in (64, 128, 256)
]


def check_domain(inputs, expected):
    """What makes the comparison bitwise, and what the `sums` output actually is: the total in lane 0
    of a fully initialized 64-lane buffer, with the other 63 lanes explicitly zero."""
    x = inputs["x"]
    width = inputs["width"]
    if x.shape != (ROWS, width) or x.dtype != torch.float32:
        raise ValueError(f"the input must be float32[{ROWS}, {width}]")
    if not torch.equal(x, x.round()) or bool((x.abs() > 8).any()):
        raise ValueError("small integer values are what make the sums and products exact in FP32")
    if not torch.equal(expected["sums"][:, 0], x.sum(1)):
        raise ValueError("lane 0 of the sums buffer is the row total")
    if expected["sums"][:, 1:].count_nonzero():
        raise ValueError("the rest of the sums buffer is explicitly zeroed scratch")


def execute(case, inputs, launcher, backend, device):
    """One launch over all three rows. Both destinations arrive NaN-poisoned and are seeded in, so a
    scratch lane the kernel's own `dup` never zeroed is distinguishable from a legitimate zero."""
    op = OpExec(make_reduce(inputs["width"], device), launcher=launcher, backend=backend,
                device=device, block_dim=case["block_dim"],
                out_dir=f"tmp/{launcher}/{case['id']}", seed_outputs=True)
    o, sums = op(inputs["x"], torch.full_like(inputs["x"], float("nan")),
                 torch.full((ROWS, LANES_PER_GROUP), float("nan")))
    return {"o": o, "sums": sums}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise over every output row and the whole 64-lane total buffer."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:6s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:6s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.shape[0]}x{got.shape[1]} "
          f"FP32 elements")
    if not ok:
        got, want = got.cpu(), want.cpu()
        for row in (got != want).any(dim=1).nonzero().flatten().tolist():
            lanes = (got[row] != want[row]).nonzero().flatten().tolist()
            poisoned = [c for c in lanes if got[row, c] != got[row, c]]
            # A wrong reduction scales a whole row by the same factor; a wrong broadcast does not.
            ratios = {round(float(got[row, c] / want[row, c]), 4) for c in lanes[:8]
                      if want[row, c] != 0}
            print(f"      row {row}: {len(lanes)} lanes differ at {lanes[:6]}, got/want ratios "
                  f"{sorted(ratios)[:4]}"
                  + ("  (one ratio across the row points at the total, not the broadcast)"
                     if len(ratios) == 1 else "")
                  + (f"; {len(poisoned)} still NaN (never written)" if poisoned else ""))
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
            print(f"{case['id']:11s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        width = case["parameters"]["width"]
        print(f"{case['id']}  ({ROWS}x{width}, {width // LANES_PER_GROUP} partial sums, "
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
