# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Store 24 integer and 4 float scalar-helper results, each into its own observed lane.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case n_81           # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card
    python main.py --backend pto_isa --launcher board   # the PTO ISA backend on a card

One kernel, 28 scalar results, one lane each: arithmetic and bit helpers, `Min` / `Max` /
`CeilDiv`, the six `Align*` helpers, the three worker queries, an unsigned right shift, two
spellings of floating division, `scalar_sqrt` and `scalar_abs`. `n` arrives as a runtime `i64`,
so nothing here is constant-folded away by the host.

The integer destination arrives filled with -999 and the float one with NaN, both seeded into the
launch, so a helper whose store never landed is a lane holding the poison rather than a plausible
number. Every value is small exact integer arithmetic or an exactly representable float, so the
comparison is bitwise.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import scalar_math
from reference import make_inputs, reference

DEVICE = "a5"

# pypto_pro: the first refusal is `scalar.sqrt`. A5-UP-002 -- the inspected PyPTO Pro ordinary
# scalar API cannot express it, so the float half of this kernel cannot be emitted for that
# backend; `scalar.abs` is refused for the same reason (A5-UP-001). Its integer bitwise and
# alignment operators are mapped, so the first 24 lanes are not what blocks it. cce and pto_isa
# run both cases on a card.

POISON = {"integers": -999, "floats": float("nan")}

OUTPUTS = ("integers", "floats")

# The lane layout, so a failure can name the helper rather than an index. The order is the
# kernel's `integer_values` list.
INTEGER_LANES = ("add", "sub", "mul", "div", "mod", "and", "or", "xor", "shl", "shr", "inv",
                 "Min", "Max", "CeilDiv", "Align8", "Align16", "Align32", "Align64", "Align128",
                 "Align256", "GetVecNum", "GetVecIdx", "GetSubBlockIdx", "u64 shr 63")
FLOAT_LANES = ("div(1.5, 2.0)", "div(cell, cell)", "sqrt(9.0)", "abs(-n)")

CASES = [
    {"id": "n_37", "seed": 9001, "block_dim": 1,
     "purpose": "37, below 64: the six alignment helpers answer 40, 48, 64, 64, 128, 256, so "
                "Align32 and Align64 agree here and only the other case can separate them",
     "parameters": {"n": 37}},
    {"id": "n_81", "seed": 9001, "block_dim": 1,
     "purpose": "81, above 64: the same six answer 88, 96, 96, 128, 128, 256 -- a different "
                "pattern of agreements, so between the two cases every helper is pinned to its "
                "own modulus rather than to a value that happens to fit",
     "parameters": {"n": 81}},
]


def check_domain(inputs, expected):
    """The declared domain, and the two properties that make the poison meaningful: `n` is a
    positive integer (the division and modulo lanes are only defined for those here), and neither
    poison value appears in the reference."""
    n = inputs["n"]
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if len(INTEGER_LANES) != expected["integers"].numel():
        raise ValueError("the lane names and the integer output disagree in length")
    if (expected["integers"] == POISON["integers"]).any():
        raise ValueError("the reference contains the integer poison value")
    if torch.isnan(expected["floats"]).any():
        raise ValueError("the reference contains NaN, which the float poison uses")


def execute(case, inputs, launcher, backend):
    """One launch producing both destinations, each pre-poisoned and seeded in."""
    op = OpExec(scalar_math, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    integers = torch.full((1, len(INTEGER_LANES)), POISON["integers"], dtype=torch.int64)
    floats = torch.full((1, len(FLOAT_LANES)), POISON["floats"])
    produced = op(integers, floats, inputs["n"])
    return dict(zip(OUTPUTS, produced, strict=True))


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise, and a failure names the helper whose lane disagreed."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    lanes = INTEGER_LANES if name == "integers" else FLOAT_LANES
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {len(lanes)} lanes")
    if not ok:
        flat_got, flat_want = got.cpu().reshape(-1), want.cpu().reshape(-1)
        outside = raw(got).view(got.numel(), -1).ne(raw(want).view(got.numel(), -1)).any(dim=1)
        for index in outside.nonzero().flatten().tolist():
            value = flat_got[index].item()
            poisoned = (value == POISON["integers"] if name == "integers"
                        else value != value)  # NaN is the only value that is not equal to itself
            note = "  (still poisoned: the store never landed)" if poisoned else ""
            print(f"      {lanes[index]:16s} got {value!r} want {flat_want[index].item()!r}{note}")
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
            print(f"{case['id']:6s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (n={case['parameters']['n']}, backend={args.backend}, "
              f"launcher={args.launcher})")
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
