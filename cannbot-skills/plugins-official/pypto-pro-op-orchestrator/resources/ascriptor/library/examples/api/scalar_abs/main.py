# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Take a dynamic GM value's magnitude in its own width, and compare every stored byte.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case f32            # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

One case per width `api.scalar_abs` accepts. The kernel walks a 2x8 GM tile one element at a
time: `Var.GetValueFrom` reads the element, `api.scalar_abs` takes its magnitude, and
`SetValueTo` stores it back in the dtype the case declares -- so the result carries the case's
own width rather than whatever the body computed in.

The comparison is bitwise, which is the point of this unit. `abs` is precisely the operation
that has to turn -0.0 into +0.0, and `torch.equal` reads those two as the same number: a kernel
that stored its source unchanged would pass a value comparison on every float case and fail
this one on two elements of each. The signed integer minimum is kept out of the inputs on
purpose -- its magnitude is not representable in its own width, so there is no right answer to
compare against.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import entry_for
from reference import HOST_DTYPES, make_inputs, reference

DEVICE = "a5"

# pypto_pro: refused at emission, not a wrong answer. A5-UP-001 -- the inspected PyPTO Pro
# ordinary scalar API has no spelling for `scalar.abs`, so `--backend pypto_pro` cannot build
# this kernel for any case. `cce` and `pto_isa` both run all six on a card.

OUTPUTS = ("out",)

CASES = [
    {"id": "i8", "seed": 106201, "block_dim": 1,
     "purpose": "The narrowest width: +/-127 is the whole reachable magnitude, so a body that "
                "computed wide and stored back without narrowing wraps somewhere visible",
     "parameters": {"kind": "i8"}},
    {"id": "i16", "seed": 106201, "block_dim": 1,
     "purpose": "16-bit: the same sixteen values scaled to this width, including both "
                "neighbours of the maximum",
     "parameters": {"kind": "i16"}},
    {"id": "i32", "seed": 106201, "block_dim": 1,
     "purpose": "32-bit: the width the C spelling reaches with its plain `::abs` overload",
     "parameters": {"kind": "i32"}},
    {"id": "i64", "seed": 106201, "block_dim": 1,
     "purpose": "64-bit: the one integer width whose magnitude an `int` overload of `abs` would "
                "silently truncate, so +/-2147483648 and beyond are what this case carries",
     "parameters": {"kind": "i64"}},
    {"id": "f16", "seed": 106201, "block_dim": 1,
     "purpose": "FP16 is not a native A5 scalar width: the backend widens to float, takes the "
                "magnitude and narrows back with an explicit `(half)` cast, and this is the "
                "case that checks that round trip keeps the tiny and infinite endpoints",
     "parameters": {"kind": "f16"}},
    {"id": "f32", "seed": 106201, "block_dim": 1,
     "purpose": "The native float width, carrying both zero signs, both infinities and the "
                "normal minimum: `abs(-0.0)` has to store +0.0, which only a byte comparison "
                "can tell apart from storing -0.0",
     "parameters": {"kind": "f32"}},
]


def check_domain(inputs, expected):
    """The two statements about the inputs and the reference that the cases rely on.

    Checked rather than promised, because both are easy to break while editing `reference.py`
    and neither shows up as a comparison failure: a NaN payload would make an exact-bit
    comparison meaningless, the signed minimum has no representable magnitude at all, and a
    reference that returned a widened result would compare a different dtype than the kernel
    stores.
    """
    source = inputs["source"]
    dtype = HOST_DTYPES[inputs["kind"]]
    if source.dtype != dtype or tuple(source.shape) != (2, 8):
        raise ValueError("the source must retain its case dtype and 2x8 shape")
    if dtype.is_floating_point:
        if torch.isnan(source).any():
            raise ValueError("NaN payloads are outside this example's exact-bit domain")
    elif (source == torch.iinfo(dtype).min).any():
        raise ValueError("the signed minimum has no representable positive magnitude")
    if expected["out"].dtype != source.dtype:
        raise ValueError("the reference must preserve the case's native width")


def execute(case, inputs, launcher, backend):
    """One launch. The destination is handed in poisoned -- NaN for a float case, -99 for an
    integer one -- and seeded into the launch, so an element the kernel never wrote reads back
    as the poison rather than as a plausible magnitude."""
    source = inputs["source"]
    poison = float("nan") if source.dtype.is_floating_point else -99
    op = OpExec(entry_for(inputs["kind"]), launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"out": op(source, torch.full_like(source, poison))}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise. Deliberately not `torch.equal`: it reads +0.0 and -0.0 as equal, and turning
    one into the other is the operation under test."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} elements")
    if not ok:
        # Destinations arrive poisoned, so an element still holding the poison was never written.
        flat_got, flat_want = got.cpu().reshape(-1), want.cpu().reshape(-1)
        outside = raw(got).view(got.numel(), -1).ne(raw(want).view(got.numel(), -1)).any(dim=1)
        index = outside.nonzero().flatten()
        shown = ", ".join(f"[{i}] {flat_got[i].item()!r} != {flat_want[i].item()!r}"
                          for i in index[:3].tolist())
        print(f"      {len(index)}/{got.numel()} elements differ in their bytes; {shown}")
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
        print(f"{case['id']}  (kind={case['parameters']['kind']}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
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
