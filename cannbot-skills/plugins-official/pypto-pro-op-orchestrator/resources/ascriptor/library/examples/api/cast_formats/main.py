# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""64-bit casts that publish only their defined lanes, and packed INT4 in both directions.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case i4_widen_8841     # one of them
    python main.py --variant b64_float      # one variant's cases
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

Four variants, each its own kernel:

  b64_widen   i32 to i64, and i64 to i32. The widening reaches one destination register, so only
              source lanes 0..31 are defined; the narrowing is the argument-free form, which
              discards high bits independently of CTRL (D-233).
  b64_float   i64 to f32 and f32 to i64, both round-toward-zero.
  i4_narrow   f16 to i4 (ties to even, saturating) and i16 to i4 (saturating), each packed four
              nibbles to the byte by `pack4()`.
  i4_widen    `ub_to_reg_unpack4` then i4 to f16, bf16 and i16.

A single-register 64-bit form publishes its first 32 logical lanes and nothing else: the stores go
through a `LOWEST32` mask so that the undefined second source register is never read. Getting that
wrong produces plausible-looking garbage in the upper half, which is why the domain says so and the
outputs stop where they do.

Everything is compared as `uint8` carriers, so the low bits of wide integers, the nibble order
inside a packed byte and the exact float rounding are all covered by one comparison.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

import kernel
from reference import make_inputs, reference

DEVICE = "a5"

POISON = 97

OUTPUTS = ("carriers",)

# Each variant's destinations, in the order its kernel returns them.
DESTINATIONS = {
    "b64_widen": [(torch.int64, (4, 32)), (torch.int32, (4, 32))],
    "b64_float": [(torch.float32, (4, 32)), (torch.int64, (4, 32))],
    "i4_narrow": [(torch.uint8, (4, 64)), (torch.uint8, (4, 64))],
    "i4_widen": [(torch.float16, (4, 128)), (torch.bfloat16, (4, 128)), (torch.int16, (4, 128))],
}

VARIANTS = [
    ("b64_widen", 1536,
     "i32 widened to i64 and i64 narrowed back. The widening's source lanes 32..63 have nowhere "
     "to go in one destination register, so only the first 32 are published; the narrowing is the "
     "argument-free form, which drops high bits whatever CTRL says (D-233). The inputs include "
     "+/-268435472 and 2^30 - 1, magnitudes where a lost high word is unmistakable"),
    ("b64_float", 1536,
     "i64 to f32 and f32 to i64, both toward zero. The i64 inputs reach 2^40, past FP32's exact "
     "range, so the direction of the rounding is observable -- the reference has to correct "
     "Torch's nearest-rounding with `nextafter` to state it. The float inputs include 2^33, which "
     "no int32 could hold"),
    ("i4_narrow", 512,
     "f16 and i16 each narrowed to i4 and packed four nibbles per byte. The f16 path rounds ties "
     "to even and saturates; the i16 path only saturates. Inputs ramp through -16..15, so the "
     "saturation to [-8, 7] bites on half of every row"),
    ("i4_widen", 3072,
     "Packed i4 carriers unpacked and widened to f16, bf16 and i16 from one register. The carrier "
     "bytes include 0x8F and 0x70 -- a negative nibble beside a positive one in the same byte -- "
     "so a widening that took the nibbles in the wrong order disagrees in every lane"),
]

CASES = [
    {"id": f"{variant}_{seed}", "seed": seed, "block_dim": 1,
     "purpose": (why if seed == 8841 else
                 f"The same {variant} conversions over a shifted input, which is what shows the "
                 f"lane counts, the rounding mode and the nibble order belong to the kernel "
                 f"rather than to these values"),
     "parameters": {"variant": variant, "output_bytes": size}}
    for variant, size, why in VARIANTS
    for seed in (8841, 8842)
]


def check_domain(inputs, expected):
    """The declared value domain, which is narrower than the instructions': finite values, int64
    magnitudes below 2^40, and float-to-int64 inputs strictly inside int64. Nothing here promises
    anything about a NaN or an overflowing conversion."""
    variant, operands = inputs["variant"], inputs["operands"]
    if variant not in DESTINATIONS:
        raise ValueError(f"unknown variant {variant!r}")
    for operand in operands:
        if operand.is_floating_point() and not bool(torch.isfinite(operand).all()):
            raise ValueError("this example's domain is finite values")
        if operand.dtype == torch.int64 and bool((operand.abs() >= 2**40).any()):
            raise ValueError("the int64 inputs stay below 2^40 in magnitude")
    size = sum(torch.empty(shape, dtype=dtype).nbytes for dtype, shape in DESTINATIONS[variant])
    if expected["carriers"].numel() != size:
        raise ValueError(f"the {variant} reference is {expected['carriers'].numel()} bytes, "
                         f"not the {size} its destinations hold")


def execute(case, inputs, launcher, backend):
    """One launch per case, every destination filled with 97 and seeded in."""
    variant = inputs["variant"]
    op = OpExec(getattr(kernel, "cast_" + variant), launcher=launcher, backend=backend,
                device=DEVICE, block_dim=case["block_dim"],
                out_dir=f"tmp/{launcher}/{case['id']}", seed_outputs=True)
    poisoned = [torch.full(shape, POISON, dtype=dtype)
                for dtype, shape in DESTINATIONS[variant]]
    produced = op(*inputs["operands"], *poisoned)
    values = produced if isinstance(produced, (tuple, list)) else (produced,)
    return {"carriers": torch.cat([value.contiguous().view(torch.uint8).flatten()
                                   for value in values])}


def compare(name, got, want):
    """Bitwise over the concatenated carriers."""
    if got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.numel()} bytes != {want.numel()} bytes")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} bytes")
    if not ok:
        index = (got != want).nonzero().flatten().tolist()
        unwritten = [i for i in index if got[i].item() == POISON]
        print(f"      {len(index)}/{got.numel()} bytes differ, first at {index[:6]}: got "
              f"{[hex(got[i].item()) for i in index[:4]]} want "
              f"{[hex(want[i].item()) for i in index[:4]]}"
              + (f"; {len(unwritten)} still hold the 97 fill (never written)"
                 if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--variant", default="all", choices=("all", *DESTINATIONS))
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:20s} {case['purpose']}")
        return 0

    selected = [case for case in CASES
                if args.case in ("all", case["id"])
                and args.variant in ("all", case["parameters"]["variant"])]
    if not selected:
        parser.error(f"no case matches --case {args.case!r} --variant {args.variant!r}")
    failed = []
    for case in selected:
        print(f"{case['id']}  (variant={case['parameters']['variant']}, "
              f"{case['parameters']['output_bytes']} bytes, launcher={args.launcher})")
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
