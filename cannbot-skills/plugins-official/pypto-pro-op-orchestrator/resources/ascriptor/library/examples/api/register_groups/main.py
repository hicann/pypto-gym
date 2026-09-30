# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Six two-register groups -- integer, reduction, cast, memory and two complex widths.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case memory_8871    # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

Every register here is declared `reg_num=2`: one logical register spanning two physical ones, 512
bytes. That matters because the masks are too -- a 47-lane tail on an int64 group crosses the
single-register boundary at 32, and a 95-lane one on a complex32 group crosses it at 64. A masked
operation that handled only the first physical register would pass a 32-lane test and fail these.

Each variant is its own kernel, and all of their outputs are compared as one `uint8` carrier
buffer. That is not laziness: the outputs include `complex32` and `uint64`, and a byte comparison
is the only one that covers every defined byte of both complex components, the low bits of integers
past 2^54, and the lanes a mask left inactive.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

import kernel
from reference import make_inputs, reference

DEVICE = "a5"

POISON = 97  # in every destination dtype, so an unwritten byte is not a plausible result

OUTPUTS = ("carriers",)

# Each variant's destinations, in the order its kernel returns them.
DESTINATIONS = {
    "i64": [(torch.int64, (9, 64))],
    "reduce": [(torch.int64, (7, 64))],
    "cast": [(torch.int64, (1, 64)), (torch.int32, (1, 64)),
             (torch.float32, (1, 64)), (torch.int64, (1, 64))],
    "memory": [(torch.uint64, (3, 64))],
    "complex": [(torch.complex32, (4, 128))],
    "complex64": [(torch.complex64, (2, 64))],
}

VARIANTS = [
    ("i64", 4608,
     "Nine rows over two int64 registers: add, subtract, multiply, bitwise and, floor remainder, "
     "a fused multiply-add masked to 47 lanes, a maximum built from compare and select, and two "
     "arange ramps based at +/-18014398509481991 -- past 2^54, so a reference computed in float64 "
     "would lose the low bits this compares"),
    ("reduce", 3584,
     "Seven rows: sum, maximum and minimum in lane 0 with the rest zero, an interleave of the two "
     "inputs, and a deinterleave whose outputs alias its own inputs and must still recover both "
     "halves"),
    ("cast", 1536,
     "Four conversions keeping 64 logical lanes while the physical width changes between 256 and "
     "512 bytes: i32 widened to i64, narrowed back, widened to f32, and an f32 truncated toward "
     "zero into i64"),
    ("memory", 1536,
     "Three uint64 rows that separate two mask semantics: a masked gather clears the lanes it did "
     "not fill, while a masked scatter leaves untouched destination lanes alone -- so its "
     "destination is explicitly zeroed first. The third row is a logical right shift of inputs "
     "whose high bit is set"),
    ("complex", 2048,
     "Four complex32 rows over 128 lanes: a sum, a product masked to 95 lanes (crossing the "
     "one-register boundary at 64), a sum with a complex literal, and a product masked by two "
     "woven predicates"),
    ("complex64", 1024,
     "Two complex64 rows over 64 lanes: a sum, and a product masked to 47 -- the same shape as "
     "the complex32 variant at twice the component width, so a mask that counted bytes instead of "
     "lanes disagrees with exactly one of the two"),
]

CASES = [
    {"id": f"{variant}_{seed}", "seed": seed, "block_dim": 1,
     "purpose": (why if seed == 8871 else
                 f"The same {variant} group over a shifted input, which is what shows the lane "
                 f"counts, the mask extents and the ramp bases are properties of the kernel "
                 f"rather than of these values"),
     "parameters": {"variant": variant, "output_bytes": size}}
    for variant, size, why in VARIANTS
    for seed in (8871, 8872)
]


def check_domain(inputs, expected):
    """What each variant's comparison rests on, checked rather than promised."""
    variant, operands = inputs["variant"], inputs["operands"]
    if variant not in DESTINATIONS:
        raise ValueError(f"unknown variant {variant!r}")
    if variant in ("i64", "reduce"):
        # The ramps are based past 2^54 on purpose; a float64 reference would round them.
        if not any(abs(int(value)) > 2**32 for value in operands[0].flatten()):
            raise ValueError("the integer inputs must exceed 32 bits somewhere")
    if variant == "memory":
        # Torch has no CPU right shift for uint64, and the same bits read as int64 are negative
        # exactly when the high bit is set -- which is the property being asserted.
        if not bool((operands[0].view(torch.int64) < 0).any()):
            raise ValueError("the uint64 inputs must set the high bit, or the logical shift would "
                             "not be distinguishable from an arithmetic one")
    if variant.startswith("complex"):
        if not bool((operands[0].imag != 0).any()):
            raise ValueError("the complex inputs must have non-zero imaginary parts")
    size = sum(torch.empty(shape, dtype=dtype).nbytes for dtype, shape in DESTINATIONS[variant])
    if expected["carriers"].numel() != size:
        raise ValueError(f"the {variant} reference is {expected['carriers'].numel()} bytes, "
                         f"not the {size} its destinations hold")


def execute(case, inputs, launcher, backend):
    """One launch per case. Every destination arrives filled with 97 and seeded into the launch, so
    a lane a mask left inactive is distinguishable from a lane nothing wrote -- the two are the
    likeliest failures here and they are not the same defect."""
    variant = inputs["variant"]
    entry = getattr(kernel, "groups_" + variant)
    op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    poisoned = [torch.full(shape, POISON, dtype=dtype)
                for dtype, shape in DESTINATIONS[variant]]
    produced = op(*inputs["operands"], *poisoned)
    values = produced if isinstance(produced, (tuple, list)) else (produced,)
    return {"carriers": torch.cat([value.contiguous().view(torch.uint8).flatten()
                                   for value in values])}


def compare(name, got, want):
    """Bitwise over the concatenated carriers: every defined byte of every declared output."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.numel()} bytes != {want.numel()} bytes")
        return False
    ok = torch.equal(got.cpu(), want.cpu())
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} bytes")
    if not ok:
        index = (got.cpu() != want.cpu()).nonzero().flatten().tolist()
        unwritten = [i for i in index if got.cpu()[i].item() == POISON]
        print(f"      {len(index)}/{got.numel()} bytes differ, first at {index[:6]}"
              + (f"; {len(unwritten)} still hold the 97 fill (never written)"
                 if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--variant", default="all", choices=("all", *DESTINATIONS),
                        help="run only the cases of one variant")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES
                if args.case in ("all", case["id"])
                and args.variant in ("all", case["parameters"]["variant"])]
    if not selected:
        parser.error(f"no case matches --case {args.case!r} --variant {args.variant!r}; "
                     f"--list prints them")
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
