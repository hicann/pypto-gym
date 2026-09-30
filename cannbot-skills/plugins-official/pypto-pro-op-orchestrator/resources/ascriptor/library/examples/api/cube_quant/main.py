# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Five fixpipe requantization paths, each with its own dtype triple, rounding and clamp.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case fp32_u8        # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

`product.requant(scale=, offset=)` on the way out of L0C. Which conversion happens is decided by
the dtype triple -- input, accumulator, destination -- and the five cases are the five combinations
this example declares:

    fp32_i8    f16 in, f32 accumulate, signed 8-bit out, scale 0.5, offset 8
    fp32_u8    the same to unsigned 8-bit, so the clamp is [0, 255] instead of [-128, 127]
    int32_i8   i8 in, i32 accumulate, signed 8-bit out, scale 0.5, offset 0
    int32_f16  i32 accumulate dequantized to FP16 with scale 0.25 -- no offset, no clamp
    fp32_f16   f32 accumulate scaled to FP16 with 0.5

The order is scale, then round, then offset, then clamp, and the reference spells it out that way.
Inputs are small integers with rows and columns forced to +8 and -8, so the products reach past both
saturation limits; odd dot products land on halves, which is where ties-to-even is decided -- before
the offset moves them.

Only exact powers of two are used as scales, because those are exact in the hardware's float19
parameter field. An arbitrary scale would need an independent reference for that packing, which this
example does not have.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_quant
from reference import make_inputs, reference

DEVICE = "a5"

POISON = 97
TILE = (32, 32)

OUTPUTS = ("carriers",)

# Each mode's destination dtype, and the scale and offset its kernel applies.
MODES = {
    "fp32_i8": (torch.int8, 0.5, 8,
                "FP16 in, FP32 accumulation, signed 8-bit out. Rounding happens before the offset "
                "of 8, and the planted +8 row against the +8 column carries the product past 127 "
                "so the clamp is reached rather than assumed"),
    "fp32_u8": (torch.uint8, 0.5, 8,
                "The same path to unsigned 8-bit: the clamp becomes [0, 255], so the -8 column "
                "that merely rounds in the signed case is clamped to zero here -- the pair is what "
                "separates the destination's signedness from everything else"),
    "int32_i8": (torch.int8, 0.5, 0,
                 "INT8 in, INT32 accumulation, requantized to signed 8-bit with no offset: the "
                 "integer path, where the dot product is exact before scaling and the 0.5 is the "
                 "only rounding in the case"),
    "int32_f16": (torch.float16, 0.25, 0,
                  "INT32 accumulation dequantized to FP16 with scale 0.25: no offset and no clamp, "
                  "so this is the case that isolates the scale parameter itself"),
    "fp32_f16": (torch.float16, 0.5, 0,
                 "FP32 accumulation scaled to FP16: the float-to-float path, where the only "
                 "questions are the scale and the narrowing"),
}

CASES = [
    {"id": mode, "seed": 8911 + index, "block_dim": 1, "purpose": why,
     "parameters": {"mode": mode, "output_bytes": TILE[0] * TILE[1] *
                    torch.empty((), dtype=dtype).element_size()}}
    for index, (mode, (dtype, _, _, why)) in enumerate(MODES.items())
]


def check_domain(inputs, expected):
    """What the five cases rest on: the saturation limits are actually reached, the scales are exact
    powers of two, and the inputs are small integers so the product is exact before scaling."""
    mode = inputs["mode"]
    x, y = inputs["operands"]
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if tuple(x.shape) != TILE or tuple(y.shape) != TILE:
        raise ValueError(f"the operands must be {TILE}")
    _, scale, offset, _ = MODES[mode]
    if scale not in (0.5, 0.25):
        raise ValueError("only exact powers of two are declared here; an arbitrary scale needs an "
                         "independent float19 parameter-packing reference")
    product = x.float() @ y.float().T
    if not torch.equal(product, product.round()):
        raise ValueError("the dot products must be exact integers before scaling")
    scaled = product * scale + offset
    if not mode.endswith("f16") and not bool(((scaled > 127) | (scaled < -128)).any()):
        raise ValueError("no product reaches a saturation limit, so the clamp is untested")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives filled with 97 -- a value in range for every one of the
    five destination dtypes -- and seeded in, so an element the fixpipe never wrote is visible."""
    mode = inputs["mode"]
    dtype = MODES[mode][0]
    op = OpExec(make_quant(mode), launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    produced = op(*inputs["operands"], torch.full(TILE, POISON, dtype=dtype))
    return {"carriers": produced.contiguous().view(torch.uint8).flatten()}


def compare(name, got, want):
    """Bitwise over the destination's bytes, whatever its dtype."""
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
              f"{[got[i].item() for i in index[:4]]} want {[want[i].item() for i in index[:4]]}"
              + (f"; {len(unwritten)} still hold the 97 fill (never written)"
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
            print(f"{case['id']:11s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        mode = case["parameters"]["mode"]
        dtype, scale, offset, _ = MODES[mode]
        print(f"{case['id']}  (-> {str(dtype).removeprefix('torch.')}, scale={scale}, "
              f"offset={offset}, launcher={args.launcher})")
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
