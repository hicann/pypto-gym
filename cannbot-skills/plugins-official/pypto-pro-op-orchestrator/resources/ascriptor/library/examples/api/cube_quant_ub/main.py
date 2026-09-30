# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Quantized L0C-to-UB transfers, and a cube-to-vector ownership handoff that publishes once.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case split_i8       # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

A5 cube computes `x[M,K] @ y[N,K].T`, the L0C result is converted straight into a **vector** UB
allocation, and subblock 0 publishes it to GM. `CvMutex(..., src_end_pipe=Pipe.FIX,
dst_end_pipe=Pipe.V)` is the bridge, in four beats -- lock, ready, wait, free -- and both vector
participants run all four while only subblock 0 writes GM. That is the whole ownership handoff.

`.requant(...).subblk(0)` is the conversion. The subblock rider is not optional: the sources these
cases came from omitted it even though their contract described a single subblock, and it is now
required.

The byte quantization has two saturations, in this order: scale (truncated to FixP19), round to
nearest even, clamp to the intermediate signed 9-bit range [-256, 255], add the offset, then
saturate to the destination's own range. The reference states each step, and a separate scalar
implementation checks every byte of it.

Operands are integers in [-8, 8] and the scales are exact powers of two, so the products and the
FP16 results are exact and the comparison is bitwise over the destination's bytes.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import ENTRIES
from reference import GEOMETRY, fp19, make_inputs, reference

DEVICE = "a5"

# The destination dtype and poison per mode. 0x55 is in range for both byte destinations.
DESTINATIONS = {"qf_i8": (torch.int8, 0x55), "qf_u8": (torch.uint8, 0x55),
                "rq_i8": (torch.int8, 0x55), "deq_f16": (torch.float16, float("nan")),
                "scaled_f16": (torch.float16, float("nan")),
                "relu_i8": (torch.int8, 0x55), "split_i8": (torch.int8, 0x55)}

OUTPUTS = ("carriers",)

PURPOSE = {
    "qf_i8": "FP16 operands into an FP32 accumulator, quantized to signed bytes with scale 0.5 and "
             "offset 8. The planted +8 row against the +8 column reaches the saturation limit, so "
             "both clamps in the four-step conversion are exercised",
    "qf_u8": "The same path to unsigned bytes: the destination range becomes [0, 255], so the -8 "
             "column that merely rounds in the signed case saturates to zero here",
    "rq_i8": "INT8 operands into an INT32 accumulator, requantized to signed bytes with no offset: "
             "the integer path, where the product is exact before the scale",
    "deq_f16": "INT32 dequantized to FP16 with scale 0.25 -- no rounding to a byte and no clamp, so "
               "the scale's own FixP19 truncation is the only conversion left in the case",
    "scaled_f16": "FP32 scaled to FP16 with 0.5: the float-to-float path through the same bridge",
    "relu_i8": "ReLU before the quantization. Every negative product becomes zero and then takes "
               "the offset of 8, so an omitted ReLU shows up as a byte below 8 rather than as a "
               "small difference",
    "split_i8": "Two 16-row slices of one 32-row L0C, each through its own CvMutex, keeping the "
                "full L0C pitch. A slice that read the wrong 16 rows duplicates one half and loses "
                "the other, which the whole-output comparison catches in both halves at once",
}

CASES = [
    {"id": mode, "seed": 9700 + index, "block_dim": 1, "purpose": PURPOSE[mode],
     "parameters": {"mode": mode, "M": m, "N": n, "K": k,
                    "output_bytes": m * n * torch.empty((), dtype=DESTINATIONS[mode][0]).element_size()}}
    for index, (mode, (m, n, k)) in enumerate(GEOMETRY.items())
]


def check_domain(inputs, expected):
    """The bounded integral domain the exactness rests on, and one property of the scales: at 0.5
    and 0.25 the FixP19 truncation is the identity, so the reference's `fp19` is stating the rule
    rather than changing a value -- which is what makes it reusable for a scale where it would."""
    mode = inputs["mode"]
    m, n, k = GEOMETRY[mode]
    for name, shape in (("x", (m, k)), ("y", (n, k))):
        value = inputs[name]
        if tuple(value.shape) != shape or not value.is_contiguous():
            raise ValueError(f"{name} must be a contiguous {shape}")
        if not bool(torch.isfinite(value).all()) or bool((value.float().abs() > 8).any()):
            raise ValueError("the declared domain is integer values in [-8, 8]")
        if not torch.equal(value.float(), value.float().round()):
            raise ValueError("non-integral operands are outside this example")
    for scale in (0.5, 0.25):
        if fp19(scale) != scale:
            raise ValueError(f"{scale} is not exact under the FixP19 truncation, so the bitwise "
                             f"comparison would be measuring the scale's rounding")


def execute(case, inputs, launcher, backend):
    """One launch, one core. The destination arrives poisoned -- NaN for FP16, 0x55 for a byte -- and
    seeded in, so a row subblock 0 never published is distinguishable from a legitimately quantized
    one, which matters most in the split case where two stores cover disjoint halves."""
    mode = inputs["mode"]
    m, n, k = GEOMETRY[mode]
    dtype, poison = DESTINATIONS[mode]
    op = OpExec(ENTRIES[mode], launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    produced = op(inputs["x"], inputs["y"], torch.full((m, n), poison, dtype=dtype), m, n, k)
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
        unwritten = [i for i in index if got[i].item() == 0x55]
        print(f"      {len(index)}/{got.numel()} bytes differ, first at {index[:6]}: got "
              f"{[got[i].item() for i in index[:4]]} want {[want[i].item() for i in index[:4]]}"
              + (f"; {len(unwritten)} still hold the 0x55 fill (never published)"
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
            print(f"{case['id']:12s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        dtype, _ = DESTINATIONS[p["mode"]]
        print(f"{case['id']}  ({p['M']}x{p['N']}x{p['K']} -> "
              f"{str(dtype).removeprefix('torch.')}, launcher={args.launcher})")
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
