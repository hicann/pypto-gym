# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four A2/A3 fixpipe paths that quantize or dequantize straight from L0C into GM.

    python main.py                     # every case, functional simulator
    python main.py --list              # the case ids, with their purpose
    python main.py --case boundaries   # one of them
    python main.py --device a3         # the same source against the A3 facade
    python main.py --launcher pipesim  # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn    # the cce backend, on this machine's card

There is no UB staging anywhere here: each kernel accumulates into L0C and the fixpipe writes the
converted result directly to GM. Four conversions, four named outputs:

    qf_i8    an FP32 accumulator quantized to signed bytes, scale 0.5 and offset 8
    qf_u8    the same kernel with an unsigned destination -- signedness follows the dtype, and the
             pair is what shows that it does
    rq_i8    an INT32 accumulator requantized to signed bytes, scale 0.5 and offset 0
    deq_f16  an INT32 accumulator dequantized to FP16, scale 0.25

Byte quantization saturates twice, in an order that matters: scale, round to nearest even, clamp to
the intermediate signed 9-bit range [-256, 255], add the offset, then saturate to the destination's
own range. An implementation that added the offset before the intermediate clamp, or skipped it,
disagrees at the planted boundary values. The scalar scale is truncated to the top 19 FP32 bits
before it is applied, and the reference does that truncation explicitly.

Every result is compared as raw bytes, including the FP16 one. Nothing is narrowed on the host, so a
carrier is compared to a carrier.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernels
from reference import KF, KI, M, N, NAMES, fp19, make_inputs, reference

DEVICES = ("a2", "a3")

POISON = 77               # every typed destination, seeded in
DESTINATIONS = {"qf_i8": torch.int8, "qf_u8": torch.uint8, "rq_i8": torch.int8,
                "deq_f16": torch.float16}
INTEGRAL = ("rq_i8", "deq_f16")     # the two that consume the INT8 operands

# Which clamp each byte output actually reaches, per dataset, as (low, high). Measured, and asserted
# in both directions: the planted and random products drive every byte output to both ends, while the
# source products stay well inside -- its float products span [-86, 82] and its integers [-111, 94],
# so nothing saturates there except the unsigned floor, which the +8 offset cannot lift above zero.
# So `source` is the exactness control and says nothing about clamping; the other two are the ones
# that establish it.
SATURATES = {
    "source": {"qf_i8": (False, False), "qf_u8": (True, False), "rq_i8": (False, False)},
    "boundaries": {"qf_i8": (True, True), "qf_u8": (True, True), "rq_i8": (True, True)},
    "random": {"qf_i8": (True, True), "qf_u8": (True, True), "rq_i8": (True, True)},
}

OUTPUTS = tuple(NAMES)

CASES = [
    {"id": "source", "seed": 0, "block_dim": 1,
     "parameters": {"dataset": "source", "M": M, "N": N},
     "purpose": "The original small-integer operands in [-3, 3]. Products land in [-86, 82] and "
                "[-111, 94], well inside every destination range, so this case establishes the "
                "arithmetic and the rounding and says nothing at all about the clamping"},
    {"id": "boundaries", "seed": 104086, "block_dim": 1,
     "parameters": {"dataset": "boundaries", "M": M, "N": N},
     "purpose": "Isolated products planted at the conversion seams: -1024, +/-257, +/-255, the odd "
                "values that tie under round-to-nearest-even, and the signed-9 clamp edges. This is "
                "the case where the two saturations in the wrong order become visible"},
    {"id": "random", "seed": 104096, "block_dim": 1,
     "parameters": {"dataset": "random", "M": M, "N": N},
     "purpose": "Operands in [-8, 8]. The products reach both clamps in every byte output without "
                "having been chosen to -- which is what makes it a check the planted case does not "
                "already cover"},
]


def check_domain(inputs, expected, case):
    """The exact-arithmetic domain, and what each case is able to observe: the scale truncation must
    be the identity here, and the boundary case must actually reach both saturations."""
    if fp19(0.5) != 0.5 or fp19(0.25) != 0.25:
        raise ValueError("0.5 and 0.25 must survive the FP19 truncation exactly")
    for name, kind, shape in (("xf", torch.float16, (M, KF)), ("yf", torch.float16, (N, KF)),
                              ("xi", torch.int8, (M, KI)), ("yi", torch.int8, (N, KI))):
        if inputs[name].dtype != kind or tuple(inputs[name].shape) != shape:
            raise ValueError(f"{name} must be a complete {kind} {shape} operand")
    for name in OUTPUTS:
        width = 2 if name == "deq_f16" else 1
        if expected[name].dtype != torch.uint8 or tuple(expected[name].shape) != (M, N * width):
            raise ValueError(f"{name} is compared as {M}x{N * width} raw bytes")
    # Exactly which clamps this dataset reaches, asserted both ways: a generator change that stopped
    # driving a clamp, or started, is a change in what the case can observe and should be noticed.
    declared = SATURATES[case["parameters"]["dataset"]]
    for name, (low, high) in (("qf_i8", (0x80, 0x7F)), ("qf_u8", (0x00, 0xFF)),
                              ("rq_i8", (0x80, 0x7F))):
        reached = tuple(bool((expected[name] == end).any()) for end in (low, high))
        if reached != declared[name]:
            raise ValueError(f"{name} reaches (low, high) = {reached} here, not {declared[name]}: "
                             f"this case no longer observes what it was built to observe")
    if bool((expected["deq_f16"].view(torch.float16) == POISON).any()):
        raise ValueError("the dequantized reference contains the poison value")


def execute(case, inputs, launcher, backend, device):
    """One launch per conversion. Each destination arrives filled with 77 in its own dtype and seeded
    in, so an element the fixpipe never wrote is a value none of these products reaches."""
    outputs = {}
    for name, entry in make_kernels(device).items():
        x, y = ((inputs["xi"], inputs["yi"]) if name in INTEGRAL
                else (inputs["xf"], inputs["yf"]))
        op = OpExec(entry, launcher=launcher, backend=backend, device=device,
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}/{name}",
                    seed_outputs=True)
        produced = op(x, y, torch.full((M, N), POISON, dtype=DESTINATIONS[name]), M, N,
                      KI if name in INTEGRAL else KF)
        outputs[name] = produced.contiguous().view(torch.uint8)
    return outputs


def compare(name, got, want):
    """Bitwise over the carriers. Exact integer and dyadic products, an exact scale truncation and an
    explicit rounding rule define every byte."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:8s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    print(f"    {name:8s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} carrier bytes")
    if not ok:
        differ = got != want
        index = differ.nonzero()
        # An off-by-the-offset and a missing clamp look different; printing both values says which.
        print(f"      {int(differ.sum())}/{got.numel()} bytes differ, first "
              f"{[tuple(i.tolist()) for i in index[:3]]}: got "
              f"{[hex(int(got[tuple(i)])) for i in index[:4]]} want "
              f"{[hex(int(want[tuple(i)])) for i in index[:4]]}")
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
            print(f"{case['id']:12s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  ({case['parameters']['dataset']} operands, {M}x{N}, K={KF} floating "
              f"and {KI} integral, device={args.device}, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected, case)
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
