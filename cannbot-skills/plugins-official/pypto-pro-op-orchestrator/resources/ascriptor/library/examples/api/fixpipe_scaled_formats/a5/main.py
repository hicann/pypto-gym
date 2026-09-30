# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Six A5 fixpipe paths that scale, cast or dequantize straight from L0C into GM.

    python main.py                     # every case, functional simulator
    python main.py --list              # the case ids, with their purpose
    python main.py --case boundaries   # one of them
    python main.py --launcher pipesim  # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn    # the cce backend, on this machine's card

No UB staging: each kernel accumulates into L0C and the fixpipe writes the converted result directly
to GM. Six conversions, six named outputs:

    scaled_bf16   an FP32 accumulator scaled by 0.5 and cast to BF16
    scaled_f32    the same scaled accumulator kept in FP32 -- the control the two casts are read
                  against, and the one output with a tolerance rather than a byte comparison
    scaled_e4m3   the same scaled accumulator cast to float8_e4m3fn
    deq_bf16      an INT32 accumulator dequantized to BF16, scaled twice by 0.5
    hif8_ta       the scaled FP32 value encoded to HiFloat8 with the ta rounding policy
    hif8_hybrid   and with the hybrid policy. The pair is the point: one value, two policies

The cube operands are FP16 throughout, yet the HiFloat8 encoders consume the *FP32* L0C value after
the scale -- the accumulator's width is not the operands' width, and these two outputs are where that
matters.

The scalar scale is truncated to the top 19 FP32 bits before it is applied; 0.5 is exact under that
truncation and `check_domain` asserts it. Five outputs are compared as raw carriers with no host
narrowing; `scaled_f32` carries the source's reviewed element bound plus a stricter norm bound,
because the host product is FP64 and the device accumulates in FP32 in its own order.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernels
from reference import NAMES, fp19, geometry, make_inputs, reference

DEVICE = "a5"

POISON = 0x55             # every destination's carrier bytes, seeded in
DESTINATIONS = {"scaled_bf16": (torch.bfloat16, 2), "scaled_f32": (torch.float32, 4),
                "scaled_e4m3": (torch.float8_e4m3fn, 1), "deq_bf16": (torch.bfloat16, 2),
                "hif8_ta": (torch.uint8, 1), "hif8_hybrid": (torch.uint8, 1)}
INTEGRAL = ("deq_bf16",)            # the one that consumes the INT8 operands
HIF = ("hif8_ta", "hif8_hybrid")    # the two that consume the separate HiFloat8 operands

# Only the FP32 output has a tolerance. The other five are carrier comparisons.
TOLERANCE = {"scaled_f32": {"rtol": 0.001, "atol": 0.001, "max_relative_l2": 1e-06}}

OUTPUTS = tuple(NAMES)

CASES = [
    {"id": "source", "seed": 0, "block_dim": 1,
     "parameters": {"dataset": "source", "M": 16, "N": 32},
     "purpose": "The original small-integer operands in [-3, 3] at M16 N32. Products are exact, so "
                "every carrier byte is defined by integer arithmetic and the format's rounding"},
    {"id": "boundaries", "seed": 104086, "block_dim": 1,
     "parameters": {"dataset": "boundaries", "M": 16, "N": 32},
     "purpose": "Isolated products planted at each format's seams: the E4M3 lattice edges 448 and "
                "464, the BF16 rounding ties, powers of two down to 2**-22, and one value chosen to "
                "be the exact FP32 predecessor of 16 after the product and the scale"},
    {"id": "corrected", "seed": 0, "block_dim": 1,
     "parameters": {"dataset": "corrected", "M": 32, "N": 64},
     "purpose": "The corrected helper's own geometry, M32 N64 K64, on Gaussian FP16 operands. The "
                "only case whose products are not exact, and the reason scaled_f32 has a tolerance"},
]


def check_domain(inputs, expected, case):
    """The scale must be exact under the FP19 truncation, the geometry must be one of the two
    declared ones, and the two HiFloat8 policies must actually disagree somewhere -- otherwise the
    pair that exists to distinguish them would be two copies of one check."""
    if fp19(0.5) != 0.5:
        raise ValueError("0.5 must survive the FP19 truncation exactly")
    m, n, kf, ki = geometry(case)
    if inputs["geometry"] != (m, n, kf, ki):
        raise ValueError("the inputs' geometry must be this case's declared one")
    for name, (dtype, width) in DESTINATIONS.items():
        shape = (m, n) if name == "scaled_f32" else (m, n * width)
        kind = torch.float32 if name == "scaled_f32" else torch.uint8
        if expected[name].dtype != kind or tuple(expected[name].shape) != shape:
            raise ValueError(f"{name} is compared as {kind} {shape}")
    if torch.equal(expected["hif8_ta"], expected["hif8_hybrid"]):
        raise ValueError("the two HiFloat8 policies agree everywhere in this case, so it cannot "
                         "tell them apart")
    # The HiFloat8 operands are their own tensors in two of the three cases; where they are not, the
    # encoders still read the FP32 accumulator rather than the FP16 operands.
    if not bool(torch.isfinite(expected["scaled_f32"]).all()):
        raise ValueError("the FP32 control must stay finite over this domain")


def execute(case, inputs, launcher, backend):
    """One launch per conversion. Every destination's bytes arrive as 0x55 and are seeded in, so an
    element the fixpipe never wrote is a carrier none of these conversions produces."""
    m, n, kf, ki = inputs["geometry"]
    outputs = {}
    for name, entry in make_kernels(DEVICE).items():
        x, y = ((inputs["xi"], inputs["yi"]) if name in INTEGRAL
                else (inputs["hx"], inputs["hy"]) if name in HIF
                else (inputs["xf"], inputs["yf"]))
        dtype, width = DESTINATIONS[name]
        # Poison the raw carrier bytes, so even the FP8 destination needs no host narrowing.
        seed = torch.full((m, n * width), POISON, dtype=torch.uint8).view(dtype)
        op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}/{name}",
                    seed_outputs=True)
        produced = op(x, y, seed, m, n, ki if name in INTEGRAL else kf)
        outputs[name] = (produced if name == "scaled_f32"
                         else produced.contiguous().view(torch.uint8))
    return outputs


def compare(name, got, want):
    """Carriers byte for byte, except the FP32 control, which carries the reviewed element bound and
    a stricter relative-norm bound."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:12s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    got, want = got.cpu(), want.cpu()
    rule = TOLERANCE.get(name)
    if rule is None:
        ok = torch.equal(got, want)
        print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.numel()} carrier bytes")
        if not ok:
            differ = got != want
            index = differ.nonzero()
            print(f"      {int(differ.sum())}/{got.numel()} bytes differ, first "
                  f"{[tuple(i.tolist()) for i in index[:3]]}: got "
                  f"{[hex(int(got[tuple(i)])) for i in index[:4]]} want "
                  f"{[hex(int(want[tuple(i)])) for i in index[:4]]}")
        return ok
    bounds = {"rtol": rule["rtol"], "atol": rule["atol"]}
    room = (bounds["atol"] + bounds["rtol"] * want.double().abs()).clamp(min=1e-30)
    margin = ((got.double() - want.double()).abs() / room).max().item()
    norm = torch.linalg.vector_norm(want.double().flatten())
    residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
    relative = (residual / norm).item() if norm > 0 else residual.item()
    ok = bool(torch.allclose(got, want, **bounds)) and relative <= rule["max_relative_l2"]
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  allclose={margin:.3f}x  "
          f"rel_l2={relative:.2e}/{rule['max_relative_l2']:g}")
    if not ok:
        outside = ~torch.isclose(got, want, **bounds)
        index = outside.nonzero()
        print(f"      {len(index)}/{got.numel()} outside the element bound, first "
              f"{[tuple(i.tolist()) for i in index[:3]]}")
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
        m, n, kf, ki = geometry(case)
        print(f"{case['id']}  ({case['parameters']['dataset']} operands, {m}x{n}, K={kf} floating "
              f"and {ki} integral, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected, case)
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
