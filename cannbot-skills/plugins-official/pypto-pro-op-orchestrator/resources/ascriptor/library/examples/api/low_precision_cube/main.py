# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cube products over low-precision carriers: E5M2, HiFloat8 every which way, and scaled MXFP4.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case identity_scales    # one of them
    python main.py --only mxfp4             # one entry: e5m2, hif_manual, hif_matrix or mxfp4
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

Four cube entries publish fifteen independently compared products:

  e5m2        an ordinary unscaled product over float8_e5m2 operands, M64 N48 K32
  hif_manual  one HiFloat8 product assembled by hand, M32 N64 K64
  hif_*       twelve more of the same product: all four transpose pairings of A and B, each with no
              split, an N32 split and a K32 split, from a single launch
  mxfp4       FP4 E2M1 against FP4 E1M2 with row-major packed E8M0 scale codes, one scale per 32
              logical K values

The thirteen HiFloat8 outputs have *one* reference between them. `hif_at` and `hif_bt` are the
transposes of the very operands the untransposed entries take, so every pairing and every split
policy is computing the same product by a different route -- and main.py also compares the thirteen
to each other, which no single comparison against the reference can do.

The references decode each format from its own bit fields and multiply in FP64. No execution codec,
no kernel helper and no library cast is involved, so the products are evidence about the cube.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel.e5m2 import matmul_e5m2_shortcut_kernel
from kernel.hif8 import hif8_carrier_matmul_kernel, hif8_carrier_matmul_matrix_kernel
from kernel.mxfp4 import mxfp4_carrier_matmul_kernel
from reference import (FP4_E1M2, FP4_E2M1, HIF_NAMES, fp4_decode, make_inputs,
                       reference)

DEVICE = "a5"

POISON = -777.0           # every FP32 destination, seeded in
IDENTITY_SCALE = 127      # the E8M0 code whose scale factor is exactly 1

# Per output, the source's reviewed element bound plus a matching relative-norm bound. E5M2's element
# bound is twenty-five times the others'. Neither bound is tight on any of these outputs: measured on
# sim, the worst relative L2 across all three cases is 3.6e-08 against the 2e-03 allowed, and every
# allclose margin prints as 0.000x. The bounds are inherited slack, and what actually fails here is a
# wrong address or a dropped operand, which is wrong by orders of magnitude rather than by a fraction.
TOLERANCE = {"e5m2": {"rtol": 0.05, "atol": 0.1, "max_relative_l2": 0.002},
             "mxfp4": {"rtol": 0.002, "atol": 0.002, "max_relative_l2": 0.002}}
TOLERANCE.update({name: {"rtol": 0.002, "atol": 0.02, "max_relative_l2": 0.002}
                  for name in ["hif_manual"] + ["hif_" + n for n in HIF_NAMES]})

HIF_OUTPUTS = tuple(name for name in TOLERANCE if name.startswith("hif_"))
OUTPUTS = ("e5m2",) + HIF_OUTPUTS + ("mxfp4",)
ENTRIES = ("e5m2", "hif_manual", "hif_matrix", "mxfp4")

CASES = [
    {"id": "source", "seed": 11, "block_dim": 1, "parameters": {"dataset": "source"},
     "purpose": "The original composition: Gaussian values encoded into each format, and the MX "
                "scales spread over the three exponent codes 126, 127 and 128"},
    {"id": "finite_carriers", "seed": 103901, "block_dim": 1,
     "parameters": {"dataset": "finite_carriers"},
     "purpose": "Carrier bytes chosen directly from each format's finite code table rather than "
                "encoded from values, with both signs -- so the operands cover codes an encoder "
                "would rarely produce"},
    {"id": "identity_scales", "seed": 103902, "block_dim": 1,
     "parameters": {"dataset": "identity_scales"},
     "purpose": "Every MX scale code is 127, whose factor is exactly 1. The control: with the "
                "scaling switched off, a disagreement is in the FP4 product and not in the scales"},
]


def check_domain(inputs, expected):
    """What the thirteen HiFloat8 outputs rest on, and what the MX scale codes are doing in this
    case. The finite carrier domains and the transpose identities are checked by reference.py's own
    validate, which both make_inputs and reference call."""
    if any(bool((expected[name] == POISON).any()) for name in OUTPUTS):
        raise ValueError("a reference contains the poison value")
    first = expected[HIF_OUTPUTS[0]]
    if any(not torch.equal(expected[name], first) for name in HIF_OUTPUTS[1:]):
        raise ValueError("the thirteen HiFloat8 outputs must share one reference product; if they "
                         "did not, the transposes and splits would not be computing the same thing")
    codes = torch.cat([inputs["scale_a"].flatten(), inputs["scale_b"].flatten()]).tolist()
    if not all(126 <= code <= 128 for code in codes):
        raise ValueError("the packed E8M0 scale codes stay in the source's 126..128 domain")
    identity = all(code == IDENTITY_SCALE for code in codes)
    if identity:
        # Code 127 is a factor of exactly 1, so the reference must be the plain decoded FP4 product.
        plain = (fp4_decode(inputs["mx_a"], FP4_E2M1)
                 @ fp4_decode(inputs["mx_b"], FP4_E1M2).T).float()
        if not torch.equal(expected["mxfp4"], plain):
            raise ValueError("with every scale code 127 the reference must be the unscaled product")
    elif len(set(codes)) < 2:
        raise ValueError("a scaled case needs at least two different scale codes, or it cannot "
                         "distinguish a per-block scale from a single constant one")


def execute(case, inputs, launcher, backend, only):
    """Four launches. Every FP32 destination arrives filled with -777 and seeded in, so an unwritten
    element is a value no product here produces."""
    def op(entry, name):
        return OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                      block_dim=case["block_dim"],
                      out_dir=f"tmp/{launcher}/{case['id']}/{name}", seed_outputs=True)

    outputs = {}
    if only in (None, "e5m2"):
        outputs["e5m2"] = op(matmul_e5m2_shortcut_kernel, "e5m2")(
            inputs["e5_a"].view(torch.float8_e5m2), inputs["e5_b"].view(torch.float8_e5m2),
            torch.full((64, 48), POISON), 64, 48, 32)
    if only in (None, "hif_manual"):
        outputs["hif_manual"] = op(hif8_carrier_matmul_kernel, "hif_manual")(
            inputs["hif_a"], inputs["hif_b"], torch.full((32, 64), POISON), 32, 64, 64)
    if only in (None, "hif_matrix"):
        # One launch, twelve destinations: four transpose pairings x three split policies.
        products = op(hif8_carrier_matmul_matrix_kernel, "hif_matrix")(
            inputs["hif_a"], inputs["hif_at"], inputs["hif_b"], inputs["hif_bt"],
            *[torch.full((32, 64), POISON) for _ in HIF_NAMES], 0)
        outputs.update({"hif_" + name: value for name, value in
                        zip(HIF_NAMES, products, strict=True)})
    if only in (None, "mxfp4"):
        outputs["mxfp4"] = op(mxfp4_carrier_matmul_kernel, "mxfp4")(
            inputs["mx_a"], inputs["mx_b"], inputs["scale_a"], inputs["scale_b"],
            torch.full((16, 16), POISON), 0)
    return outputs


def compare(name, got, want):
    """This output's element bound and relative-norm bound. Both are reported, because which one is
    doing the work is part of what the run tells you."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:15s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    rule = TOLERANCE[name]
    bounds = {"rtol": rule["rtol"], "atol": rule["atol"]}
    got, want = got.cpu(), want.cpu()
    room = (bounds["atol"] + bounds["rtol"] * want.double().abs()).clamp(min=1e-30)
    margin = ((got.double() - want.double()).abs() / room).max().item()
    norm = torch.linalg.vector_norm(want.double().flatten())
    residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
    relative = (residual / norm).item() if norm > 0 else residual.item()
    ok = bool(torch.allclose(got, want, **bounds)) and relative <= rule["max_relative_l2"]
    print(f"    {name:15s} {'ok  ' if ok else 'FAIL'}  allclose={margin:.3f}x  "
          f"rel_l2={relative:.2e}/{rule['max_relative_l2']:g}")
    if not ok:
        outside = ~torch.isclose(got, want, **bounds)
        index = outside.nonzero()
        unwritten = int((got == POISON).sum())
        print(f"      {len(index)}/{got.numel()} outside the element bound, first "
              f"{[tuple(i.tolist()) for i in index[:3]]}"
              + (f"; {unwritten} still hold the {POISON} fill (never written)"
                 if unwritten else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--only", default=None, choices=ENTRIES,
                        help="run one of the four entries instead of all of them")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:17s} {case['purpose']}")
        print(f"\n{len(OUTPUTS)} outputs from the entries: " + ", ".join(ENTRIES))
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  ({case['parameters']['dataset']} carriers, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend, args.only)
        for name in (n for n in OUTPUTS if n in actual):
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
        present = [name for name in HIF_OUTPUTS if name in actual]
        if len(present) > 1:
            # Thirteen routes to one product: how far apart they are is worth a number, and the
            # element bound they each meet against the reference is the bound they must meet here.
            stack = torch.stack([actual[name].double() for name in present])
            spread = (stack.amax(0) - stack.amin(0)).abs().max().item()
            room = TOLERANCE[present[0]]["atol"] + TOLERANCE[present[0]]["rtol"] * float(
                expected[present[0]].abs().max())
            print(f"    {'hif spread':15s} {'ok  ' if spread <= room else 'FAIL'}  "
                  f"{spread:.3e} between the {len(present)} routes, against {room:.3e} of room")
            if spread > room:
                failed.append(f"{case['id']}/hif-spread")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
