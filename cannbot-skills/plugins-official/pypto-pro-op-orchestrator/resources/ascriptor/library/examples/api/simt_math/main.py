# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SIMT scalar math: nine transcendentals with a budget, five roundings and six integer rows exact.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case rounding_and_bits # one of them
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

One SIMT thread per lane, 22 results each. The tolerance surface is deliberately small: only the
nine transcendentals and `simt_fmod` have a budget, because a scalar FP32 primitive may round
differently from Torch's. Everything else is compared byte for byte -- the five roundings, the FMA
(its multiplier is exactly 2.0, so the FP32 result is exact), and all six integer rows.

The two roundings that matter most are `simt_rint` and `simt_round`, and they are not synonyms:
rint takes ties to even, round takes them away from zero. The `rounding_and_bits` case puts x*10 on
+/-0.5, +/-11.5 and +/-12.5 so they disagree where they should, and the reference spells the
ties-away rule out by hand rather than reusing Torch's `round`, which is ties-to-even.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import FROWS, IROWS, N, simt_math_family
from reference import make_inputs, reference

DEVICE = "a5"

POISON = {"float": float("nan"), "integer": 12345}

# Per output. `approximate` and `fmod` are the only results whose primitive may round differently
# from the host's; naming them here keeps every other row on a byte comparison.
TOLERANCE = {"approximate": {"rtol": 1e-05, "atol": 1e-06, "max_relative_l2": 1e-05},
             "fmod": {"rtol": 1e-05, "atol": 1e-06, "max_relative_l2": 1e-05}}

OUTPUTS = ("approximate", "rounding", "fmod", "fma", "integer")

ROWS = {"approximate": ("exp", "exp2", "log", "log2", "log1p", "sin", "cos", "tanh", "rsqrt"),
        "rounding": ("rint (ties to even)", "round (ties away)", "floor", "ceil", "trunc"),
        "integer": ("isnan", "isinf", "isfinite", "popc", "mul_hi", "ffs")}

CASES = [
    {"id": "range_safe", "seed": 8831, "block_dim": 1,
     "purpose": "Finite values in [-4, 4] with no tie and no special among them: the nine "
                "transcendentals and the fmod against their budgets, and the roundings, the FMA "
                "and the six integer rows bitwise",
     "parameters": {"edges": False}},
    {"id": "rounding_and_bits", "seed": 8832, "block_dim": 1,
     "purpose": "The ten leading lanes are where the two roundings part: x*10 lands on -12.5, "
                "-11.5, -0.5, -0.0, 0.0, 0.5, 11.5, 12.5 and +/-40, so rint answers even and "
                "round answers away from zero on four of them. The bit carriers lead with 0, -1, "
                "-2^31 and 2^31-1, so popc sees both 0 and 32 set bits, mul_hi sees the widest "
                "square, and ffs sees the zero word it has to answer 0 for",
     "parameters": {"edges": True}},
]


def check_domain(inputs, expected):
    """The declared domain, and the two properties the exact rows depend on: the FMA's multiplier
    is a power of two (so `x * 2 + 1` is exact in FP32 and can be compared bitwise), and the
    specials are only in the classification input, not in the arithmetic one."""
    x, specials, bits = inputs["x"], inputs["specials"], inputs["bits"]
    if x.shape != (1, N) or specials.shape != (1, N) or bits.shape != (1, N):
        raise ValueError(f"the three inputs must be [1, {N}]")
    if not bool(torch.isfinite(x).all()) or bool((x.abs() > 4).any()):
        raise ValueError("the arithmetic input is finite and within [-4, 4]")
    if not torch.equal(expected["fma"], x.flatten() * 2 + 1):
        raise ValueError("the FMA reference must be x * 2 + 1, whose multiplier is exact in FP32")
    if not bool(torch.isnan(specials).any() and torch.isinf(specials).any()):
        raise ValueError("the classification input must carry a NaN and an infinity")


def execute(case, inputs, launcher, backend):
    """One launch producing both destinations, each poisoned and seeded in. The float poison is NaN,
    which three rows of the integer output legitimately classify -- so the two destinations carry
    different poisons and neither can be confused for a result."""
    op = OpExec(simt_math_family, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    of = torch.full((FROWS, N), POISON["float"])
    oi = torch.full((IROWS, N), POISON["integer"], dtype=torch.int32)
    of, oi = op(inputs["x"], inputs["specials"], inputs["bits"], of, oi)
    # The float destination is three separate results plus the nine approximate rows.
    return {"approximate": of[:9], "rounding": of[9:14], "fmod": of[14], "fma": of[15],
            "integer": oi}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise unless TOLERANCE names this output, and a failure names the row."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:12s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    rule = TOLERANCE.get(name)
    if rule is None:
        ok, detail = torch.equal(raw(got), raw(want)), "bitwise"
    else:
        bounds = {"rtol": rule["rtol"], "atol": rule["atol"]}
        room = (bounds["atol"] + bounds["rtol"] * want.double().abs()).clamp(min=1e-30)
        margin = ((got.double() - want.double()).abs() / room).max().item()
        norm = torch.linalg.vector_norm(want.double().flatten())
        residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
        relative = (residual / norm).item() if norm > 0 else residual.item()
        ok = (bool(torch.allclose(got, want, **bounds))
              and relative <= rule["max_relative_l2"])
        detail = f"allclose={margin:.2f}x  rel_l2={relative:.2e}/{rule['max_relative_l2']:g}"
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  {detail}")
    if not ok:
        flat = got.reshape(-1, N) if got.dim() > 1 else got.reshape(1, N)
        flat_want = want.reshape(-1, N) if want.dim() > 1 else want.reshape(1, N)
        labels = ROWS.get(name, (name,))
        for row in range(flat.shape[0]):
            outside = (raw(flat[row]).ne(raw(flat_want[row])).view(N, -1).any(dim=1)
                       if rule is None else ~torch.isclose(flat[row], flat_want[row],
                                                           rtol=rule["rtol"], atol=rule["atol"]))
            lanes = outside.nonzero().flatten().tolist()
            if lanes:
                print(f"      {labels[row]:20s} {len(lanes)} lanes differ at {lanes[:5]}: got "
                      f"{[round(float(flat[row, c]), 5) for c in lanes[:3]]} want "
                      f"{[round(float(flat_want[row, c]), 5) for c in lanes[:3]]}")
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
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  ({'with' if case['parameters']['edges'] else 'without'} the tie and "
              f"bit-boundary lanes, launcher={args.launcher})")
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
