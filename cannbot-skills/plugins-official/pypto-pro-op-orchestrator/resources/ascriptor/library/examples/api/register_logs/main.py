# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three logarithm bases from one FP32 register, against independent Torch references.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case second_seed    # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

`Reg.ln()`, `Reg.log2()` and `Reg.log10()` over sixteen rows of 64 FP32 lanes. The backend has one
logarithm primitive; the other two bases may be composed from `ln` and an FP32 scale
multiplication, so the results carry rounding that a bitwise comparison would reject for the wrong
reason. Hence a tolerance -- and two bounds rather than one: each element against `allclose`, and
the whole output against a relative L2 residual, so neither a single bad lane nor a small
systematic bias can pass.

The domain is finite positive values. Zero, negatives and non-finite inputs are refused by
`check_domain` rather than given a defined answer here.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import ROWS, vf_log_family
from reference import make_inputs, reference

DEVICE = "a5"

COLUMNS = 64

# One rule for all three outputs. `max_relative_l2` is the second bound: allclose alone would let a
# uniformly biased output through if the bias stayed inside atol + rtol*|want| everywhere.
TOLERANCE = {"rtol": 1e-05, "atol": 1e-06, "max_relative_l2": 1e-05}

OUTPUTS = ("ln", "log2", "log10")

CASES = [
    {"id": "powers_and_positive", "seed": 8771, "block_dim": 1,
     "purpose": "The first eight lanes are 1, 2, 4, 10, 100, 0.5, 0.1 and 1024: the points where "
                "log2 and log10 have integer answers and where all three bases agree on zero, so "
                "a wrong base scale shows up as a clean factor rather than as noise. The "
                "remaining 1016 lanes are positives spread over [0.001, 1000]",
     "parameters": {}},
    {"id": "second_seed", "seed": 8772, "block_dim": 1,
     "purpose": "The same three bases over a different draw, which is what shows the scale from "
                "ln to log2 and log10 is a property of the kernel rather than of these values",
     "parameters": {}},
]


def check_domain(inputs, expected):
    """The declared domain, refused rather than defined: a logarithm of zero or of a negative has
    no finite answer, and this unit makes no claim about what the primitive does with one."""
    x = inputs["x"]
    if x.shape != (ROWS, COLUMNS) or x.dtype != torch.float32:
        raise ValueError(f"the input must be float32[{ROWS}, {COLUMNS}]")
    if not bool(torch.isfinite(x).all()) or bool((x <= 0).any()):
        raise ValueError("this example's domain is finite positive values")
    for name in OUTPUTS:
        if not bool(torch.isfinite(expected[name]).all()):
            raise ValueError(f"the {name} reference is not finite over this domain")


def execute(case, inputs, launcher, backend):
    """One launch producing all three bases. Each destination arrives NaN-poisoned and is seeded
    into the launch, so a row the vector function never reached reads back as NaN -- which matters
    because the row loop writes one register per row per base."""
    op = OpExec(vf_log_family, launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    poisoned = [torch.full((ROWS, COLUMNS), float("nan")) for _ in OUTPUTS]
    return dict(zip(OUTPUTS, op(inputs["x"], *poisoned), strict=True))


def compare(name, got, want):
    """Element bound and relative-norm bound, both of them."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:6s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    bounds = {"rtol": TOLERANCE["rtol"], "atol": TOLERANCE["atol"]}
    # `atol` alone is not a bound on the worst element: allclose allows atol + rtol*|want| each.
    # Print the worst one as a fraction of its own allowance, so its bound is 1 and a passing line
    # cannot read as a failing one.
    room = (bounds["atol"] + bounds["rtol"] * want.double().abs()).clamp(min=1e-30)
    margin = ((got.double() - want.double()).abs() / room).max().item()
    norm = torch.linalg.vector_norm(want.double().flatten())
    residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
    relative = (residual / norm).item() if norm > 0 else residual.item()
    ok = bool(torch.allclose(got, want, **bounds)) and relative <= TOLERANCE["max_relative_l2"]
    print(f"    {name:6s} {'ok  ' if ok else 'FAIL'}  allclose={margin:.2f}x  "
          f"rel_l2={relative:.2e}/{TOLERANCE['max_relative_l2']:g}")
    if not ok:
        outside = ~torch.isclose(got, want, **bounds)
        index = outside.nonzero()
        poisoned = int((outside & torch.isnan(got)).sum())
        note = f", {poisoned} still NaN-poisoned (never written)" if poisoned else ""
        print(f"      {len(index)}/{got.numel()} lanes outside the element bound{note}; first "
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
            print(f"{case['id']:21s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (seed={case['seed']}, launcher={args.launcher}, "
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
