# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eight complex register operations at both physical widths, each its own output and bound.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case zero_numerator # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

Eight operations -- add, sub, mul, div, adds, muls, abs, dup -- run at `complex64` (32 lanes of FP32
components) and at `complex32` (64 lanes of FP16 components), from eight separate kernels. Each of
the sixteen results is its own named output with its own comparison, so a missing operation is a
failed output rather than a number absorbed into an average.

Complex results are converted to FP32 real/imaginary planes before comparison. That is not a
convenience: NumPy cannot represent Torch's `complex32`, so on some hosts the imaginary component of
a half-precision complex result is simply not observable. As planes it always is.

The reference expands the real and imaginary arithmetic in FP64 -- `mul` as
`(ar*br - ai*bi, ar*bi + ai*br)`, `div` over `br*br + bi*bi` -- and calls no DSL complex helper.

Bounds differ by width by a factor of twenty, because `complex32`'s components are FP16.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

import kernel
from reference import OPS, make_inputs, reference

DEVICE = "a5"

POISON = -777 + 333j     # for a complex destination; the real ones take -777
WIDTHS = {"64": (32, torch.complex64, torch.float32, 3e-04),
          "32": (64, torch.complex32, torch.float16, 6e-03)}
# The four kernels per width and the operations each returns, in return order.
FAMILIES = (("arith", ("add", "sub", "mul", "div")), ("scalar", ("adds", "muls")),
            ("abs", ("abs",)), ("dup", ("dup",)))

# Per output: the element bound and a matching relative L2 bound, at the width's own tolerance.
TOLERANCE = {f"c{kind}_{op}": {"rtol": bound, "atol": bound, "max_relative_l2": bound}
             for kind, (_, _, _, bound) in WIDTHS.items() for op in OPS}

OUTPUTS = tuple(f"c{kind}_{op}" for kind in WIDTHS for op in OPS)

CASES = [
    {"id": "literal_one_row", "seed": 103400, "block_dim": 1,
     "purpose": "One row of eight literal values: zero, the four sign quadrants of 3+4j, 1, j and "
                "-0.5+0.25j. abs has an exact answer of 5 at the Pythagorean values, and a multiply "
                "that dropped or mis-signed the cross term shows up in a different quadrant each time",
     "parameters": {"dataset": "literal", "rows": 1}},
    {"id": "source_rows", "seed": 0, "block_dim": 1,
     "purpose": "The original 16-row Gaussian composition, spread over the conditioned domain, with "
                "the denominator offset away from zero",
     "parameters": {"dataset": "source", "rows": 16}},
    {"id": "reuse_three_rows", "seed": 103401, "block_dim": 1,
     "purpose": "Three rows of eighth-unit components: the row loop reuses its single staging tile "
                "three times, and the arithmetic is exactly representable so the bounds are not "
                "doing the work",
     "parameters": {"dataset": "dyadic", "rows": 3}},
    {"id": "zero_numerator", "seed": 103402, "block_dim": 1,
     "purpose": "A zero numerator: multiply, divide, absolute value and scalar multiply are all "
                "legitimately zero here, so a zero-replacement control proves nothing for those "
                "four and only the corruption controls cover them",
     "parameters": {"dataset": "zero", "rows": 1}},
]


def check_domain(inputs, expected):
    """Three properties of the reference the sixteen outputs rest on. The bounded domain itself --
    finite, `|a| <= 8`, `0.125 <= |b| <= 12` so no denominator approaches zero -- is enforced by
    reference.py's own `validate`, which both `make_inputs` and `reference` call."""
    for kind, (columns, complex_dtype, _, _) in WIDTHS.items():
        for name in ("a", "b"):
            x = inputs[name + kind]
            if x.dtype != complex_dtype or x.shape[1] != columns:
                raise ValueError(f"{name}{kind} must be a complete {complex_dtype} register row")
        if not bool((expected[f"c{kind}_abs"] >= 0).all()):
            raise ValueError("an absolute value cannot be negative")
        # dup takes no operand: whatever the input row holds, the answer is the literal 3 - 2j.
        if not bool((expected[f"c{kind}_dup"][..., 0] == 3).all()
                    and (expected[f"c{kind}_dup"][..., 1] == -2).all()):
            raise ValueError("dup must be the literal 3 - 2j regardless of the input")
    if not all(bool(torch.isfinite(value).all()) for value in expected.values()):
        raise ValueError("no reference output may be non-finite over this domain")


def execute(case, inputs, launcher, backend):
    """Eight launches, one per kernel. A complex destination arrives poisoned with -777 + 333j and a
    real one with -777, both seeded in, so an unwritten lane is out of range in both components."""
    outputs = {}
    for kind, (_, _, real_dtype, _) in WIDTHS.items():
        a, b = inputs["a" + kind], inputs["b" + kind]
        for family, names in FAMILIES:
            entry = getattr(kernel, f"c{kind}_{family}_kernel")
            op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                        block_dim=case["block_dim"],
                        out_dir=f"tmp/{launcher}/{case['id']}/c{kind}_{family}",
                        seed_outputs=True)
            poisoned = [torch.full(a.shape, -777, dtype=real_dtype) if family == "abs"
                        else torch.full_like(a, POISON) for _ in names]
            args = ((a, b, *poisoned, a.shape[0]) if family == "arith"
                    else (a, *poisoned, a.shape[0]))
            produced = op(*args)
            values = produced if isinstance(produced, (tuple, list)) else [produced]
            for name, value in zip(names, values, strict=True):
                # Real/imaginary planes, so both components are comparable on any host.
                outputs[f"c{kind}_{name}"] = (value.float() if name == "abs"
                                              else torch.view_as_real(value.to(torch.complex64)))
    return outputs


def compare(name, got, want):
    """The element bound and the relative-norm bound for this output's width."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:10s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    rule = TOLERANCE[name]
    bounds = {"rtol": rule["rtol"], "atol": rule["atol"]}
    room = (bounds["atol"] + bounds["rtol"] * want.double().abs()).clamp(min=1e-30)
    margin = ((got.double() - want.double()).abs() / room).max().item()
    norm = torch.linalg.vector_norm(want.double().flatten())
    residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
    relative = (residual / norm).item() if norm > 0 else residual.item()
    ok = bool(torch.allclose(got, want, **bounds)) and relative <= rule["max_relative_l2"]
    print(f"    {name:10s} {'ok  ' if ok else 'FAIL'}  allclose={margin:.2f}x  "
          f"rel_l2={relative:.2e}/{rule['max_relative_l2']:g}")
    if not ok:
        outside = ~torch.isclose(got, want, **bounds)
        index = outside.nonzero()
        # A complex output's last axis is the component, so which one disagrees is worth naming.
        components = ({int(i[-1]) for i in index[:16]} if got.dim() == 3 else set())
        print(f"      {len(index)}/{got.numel()} values outside the element bound; first "
              f"{[tuple(i.tolist()) for i in index[:3]]}"
              + (f"; components {sorted(components)} (0 real, 1 imaginary)" if components else ""))
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--width", default="all", choices=("all", "32", "64"),
                        help="run only one physical width's outputs")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    names = [name for name in OUTPUTS
             if args.width == "all" or name.startswith(f"c{args.width}_")]
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['rows']} row(s), {p['dataset']} data, launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in names:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
