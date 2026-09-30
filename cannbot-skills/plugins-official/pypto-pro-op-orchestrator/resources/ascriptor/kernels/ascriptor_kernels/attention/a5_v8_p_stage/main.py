# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the V8 P stage through OpExec and check it against the independent reference.

    python main.py                          # both cases, functional simulator
    python main.py --launcher pipesim
    python main.py --launcher aclnn --case padding_poison

This is a stage diagnostic, not a complete attention kernel: exp over three score tiles, the
fused accumulator rescale, and the strided publication of the three P operands in their packed
physical layout. Every intermediate is published to GM so the reference can compare it.

The three P outputs are compared against a SET of admissible encodings rather than one value.
The device's native `exp` is within one measured FP32 ULP of a correctly rounded exp, and at a
HiFloat8 quantization boundary that one ULP selects the neighbouring code — so both endpoints
are accepted, but only at the elements where the interval actually crosses a boundary.
Everywhere else exactly one code is admissible and the check is as strict as bit equality.
That is a measured budget for this stage on A5, not a general native-exp accuracy claim.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import v8_p_path
from reference import make_inputs, reference, reference_candidates

# p0/p1/p2 take the one_of treatment described above. fin is bf16 out of an FP32 accumulator,
# so it carries the wider rounding budget; the st* probes and acc_out are FP32 throughout and
# only reduction order is free.
ONE_OF = ("p0", "p1", "p2")
TOLERANCE = {
    "default": {"atol": 0.0001, "rtol": 1e-05, "max_relative_l2": 1e-05},
    "fin": {"atol": 2e-06, "rtol": 0.008, "max_relative_l2": 0.004},
}

INPUTS = {"s0": ((128, 64), torch.float32), "s1": ((128, 64), torch.float32),
          "s2": ((128, 64), torch.float32), "acc_in": ((64, 128), torch.float32),
          "pv_in": ((64, 128), torch.float32), "w_in": ((1, 64), torch.float32),
          "zp": ((33, 256), torch.uint8)}
OUTPUTS = {"p0": ((33, 256), torch.uint8), "p1": ((33, 256), torch.uint8),
           "p2": ((33, 256), torch.uint8), "st1": ((2, 64), torch.float32),
           "st2": ((2, 64), torch.float32), "st3": ((2, 64), torch.float32),
           "st4": ((3, 64), torch.float32), "acc_out": ((64, 128), torch.float32),
           "fin": ((64, 128), torch.bfloat16)}

CASES = [
    {"id": "source_seed", "seed": 2026, "block_dim": 1,
     "purpose": "Zero packed-byte padding: the ordinary case",
     "parameters": {"padding_byte": 0}},
    {"id": "padding_poison", "seed": 2031, "block_dim": 1,
     "purpose": "The same inputs with the packed padding filled with 0xA5; every byte the "
                "stage publishes must be defined, so the answer must not move",
     "parameters": {"padding_byte": 165}},
]


def execute(case, inputs, launcher, backend):
    """Outputs are poisoned before the launch — NaN for float, 0xDF for the packed byte
    carriers — and seeded into it, so a byte the strided stores skip is visible as poison
    rather than as a plausible zero."""
    outputs = {name: torch.full(shape, 0xDF if dtype is torch.uint8 else float("nan"), dtype=dtype)
               for name, (shape, dtype) in OUTPUTS.items()}
    op = OpExec(v8_p_path, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    actual = op(*(inputs[name] for name in INPUTS), *outputs.values())
    return dict(zip(outputs, actual, strict=True))


def compare_one_of(name, got, choices):
    """Every element must equal one of its admissible encodings. `ambiguous` counts the
    elements where more than one is admissible — if it is 0 this was a bit-exact check."""
    got = got.cpu()
    stacked = torch.stack([c.cpu() for c in choices])
    ok = bool((stacked == got).any(dim=0).all())
    ambiguous = int((stacked != stacked[0]).any(dim=0).sum())
    wrong = int((~(stacked == got).any(dim=0)).sum())
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  {wrong} of {got.numel()} elements match "
          f"no admissible code  ({ambiguous} elements had two)")
    return ok


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail, outside = torch.equal(got, want), "  bitwise", got != want
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        atol, rtol = bounds.get("atol", 0.0), bounds.get("rtol", 0.0)
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for each
        # element. Print the worst element as a fraction of its own allowance, so the number has a
        # bound of 1 and a passing line cannot read as a failing one.
        room = (atol + rtol * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        outside = ~torch.isclose(got.float(), want.float(), **bounds)
        ok = not outside.any().item()
        detail = f"  allclose={margin:.2f}x (atol={atol:g} rtol={rtol:g})"
        if "max_relative_l2" in tolerance:
            norm = torch.linalg.vector_norm(want.double().flatten())
            residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
            relative = (residual / norm).item() if norm > 0 else residual.item()
            ok = ok and relative <= tolerance["max_relative_l2"]
            detail += f"  rel_l2={relative:.3e}/{tolerance['max_relative_l2']}"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok and outside.any():
        # The destinations arrive NaN-poisoned, so an element that is still NaN was never written.
        # Saying which, and where, is the difference between "something is nan" and "row 127 is".
        idx, poison = outside.nonzero(), int((outside & torch.isnan(got.float())).sum())
        note = f", {poison} still NaN-poisoned (never written)" if poison else ""
        print(f"      {len(idx)}/{outside.numel()} elements outside{note}; "
              f"first {' '.join(str(tuple(i.tolist())) for i in idx[:3])}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (padding_byte={case['parameters']['padding_byte']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        choices = reference_candidates(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            ok = (compare_one_of(name, actual[name], choices[name]) if name in ONE_OF
                  else compare(name, actual[name], expected[name],
                               TOLERANCE.get(name, TOLERANCE["default"])))
            if not ok:
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
