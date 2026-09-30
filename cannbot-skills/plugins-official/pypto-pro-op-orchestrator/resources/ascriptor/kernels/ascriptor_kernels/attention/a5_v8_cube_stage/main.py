# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the V8 Cube stage through OpExec and check it against the independent reference.

    python main.py                          # both cases, functional simulator
    python main.py --launcher pipesim
    python main.py --launcher aclnn --case padding_poison

This is a stage diagnostic, not a complete attention kernel: one QK, both drain modes of the
same L0C tile, and one PV. It publishes every intermediate to GM so the reference can compare
them, which is what makes the two drain modes directly comparable — `score2` is the two 64-query
halves a SPLITN drain writes per sub-block, `score3` is the whole tile a SINGLE drain writes.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import v8_cube_path
from reference import make_inputs, reference

# FP32 arithmetic over independently decoded finite HiFloat8 values: the only freedom the
# kernel has is the order it reduces in, so the bound is tight and the relative residual is
# what catches an output that is merely small rather than right.
TOLERANCE = {"atol": 0.0001, "rtol": 1e-05, "max_relative_l2": 1e-05}

INPUTS = {"q": ((128, 128), torch.uint8), "k": ((128, 128), torch.uint8),
          "v": ((128, 128), torch.uint8), "pbytes": ((33, 256), torch.uint8),
          "pbytes2": ((33, 256), torch.uint8)}
OUTPUTS = {"score": ((128, 128), torch.float32), "score2": ((256, 64), torch.float32),
           "score3": ((128, 128), torch.float32), "pv": ((128, 128), torch.float32),
           "pv2": ((128, 128), torch.float32)}

CASES = [
    {"id": "source_seed", "seed": 2027, "block_dim": 1,
     "purpose": "Zero padding in the NZ fractal's dead row: the ordinary case",
     "parameters": {"padding_byte": 0}},
    {"id": "padding_poison", "seed": 2031, "block_dim": 1,
     "purpose": "The same inputs with the dead row filled with 0xA5; the answer must not move",
     "parameters": {"padding_byte": 165}},
]


def execute(case, inputs, launcher, backend):
    """Every output is allocated NaN-poisoned and seeded into the launch, so an element no
    drain writes reads back as NaN rather than as a plausible zero."""
    outputs = {name: torch.full(shape, float("nan"), dtype=dtype)
               for name, (shape, dtype) in OUTPUTS.items()}
    op = OpExec(v8_cube_path, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    actual = op(*(inputs[name] for name in INPUTS), *outputs.values())
    return dict(zip(outputs, actual, strict=True))


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
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
