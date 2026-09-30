# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the weighted block gather through OpExec and check it bit for bit against the reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case half_midpoint

One A2/A3 vector core owns the whole 256-value output. `kernel_for` binds the body with
`mode="vec"` and `block_dim=1` for exactly that reason: under the source's implicit MIX mode
both vector participants owned the whole output and wrote the same bytes, which every case
here still agrees with numerically. Only a pipe-level control sees it, so the one-owner launch
is part of the kernel, not a tuning choice.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import kernel_for
from reference import make_inputs, reference

# Bitwise. Every output bit is determined: the byte offsets pick exact source halves, and the
# kernel rounds to half once after each product and once after each add. The reference does the
# same two roundings in Python, so a fused multiply-add, a different midpoint rule, a shifted
# block address or an unwritten lane all show up as a changed bit rather than as a small error.
TOLERANCE = None

CASES = [
    {"id": "source_literal", "seed": 13200, "block_dim": 1,
     "purpose": "The source recipe verbatim: block addresses 0, 32, ... 992 in order",
     "parameters": {"mode": "source_literal", "inner_rep": 4}},
    {"id": "permuted", "seed": 13201, "block_dim": 1,
     "purpose": "Shuffled block addresses: a gather that ignores the offsets cannot pass",
     "parameters": {"mode": "permuted", "inner_rep": 4}},
    {"id": "repeated", "seed": 13202, "block_dim": 1,
     "purpose": "The same block fetched several times, within and across repeats",
     "parameters": {"mode": "repeated", "inner_rep": 4}},
    {"id": "product_rounding", "seed": 13203, "block_dim": 1,
     "purpose": "Products that only cancel if each one was rounded to half before the add",
     "parameters": {"mode": "product_rounding", "inner_rep": 4}},
    {"id": "half_midpoint", "seed": 13204, "block_dim": 1,
     "purpose": "Exact half midpoints: nearest-even, not truncation and not away-from-zero",
     "parameters": {"mode": "half_midpoint", "inner_rep": 4}},
    {"id": "unused_weights", "seed": 13205, "block_dim": 1,
     "purpose": "Weights 4..15 carry real values and must stay out of the output",
     "parameters": {"mode": "unused_weights", "inner_rep": 4}},
]


def execute(case, inputs, launcher, backend):
    """Launch the one vector kernel. The destination is handed in poisoned with NaN and seeded
    into the launch, so an output half the kernel never writes reads back as NaN instead of as
    a zero this recipe can also legitimately produce."""
    output = torch.full((1, 256), float("nan"), dtype=torch.float16)
    op = OpExec(kernel_for("a2"), launcher=launcher, backend=backend, device="a2",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"output": op(inputs["src"], inputs["offsets"], inputs["weights"], output,
                         inputs["inner_rep"])}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail = torch.equal(got, want), "  bitwise"
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (bounds.get("atol", 0.0) + bounds.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok, detail = torch.allclose(got.float(), want.float(), **bounds), f"  allclose={margin:.2f}x ({bounds})"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok:
        # Destinations arrive NaN-poisoned, so an element still NaN was never written.
        outside = (got != want) if tolerance is None else ~torch.isclose(got.float(), want.float(), **bounds)
        if outside.any():
            idx = outside.nonzero()
            poison = int((outside & torch.isnan(got.float())).sum())
            note = f", {poison} still NaN-poisoned (never written)" if poison else ""
            print(f"      {len(idx)}/{outside.numel()} elements outside{note}; "
                  f"first {' '.join(str(tuple(i.tolist())) for i in idx[:3])}")
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
        print(f"{case['id']}  (launcher={args.launcher}, block_dim={case['block_dim']})")
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
