# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the E4M3 MLA decode kernel through OpExec and check it against the independent reference.

    python main.py                              # every case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --launcher aclnn --case all  # cce backend on this machine's card, every case
    python main.py --launcher pypto --case odd_heads

One query row per head, B=1, and two scores per key: a 512-wide non-positional product and a
64-wide positional one. They are combined in the vector stage as `(rope * scale_rope + nope) *
1/sqrt(576)`, with `scale_rope` read from GM at run time. K_nope is also the value matrix, so
the same L1 tile feeds QK and PV.

The 16x probability headroom is deliberately NOT divided back out: `E4M3(16 * P)` multiplies V
and the sum in the denominator is the unscaled FP32 one, so the output carries the source's
16x convention. The reference reproduces that exactly rather than correcting it.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import mla_hif8
from reference import NAMES, make_inputs, reference

# Both sides quantize 16*P to E4M3 per 256-key tile and both keep the resulting 16x in the
# output, so what the tolerance covers is the order of the FP32 rescale chain on chip against
# Torch's, not a difference in convention. The relative L2 ceiling is what rejects a vacuous
# answer: every element is divided by the row sum, so atol alone would accept an output that
# is merely uniformly small.
TOLERANCE = {"atol": 0.001, "rtol": 0.0001, "max_relative_l2": 0.005}

CASES = [
    {"id": "paired", "seed": 42, "block_dim": 1,
     "purpose": "One complete head pair on one core over a single key tile: the shape the "
                "two-rows-per-pair partition was written for",
     "parameters": {"B": 1, "H": 2, "S": 256, "Dn": 512, "Dr": 64}},
    {"id": "odd_heads", "seed": 43, "block_dim": 2,
     "purpose": "Three heads: one full pair and an odd final head, so `rows = Min(2, H - h1)` "
                "must shorten every vf and the output store",
     "parameters": {"B": 1, "H": 3, "S": 512, "Dn": 512, "Dr": 64}},
    {"id": "idle", "seed": 44, "block_dim": 3,
     "purpose": "A single head on three cores: two cores get no pair at all and must write nothing",
     "parameters": {"B": 1, "H": 1, "S": 256, "Dn": 512, "Dr": 64}},
    {"id": "source_sequence", "seed": 45, "block_dim": 2,
     "purpose": "Four key tiles per head and three pairs over two cores: the rescale chain runs "
                "long enough to matter and the on-chip state is reused between pairs",
     "parameters": {"B": 1, "H": 5, "S": 1024, "Dn": 512, "Dr": 64}},
    {"id": "source_shape", "seed": 42, "block_dim": 1,
     "purpose": "The original source shape: 64 heads as 32 pairs on one core, four key tiles each",
     "parameters": {"B": 1, "H": 64, "S": 1024, "Dn": 512, "Dr": 64}},
]


def execute(case, inputs, launcher, backend):
    """Launch the kernel. The output is handed in poisoned with NaN and seeded into the launch,
    so a head the kernel never reaches reads back as NaN instead of as a plausible number."""
    q_nope, k_nope = inputs["q_nope"], inputs["k_nope"]
    out = torch.full(q_nope.shape, float("nan"))
    op = OpExec(mla_hif8, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(*(inputs[name] for name in NAMES), out,
                      1, q_nope.shape[1], k_nope.shape[1], 512, 64)}


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
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (H={p['H']} S={p['S']}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
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
