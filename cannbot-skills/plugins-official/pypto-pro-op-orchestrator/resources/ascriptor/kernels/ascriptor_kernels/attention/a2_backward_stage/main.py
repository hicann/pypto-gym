# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the A2/A3 dense backward stage through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator, a2 facade
    python main.py --device a3              # the same source through the a3 facade
    python main.py --launcher pipesim
    python main.py --launcher aclnn --case prepared_tail

The stage consumes a saved forward state -- BF16 O plus the FP32 row maximum and denominator
-- and produces gQ, gK and gV. Two precision boundaries inside it are the reason the reference
is written the way it is: dQK is narrowed to BF16 before the two gradient contractions, and the
gV path stores the probabilities through HiFloat8 first. Both are modelled independently in
reference.py rather than borrowed from the device codec.

One kernel body serves both devices: `build_kernel` binds it to `ascriptor.a2` or
`ascriptor.a3` at call time. Import exactly one facade per process -- the module cache does
not restore an earlier target -- which is why a device switch means a new `python main.py`.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import build_kernel
from reference import make_inputs, reference

# Not bitwise, and the reason is in the kernel rather than in the comparison: dQK rounds to
# BF16 before gQ and gK are contracted, the gV probabilities pass through a HiFloat8 store,
# and the FP32 accumulators are reduced in a tile order the reference does not reproduce.
# The relative L2 bound is what makes the loose elementwise bounds safe -- at atol 0.05 a
# kernel that wrote zeros over a small gradient would otherwise pass.
TOLERANCE = {"atol": 0.05, "rtol": 0.05, "max_relative_l2": 0.01}

CASES = [
    {"id": "source_default", "seed": 42, "block_dim": 20,
     "purpose": "The archived stage ABI at an arbitrary saved O, 257 rows over 20 cores",
     "parameters": {"B": 1, "H": 1, "S1": 257, "S2": 257, "D": 128,
                    "state_mode": "source_arbitrary"}},
    {"id": "prepared_tail", "seed": 1, "block_dim": 2,
     "purpose": "Query and key tails of different lengths (129 vs 133) against a real forward state",
     "parameters": {"B": 1, "H": 1, "S1": 129, "S2": 133, "D": 128,
                    "state_mode": "prepared"}},
    {"id": "heads_idle", "seed": 2, "block_dim": 3,
     "purpose": "Two heads over three cores: the third core owns no head and must exit clean",
     "parameters": {"B": 1, "H": 2, "S1": 65, "S2": 73, "D": 128,
                    "state_mode": "prepared"}},
]

OUTPUTS = ("gq", "gk", "gv")


def execute(case, inputs, launcher, device):
    """Launch the stage. The three gradients are handed in poisoned with NaN and seeded into
    the launch, so a row no core claimed reads back as NaN instead of as the zero that a
    correctly masked tail row also produces."""
    outputs = [torch.full_like(inputs[name], float("nan")) for name in ("q", "k", "v")]
    op = OpExec(build_kernel(device), launcher=launcher, backend="cce", device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    actual = op(*(inputs[name] for name in ("q", "k", "v", "o", "grad", "qkmax", "qksum")),
                *outputs, inputs["B"], inputs["H"], inputs["S1"], inputs["S2"], 128,
                128 ** -0.5)
    return dict(zip(OUTPUTS, actual, strict=True))


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
    parser.add_argument("--device", default="a2", choices=("a2", "a3"))
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
        p = case["parameters"]
        print(f"{case['id']}  (B={p['B']} H={p['H']} S1={p['S1']} S2={p['S2']}, "
              f"{p['state_mode']}, device={args.device}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.device)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
