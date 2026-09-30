# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the [N, 4] row quaternion-to-rotation kernel through OpExec and check it against the
independent FP64 reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn         # cce backend, on this machine's card
    python main.py --launcher pypto --case source_100

One quaternion per GM row, which is the layout a caller normally already has. The price is
paid on chip: a 4-column row is loaded into an 8-column UB row and the 9 rotation columns are
built in a 16-column one, so the vector function works on padded rows and the GM store drops
the padding again. The nine outputs are written one scalar at a time by `quat_to_rotation_vf`,
not as whole vector registers -- this is the scalar-register shape of the same maths that
`quaternion_channels` does with full-width `mul`/`add` over [4, N] channels.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import kernel_for
from reference import make_inputs, reference

DEVICE = "a5"

# The kernel normalizes and builds the matrix in FP32 while the reference does it in FP64 and
# rounds once, so one division and nine products differ in their last bits: this is a tolerance,
# not bit equality. The relative L2 residual is what rejects a vacuous answer -- a rotation
# matrix is dominated by its +-1 diagonal, so atol alone would accept a kernel that erased
# every off-diagonal term.
TOLERANCE = {"rtol": 0.0001, "atol": 0.0001, "max_relative_l2": 0.0001}

CASES = [
    {"id": "source_64", "seed": 0, "block_dim": 1,
     "purpose": "First shape of the original seed-0 stream: two whole 32-row chunks",
     "parameters": {"N": 64, "mode": "source", "source_position": 0, "variant": "source_mix"}},
    {"id": "source_128", "seed": 0, "block_dim": 1,
     "purpose": "Second shape of that stream: four whole chunks, no tail",
     "parameters": {"N": 128, "mode": "source", "source_position": 1, "variant": "source_mix"}},
    {"id": "source_100", "seed": 0, "block_dim": 1,
     "purpose": "Third shape: three whole chunks and a 4-row tail chunk",
     "parameters": {"N": 100, "mode": "source", "source_position": 2, "variant": "source_mix"}},
    {"id": "source_17", "seed": 0, "block_dim": 1,
     "purpose": "Fourth shape: one partial chunk, so the chunk loop runs exactly once",
     "parameters": {"N": 17, "mode": "source", "source_position": 3, "variant": "source_mix"}},
    {"id": "source_1025", "seed": 0, "block_dim": 1,
     "purpose": "Fifth shape: thirty-two whole chunks and a single-row tail",
     "parameters": {"N": 1025, "mode": "source", "source_position": 4, "variant": "source_mix"}},
    {"id": "single", "seed": 13710, "block_dim": 1,
     "purpose": "One quaternion: the smallest chunk this tiling can produce",
     "parameters": {"N": 1, "mode": "axes", "variant": "source_mix"}},
    {"id": "signed_axes", "seed": 13711, "block_dim": 1,
     "purpose": "The eight basis and mixed-sign axes, where a swapped sign is a different rotation",
     "parameters": {"N": 8, "mode": "axes", "variant": "source_mix"}},
    {"id": "scaled_quaternions", "seed": 13712, "block_dim": 1,
     "purpose": "Quaternions scaled from 0.25 to 4: the kernel's own normalization must remove the scale",
     "parameters": {"N": 64, "mode": "scaled", "variant": "source_mix"}},
    {"id": "slot_reuse", "seed": 13713, "block_dim": 1,
     "purpose": "Six chunks on one vector: both DBuff slots are used and reused in turn",
     "parameters": {"N": 161, "mode": "random", "variant": "source_mix"}},
    {"id": "two_core_groups", "seed": 13714, "block_dim": 2,
     "purpose": "Two core groups plus a tail: the chunk split must not let one group write "
                "another's rows",
     "parameters": {"N": 257, "mode": "random", "variant": "source_mix"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the kernel for this case's ownership variant. The rotation destination is handed
    in poisoned with NaN and seeded into the launch, so a row the kernel never writes reads back
    as NaN rather than as a plausible zero."""
    n = inputs["N"]
    mode = "vec" if inputs["variant"] == "single_vector_unaligned" else "mix"
    rotation = torch.full((n, 9), float("nan"))
    op = OpExec(kernel_for(DEVICE, mode), launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"rotation": op(inputs["quaternion"], rotation, n)}


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
            print(f"{case['id']:20s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (N={p['N']}, {p['variant']}, launcher={args.launcher}, "
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
