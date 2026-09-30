# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the BF16-state AdamW step through OpExec and check it against the FP32 reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case large   # cce backend, on this machine's card

One launch per parameter tensor: a case may carry several independent states, and each is
read once (gradient, parameter, both moments), updated entirely in FP32, and stored back as
three separate BF16 tensors. The host computes decay and both bias corrections in binary64
and narrows the eight scalars to FP32 before the launch, so the device never sees a double.

The comparison also covers `p_update`, the difference between the returned BF16 parameter
and the original one. It is not an extra output of the kernel; it exists so that a kernel
which returns the parameter unchanged fails, which a 2e-3 absolute limit on `p` alone would
not catch at these magnitudes.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import kernel_for
from reference import combine, make_inputs, reference

# The contract declares this body on both A2 and A3, and `kernel_for` binds it to one facade;
# change this to "a3" to run the identical source against the A3 target.
DEVICE = "a2"

# Every state is stored in BF16, so 2e-3 is roughly one storage quantum at these magnitudes
# rather than an allowance for sloppy arithmetic -- the reference follows the kernel's own
# operation order (AXPY, squared gradient, sqrt, host reciprocal, tensor division) and the
# same host-narrowed FP32 scalars. rtol is deliberately 0: the 0.1% relative residual is what
# makes the absolute limit mean something, because parameters and moments here live near
# 1e-2, where an output that never moved would pass atol on its own.
TOLERANCE = {"rtol": 0.0, "atol": 0.002, "max_relative_l2": 0.001}

OUTPUTS = ("p", "m", "v", "p_update")

CASES = [
    {"id": "single", "seed": 9500, "block_dim": 1,
     "purpose": "One element: the shortest possible tail in a 2048-lane tile",
     "parameters": {"sizes": [1], "hyperparameters": {}, "total_numel": 1}},
    {"id": "tiny_tail", "seed": 9501, "block_dim": 1,
     "purpose": "Seven elements: the vector unit computes all 2048 lanes and only seven are "
                "stored back to GM",
     "parameters": {"sizes": [7], "hyperparameters": {}, "total_numel": 7}},
    {"id": "tile_boundaries", "seed": 9502, "block_dim": 2,
     "purpose": "2047, 2048 and 2049 elements in one case: one lane short of a tile, exactly "
                "a tile, and one lane into a second tile",
     "parameters": {"sizes": [2047, 2048, 2049], "hyperparameters": {}, "total_numel": 6144}},
    {"id": "slot_reuse", "seed": 9503, "block_dim": 1,
     "purpose": "Four tiles on one core: each of the seven double buffers is reused twice, "
                "and the last tile carries three valid elements",
     "parameters": {"sizes": [6147], "hyperparameters": {}, "total_numel": 6147}},
    {"id": "multi_tensor", "seed": 9504, "block_dim": 2,
     "purpose": "Four independent tensors of different sizes launched in sequence: a launch "
                "must not read what the previous tensor left in the same on-chip buffers",
     "parameters": {"sizes": [21, 64, 4096, 8192], "hyperparameters": {}, "total_numel": 12373}},
    {"id": "large", "seed": 9505, "block_dim": 4,
     "purpose": "50000 elements as 25 tiles over four cores, ending on an 848-element tail",
     "parameters": {"sizes": [50000], "hyperparameters": {}, "total_numel": 50000}},
    {"id": "first_step", "seed": 9506, "block_dim": 1,
     "purpose": "Step 1 with weight_decay 0: both bias corrections are at their largest and "
                "the decay factor is exactly 1.0, so the parameter path is pure subtraction",
     "parameters": {"sizes": [128], "total_numel": 128,
                    "hyperparameters": {"step": 1, "weight_decay": 0.0, "lr": 0.001}}},
    {"id": "late_step", "seed": 9507, "block_dim": 2,
     "purpose": "Step 100 with a 1e-6 epsilon: both corrections have converged to ~1 and the "
                "epsilon, not sqrt(v), decides the denominator for small moments",
     "parameters": {"sizes": [2050], "total_numel": 2050,
                    "hyperparameters": {"step": 100, "eps": 1e-06}}},
    {"id": "bf16_midpoints", "seed": 9508, "block_dim": 1,
     "purpose": "Two elements built so the FP32 result lands exactly on a positive and a "
                "negative BF16 midpoint: the implicit store rounds ties away from zero, and "
                "a ties-to-even store gets both of them wrong",
     "parameters": {"sizes": [2], "total_numel": 2, "pattern": "bf16_midpoints",
                    "hyperparameters": {"beta1": 0.5, "beta2": 0.5, "step": 1}}},
]


def execute(case, inputs, launcher, backend):
    """One launch per parameter tensor. All three destinations are handed in poisoned with NaN
    and seeded into the launch, so a lane past the tail -- or a tensor a core never reached --
    reads back as NaN rather than as a plausible state value."""
    entry = kernel_for(DEVICE)
    outputs = []
    for state in inputs["states"]:
        poisoned = [torch.full_like(state["p"], float("nan")) for _ in range(3)]
        op = OpExec(entry, launcher=launcher, backend=backend, device=DEVICE,
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
        actual = op(state["g"], state["p"], state["m"], state["v"], *poisoned,
                    *inputs["scalars"], state["p"].numel())
        outputs.append(dict(zip(("p", "m", "v"), actual, strict=True)))
    return combine(inputs["states"], outputs)


def compare(name, got, want, tolerance):
    got, want = got.cpu().float(), want.cpu().float()
    if tolerance is None:
        ok, detail = torch.equal(got, want), ""
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (bounds.get("atol", 0.0) + bounds.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok, detail = torch.allclose(got, want, **bounds), f"  allclose={margin:.2f}x ({bounds})"
        if "max_relative_l2" in tolerance:
            norm = torch.linalg.vector_norm(want.flatten())
            residual = torch.linalg.vector_norm((got - want).flatten())
            relative = (residual / norm).item() if norm > 0 else residual.item()
            ok = ok and relative <= tolerance["max_relative_l2"]
            detail += f"  rel_l2={relative:.3e}/{tolerance['max_relative_l2']}"
    worst = (got - want).abs().max().item()
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
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
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (sizes={p['sizes']}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
