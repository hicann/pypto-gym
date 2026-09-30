# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the integrated four-stage convolution through OpExec and check it against FP64 Torch.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case edge_taps

This is the integrated level of the convolution_layout_pipeline family. The same four stages
as `stepwise` -- image to 5HD, weights to fractal, the cube convolution, and the NZ FP32 to
NCHW FP16 cast -- run inside **one** MIX launch, joined by `allvec_ready`/`allvec_wait`,
`vec_ready`/`wait_vec` and `allcube_ready`/`cube_ready`/`wait_cube` on the four cross-core
flags, with the intermediates living in `split_workspace` GM rather than in host tensors.

That is also what this demo cannot show you: no stage is observable from outside, so there is
no `--stages` here. Run `../stepwise` when you need to know which stage is wrong; run this one
when you want to see the barrier protocol that makes a single launch legal.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import kernel_for
from reference import make_inputs, reference

# FP16 operands accumulated in an FP32 L0C and cast back to FP16 once at the end cannot match
# an FP64 convolution bit for bit: 576 taps of rounding separate them. The elementwise bounds
# are the source's own. The relative-L2 limit is what a vacuous result cannot satisfy -- a
# kernel that writes zeros, or that drops every tap past the first K tile, stays inside atol
# over most of a 128x256 output and still fails the residual.
TOLERANCE = {"atol": 0.003, "rtol": 0.003, "max_relative_l2": 0.001}

CASES = [
    {"id": "source_seed11", "seed": 11, "block_dim": 1,
     "purpose": "Random values drawn in FP32 and rounded to FP16; the seed is a label, not the "
                "source's own generator",
     "parameters": {"shape": "1x64x32x32_128x64x3x3", "pattern": "random"}},
    {"id": "source_seed19", "seed": 19, "block_dim": 1,
     "purpose": "A second rounded-FP32 draw over the same geometry",
     "parameters": {"shape": "1x64x32x32_128x64x3x3", "pattern": "random"}},
    {"id": "signed_integers", "seed": 3, "block_dim": 1,
     "purpose": "Values in [-2, 2]: every partial sum is exact, so the tolerance is not what "
                "makes this case pass",
     "parameters": {"shape": "1x64x32x32_128x64x3x3", "pattern": "integer"}},
    {"id": "edge_taps", "seed": 7, "block_dim": 1,
     "purpose": "Only the two image corners are nonzero, so the pad-2 border and the outermost "
                "dilated taps decide the answer",
     "parameters": {"shape": "1x64x32x32_128x64x3x3", "pattern": "corners"}},
    {"id": "original_direct_fp16_seed11", "seed": 11, "block_dim": 1,
     "purpose": "The source driver's own direct-FP16 draws at seed 11",
     "parameters": {"shape": "1x64x32x32_128x64x3x3", "pattern": "random",
                    "sampling": "source_fp16"}},
]


def execute(case, inputs, launcher, backend):
    """One MIX launch: two vector cores and one cube core inside a single block. The image and
    weights are reshaped, not repacked -- the kernel builds 5HD and the fractal layout itself.
    The destination is handed in poisoned with NaN and seeded into the launch, so an output
    cell no stage writes reads back as NaN rather than as a plausible small number."""
    output = torch.full((128, 256), float("nan"), dtype=torch.float16)
    op = OpExec(kernel_for("a2"), launcher=launcher, backend=backend, device="a2",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"output": op(inputs["image"].reshape(64, 1024),
                         inputs["weights"].reshape(128, 576), output, 2)}


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
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:28s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (pattern={case['parameters']['pattern']}, "
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
