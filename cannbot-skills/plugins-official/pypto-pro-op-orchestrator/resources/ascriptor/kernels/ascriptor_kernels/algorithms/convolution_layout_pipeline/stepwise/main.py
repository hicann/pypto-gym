# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the four-launch convolution pipeline through OpExec and check it against FP64 Torch.

    python main.py                          # every case, functional simulator
    python main.py --stages                 # also check each of the four kernels on its own
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher aclnn --case edge_taps

This is the stepwise level of the convolution_layout_pipeline family. The same four stages as
`integrated` run as four separate launches the host composes: image to 5HD (two vector owners),
weights to fractal (two vector owners), the cube convolution (one cube owner), then NZ FP32 to
NCHW FP16 (two vector owners). Each stage's output is an ordinary host tensor, which is the
whole difference from `integrated`, where the same tensors live in `split_workspace` GM and
nobody outside can see them.

`--stages` is what that buys you. It feeds each kernel an input built by reference.py rather
than by the kernel before it, so stage 3 is checked on hand-packed 5HD and hand-packed fractal
weights, and stage 4 on a probe tensor of FP16 midpoints instead of on the cube's output. A
composed run tells you the answer is wrong; this tells you which launch made it wrong, and
distinguishes a bad stage from a stage fed bad input.

Each stage declares its own owner count in `kernel_for_*`, and the cases carry `block_dim: 1`
because that is the pipeline's launch group, not any one kernel's width. The three vector
stages partition statically over `N_VEC = 2`, so running them on one core would silently drop
lane 1's half of the work.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import (kernel_for_conv_compute, kernel_for_out_to_nchw, kernel_for_w_to_fractal,
                    kernel_for_x_to_5hd)
from reference import input_stages, make_inputs, reference, reference_stages

# Two of the four stages move halves without computing anything, and the fourth is a single
# specified rounding, so those three are exact claims and get no tolerance at all. Only the
# cube convolution rounds: FP16 operands summed over 576 taps in an FP32 accumulator cannot
# equal an FP64 sum. Its elementwise bounds are the source's own, and the relative-L2 limit is
# what a vacuous output fails -- zeros, or a kernel that stops after the first K tile, stay
# inside atol across most of a 128x256 result.
TOLERANCE = {"default": {"atol": 0.003, "rtol": 0.003, "max_relative_l2": 0.001},
             "data5hd": None, "weights_fractal": None, "cast_nchw": None}

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

OUTPUTS = ("output",)
STAGE_OUTPUTS = ("data5hd", "weights_fractal", "conv_nz", "cast_nchw")


def launch(entry, args, launcher, backend, cores):
    """One stage. `cores` is the stage's own owner count -- 2, 2, 1, 2 across the pipeline --
    and matches the `block_dim` its `kernel_for_*` factory already baked into the decorator."""
    op = OpExec(entry, launcher=launcher, backend=backend, device="a2", block_dim=cores,
                out_dir=f"tmp/{launcher}/{entry.name}", seed_outputs=True)
    return op(*args)


def execute(case, inputs, launcher, backend):
    """Launch the four stages in order, each one reading what the previous stage actually
    wrote. Every destination is handed in poisoned with NaN and seeded into the launch, so a
    cell no stage writes reads back as NaN rather than as a plausible number."""
    def run(entry, args, cores):
        return launch(entry, args, launcher, backend, cores)
    data = run(kernel_for_x_to_5hd("a2"),
               (inputs["image"].reshape(64, 1024),
                torch.full((4608, 16), float("nan"), dtype=torch.float16), 2), 2)
    weights = run(kernel_for_w_to_fractal("a2"),
                  (inputs["weights"].reshape(128, 576),
                   torch.full((128, 576), float("nan"), dtype=torch.float16), 2), 2)
    nz = run(kernel_for_conv_compute("a2"),
             (data, weights, torch.full((2048, 16), float("nan")), 1), 1)
    output = run(kernel_for_out_to_nchw("a2"),
                 (nz, torch.full((128, 256), float("nan"), dtype=torch.float16), 2), 2)
    return {"output": output}


def execute_stages(case, inputs, launcher, backend):
    """The same four kernels, but each one fed an input reference.py built rather than one the
    previous kernel produced. Stage 3 gets hand-packed 5HD and hand-packed fractal weights, and
    stage 4 gets the FP16 midpoint probes instead of the cube's output -- so a failure here is
    that stage's own defect, not an inherited one."""
    def run(entry, args, cores):
        return launch(entry, args, launcher, backend, cores)
    packed_data, packed_weights = input_stages(inputs)
    return {
        "data5hd": run(kernel_for_x_to_5hd("a2"),
                       (inputs["image"].reshape(64, 1024),
                        torch.full((4608, 16), float("nan"), dtype=torch.float16), 2), 2),
        "weights_fractal": run(kernel_for_w_to_fractal("a2"),
                               (inputs["weights"].reshape(128, 576),
                                torch.full((128, 576), float("nan"), dtype=torch.float16), 2), 2),
        "conv_nz": run(kernel_for_conv_compute("a2"),
                       (packed_data, packed_weights,
                        torch.full((2048, 16), float("nan")), 1), 1),
        "cast_nchw": run(kernel_for_out_to_nchw("a2"),
                         (inputs["cast_source"],
                          torch.full((128, 256), float("nan"), dtype=torch.float16), 2), 2),
    }


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


def tolerance_for(name):
    return TOLERANCE[name] if name in TOLERANCE else TOLERANCE["default"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--stages", action="store_true",
                        help="also run each kernel alone on an independently built input")
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
        if args.stages:
            expected = reference_stages(inputs)
            actual = execute_stages(case, inputs, args.launcher, args.backend)
            for name in STAGE_OUTPUTS:
                if not compare(name, actual[name], expected[name], tolerance_for(name)):
                    failed.append(f"{case['id']}/{name}")
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name], tolerance_for(name)):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
