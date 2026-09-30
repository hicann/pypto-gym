# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run eight gated activations through OpExec and check them against the Torch reference.

    python main.py                              # every case, functional simulator, a2 facade
    python main.py --list                       # the case ids, with their purpose
    python main.py --device a3                  # the same sources through the a3 facade
    python main.py --launcher pipesim           # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case swiglu_bf16.source_odd

Eight complete source modes share this folder: GELU by the tanh approximation at FP32, FP16
and BF16, GELU by the Abramowitz-Stegun 7.1.26 erf polynomial at FP32 and BF16, and SwiGLU at
FP32, FP16 and BF16. A case names one in `variant`; `make_kernel` in kernel.py picks the
matching factory, and nothing else about the run changes. All eight compute in FP32 and cast
the result once, so what separates the precisions is the storage format and the tile capacity
that follows from it, not the arithmetic.

One kernel body serves both devices: the factories apply `ascriptor.a2.kernel` or
`ascriptor.a3.kernel` at call time. Import exactly one facade per process -- the module cache
does not restore an earlier target -- so a device switch means a new `python main.py --device`.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernel
from reference import make_inputs, reference, settings, split_inputs

# Keyed by variant, because the bound that means anything here is the storage format's: every
# mode evaluates the same FP32 operation sequence as the reference and rounds once at the end,
# so what is left is the ulp of float32, float16 or bfloat16. These are the source's own
# elementwise limits. The relative L2 residual is the part that is not per-dtype slack: it
# rejects an erased small output and a wrong mode -- the tanh body compared against an erf
# reference agrees to a few percent almost everywhere, which atol alone would sign off.
TOLERANCE = {
    "default": {"rtol": 1e-4, "atol": 1e-4, "max_relative_l2": 1e-4},
    "gelu_tanh_f32": {"rtol": 1e-4, "atol": 1e-4, "max_relative_l2": 1e-4},
    "gelu_tanh_f16": {"rtol": 0.002, "atol": 0.002, "max_relative_l2": 0.002},
    "gelu_tanh_bf16": {"rtol": 0.01, "atol": 0.01, "max_relative_l2": 0.005},
    "gelu_erf_f32": {"rtol": 1e-4, "atol": 1e-4, "max_relative_l2": 1e-4},
    "gelu_erf_bf16": {"rtol": 0.01, "atol": 0.01, "max_relative_l2": 0.005},
    "swiglu_f32": {"rtol": 1e-4, "atol": 1e-4, "max_relative_l2": 1e-4},
    "swiglu_f16": {"rtol": 0.01, "atol": 0.01, "max_relative_l2": 0.002},
    "swiglu_bf16": {"rtol": 0.05, "atol": 0.05, "max_relative_l2": 0.005},
}

CASES = [
    {"id": "gelu_tanh_f32.source_small", "seed": 1, "block_dim": 1,
     "purpose": "The original FP32 tanh case: 549 values in one tile",
     "parameters": {"variant": "gelu_tanh_f32", "shape": [1, 549], "mode": "random", "tile_len": 549}},
    {"id": "gelu_tanh_f32.single_value", "seed": 13400, "block_dim": 1,
     "purpose": "One element in a one-element tile: the whole launch is tail",
     "parameters": {"variant": "gelu_tanh_f32", "shape": [1], "mode": "positive", "tile_len": 1}},
    {"id": "gelu_tanh_f32.tile_reuse", "seed": 13401, "block_dim": 1,
     "purpose": "785 values in 256-element tiles on one core: the UB double buffer is reused "
                "four times and the last tile holds 17",
     "parameters": {"variant": "gelu_tanh_f32", "shape": [5, 157], "mode": "random", "tile_len": 256}},
    {"id": "gelu_tanh_f32.two_cores_tail", "seed": 13402, "block_dim": 2,
     "purpose": "1025 values over two cores: the second owner's last tile holds one element",
     "parameters": {"variant": "gelu_tanh_f32", "shape": [1, 1025], "mode": "random", "tile_len": 256}},
    {"id": "gelu_tanh_f32.mode_band", "seed": 13403, "block_dim": 1,
     "purpose": "All -2.5 with a single 1.0: the negative band is where tanh and erf disagree, "
                "so a swapped mode shows instead of blending in",
     "parameters": {"variant": "gelu_tanh_f32", "shape": [1, 513], "mode": "mode_band", "tile_len": 256}},

    {"id": "gelu_tanh_f16.source_small", "seed": 0, "block_dim": 1,
     "purpose": "The original FP16 tanh case: 1024 values in one tile",
     "parameters": {"variant": "gelu_tanh_f16", "shape": [1, 1024], "mode": "random", "tile_len": 1024}},
    {"id": "gelu_tanh_f16.single_value", "seed": 13400, "block_dim": 1,
     "purpose": "One element: the FP16 load, the FP32 widen and the TO_EVEN narrow, once",
     "parameters": {"variant": "gelu_tanh_f16", "shape": [1], "mode": "positive", "tile_len": 1}},
    {"id": "gelu_tanh_f16.tile_reuse", "seed": 13401, "block_dim": 1,
     "purpose": "Four 256-element tiles through the same FP16 and FP32 buffers",
     "parameters": {"variant": "gelu_tanh_f16", "shape": [5, 157], "mode": "random", "tile_len": 256}},
    {"id": "gelu_tanh_f16.two_cores_tail", "seed": 13402, "block_dim": 2,
     "purpose": "Two cores with a one-element final tile, at 2 bytes per value",
     "parameters": {"variant": "gelu_tanh_f16", "shape": [1, 1025], "mode": "random", "tile_len": 256}},
    {"id": "gelu_tanh_f16.mode_band", "seed": 13403, "block_dim": 1,
     "purpose": "The negative band at FP16, where the tanh and erf answers are furthest apart",
     "parameters": {"variant": "gelu_tanh_f16", "shape": [1, 513], "mode": "mode_band", "tile_len": 256}},

    {"id": "gelu_tanh_bf16.source_small", "seed": 2, "block_dim": 1,
     "purpose": "The original BF16 tanh case: 1023 values, an odd tile length",
     "parameters": {"variant": "gelu_tanh_bf16", "shape": [1, 1023], "mode": "random", "tile_len": 1023}},
    {"id": "gelu_tanh_bf16.single_value", "seed": 13400, "block_dim": 1,
     "purpose": "One element through BF16's 8 mantissa bits: the coarsest of the three narrows",
     "parameters": {"variant": "gelu_tanh_bf16", "shape": [1], "mode": "positive", "tile_len": 1}},
    {"id": "gelu_tanh_bf16.tile_reuse", "seed": 13401, "block_dim": 1,
     "purpose": "Four 256-element tiles reusing the BF16 and FP32 buffers",
     "parameters": {"variant": "gelu_tanh_bf16", "shape": [5, 157], "mode": "random", "tile_len": 256}},
    {"id": "gelu_tanh_bf16.two_cores_tail", "seed": 13402, "block_dim": 2,
     "purpose": "Two cores with a one-element final tile at BF16",
     "parameters": {"variant": "gelu_tanh_bf16", "shape": [1, 1025], "mode": "random", "tile_len": 256}},
    {"id": "gelu_tanh_bf16.mode_band", "seed": 13403, "block_dim": 1,
     "purpose": "The negative band at BF16, the precision where the mode gap is hardest to see",
     "parameters": {"variant": "gelu_tanh_bf16", "shape": [1, 513], "mode": "mode_band", "tile_len": 256}},

    {"id": "gelu_erf_f32.source_small", "seed": 4, "block_dim": 1,
     "purpose": "The original FP32 erf case: 549 values in one tile of the 5120 the erf body allows",
     "parameters": {"variant": "gelu_erf_f32", "shape": [1, 549], "mode": "random", "tile_len": 549}},
    {"id": "gelu_erf_f32.single_value", "seed": 13400, "block_dim": 1,
     "purpose": "One element through the full five-term polynomial and its exponential",
     "parameters": {"variant": "gelu_erf_f32", "shape": [1], "mode": "positive", "tile_len": 1}},
    {"id": "gelu_erf_f32.tile_reuse", "seed": 13401, "block_dim": 1,
     "purpose": "Four tiles through the erf body's larger temporary set",
     "parameters": {"variant": "gelu_erf_f32", "shape": [5, 157], "mode": "random", "tile_len": 256}},
    {"id": "gelu_erf_f32.two_cores_tail", "seed": 13402, "block_dim": 2,
     "purpose": "Two cores with a one-element final tile on the erf path",
     "parameters": {"variant": "gelu_erf_f32", "shape": [1, 1025], "mode": "random", "tile_len": 256}},
    {"id": "gelu_erf_f32.mode_band", "seed": 13403, "block_dim": 1,
     "purpose": "x = -2.5, where the final `(x/2 + |z|/sqrt(2)) - tail` subtraction cancels: "
                "simplifying it algebraically changes the answer here and nowhere else",
     "parameters": {"variant": "gelu_erf_f32", "shape": [1, 513], "mode": "mode_band", "tile_len": 256}},

    {"id": "gelu_erf_bf16.source_small", "seed": 5, "block_dim": 1,
     "purpose": "The original BF16 erf case: 1023 values in one tile",
     "parameters": {"variant": "gelu_erf_bf16", "shape": [1, 1023], "mode": "random", "tile_len": 1023}},
    {"id": "gelu_erf_bf16.single_value", "seed": 13400, "block_dim": 1,
     "purpose": "One element: FP32 polynomial, one TO_EVEN narrow to BF16",
     "parameters": {"variant": "gelu_erf_bf16", "shape": [1], "mode": "positive", "tile_len": 1}},
    {"id": "gelu_erf_bf16.tile_reuse", "seed": 13401, "block_dim": 1,
     "purpose": "Four tiles through the BF16 erf body's buffers",
     "parameters": {"variant": "gelu_erf_bf16", "shape": [5, 157], "mode": "random", "tile_len": 256}},
    {"id": "gelu_erf_bf16.two_cores_tail", "seed": 13402, "block_dim": 2,
     "purpose": "Two cores with a one-element final tile on the BF16 erf path",
     "parameters": {"variant": "gelu_erf_bf16", "shape": [1, 1025], "mode": "random", "tile_len": 256}},
    {"id": "gelu_erf_bf16.mode_band", "seed": 13403, "block_dim": 1,
     "purpose": "The cancelling negative tail at BF16: the subtraction happens in FP32 and only "
                "the result is narrowed, which is why it survives at all",
     "parameters": {"variant": "gelu_erf_bf16", "shape": [1, 513], "mode": "mode_band", "tile_len": 256}},

    {"id": "swiglu_f32.source_small", "seed": 1, "block_dim": 1,
     "purpose": "The original FP32 SwiGLU case: 128 columns split into two halves of 64",
     "parameters": {"variant": "swiglu_f32", "shape": [1, 128], "mode": "random", "beta": 0.5,
                    "tile_len": 64}},
    {"id": "swiglu_f32.source_odd", "seed": 2, "block_dim": 1,
     "purpose": "1023 columns: the odd column is cropped and each 511-wide half is flattened",
     "parameters": {"variant": "swiglu_f32", "shape": [1, 1023], "mode": "random", "beta": 2.0,
                    "tile_len": 511}},
    {"id": "swiglu_f32.source_rank3", "seed": 9, "block_dim": 1,
     "purpose": "Rank 3: the split is on the last dimension only, and the leading axes are just "
                "rows of the flattened operand",
     "parameters": {"variant": "swiglu_f32", "shape": [2, 3, 512], "mode": "random", "beta": 1.0,
                    "tile_len": 1536}},
    {"id": "swiglu_f32.single_value", "seed": 13410, "block_dim": 1,
     "purpose": "The smallest legal SwiGLU input: two columns, one output value",
     "parameters": {"variant": "swiglu_f32", "shape": [1, 2], "mode": "positive", "tile_len": 1,
                    "beta": 0.5}},
    {"id": "swiglu_f32.tile_reuse", "seed": 13411, "block_dim": 1,
     "purpose": "785 outputs in 256-element tiles: both operand buffers are reused four times",
     "parameters": {"variant": "swiglu_f32", "shape": [5, 314], "mode": "random", "tile_len": 256,
                    "beta": 2.0}},
    {"id": "swiglu_f32.two_cores_tail", "seed": 13412, "block_dim": 2,
     "purpose": "1025 outputs over two cores: the split must not let one owner read the other's half",
     "parameters": {"variant": "swiglu_f32", "shape": [5, 411], "mode": "random", "tile_len": 256,
                    "beta": 1.5}},
    {"id": "swiglu_f32.small_output", "seed": 13413, "block_dim": 1,
     "purpose": "Every input 0.0625, so every product is about 2e-3: far enough below atol that "
                "only the relative residual can tell a real answer from an erased one",
     "parameters": {"variant": "swiglu_f32", "shape": [3, 257], "mode": "small_output",
                    "tile_len": 256, "beta": 0.5}},
    {"id": "swiglu_f32.beta_zero", "seed": 13414, "block_dim": 1,
     "purpose": "beta = 0 collapses the gate to exactly 1/2, so a beta that was dropped, doubled "
                "or narrowed wrongly shows up as a clean factor rather than as noise",
     "parameters": {"variant": "swiglu_f32", "shape": [3, 258], "mode": "random", "tile_len": 256,
                    "beta": 0.0}},
    {"id": "swiglu_f32.split_order", "seed": 13415, "block_dim": 1,
     "purpose": "515 columns: crop the odd column first, then flatten each half. Flattening "
                "first and splitting afterwards is a different operation and this case says so",
     "parameters": {"variant": "swiglu_f32", "shape": [3, 515], "mode": "random", "tile_len": 256,
                    "beta": 0.5}},

    {"id": "swiglu_f16.source_small", "seed": 4, "block_dim": 1,
     "purpose": "The original FP16 SwiGLU case, with beta 0.0",
     "parameters": {"variant": "swiglu_f16", "shape": [1, 256], "mode": "random", "beta": 0.0,
                    "tile_len": 128}},
    {"id": "swiglu_f16.source_odd", "seed": 5, "block_dim": 1,
     "purpose": "2047 columns at FP16: an odd last dimension and a 1023-element tile",
     "parameters": {"variant": "swiglu_f16", "shape": [1, 2047], "mode": "random", "beta": 1.5,
                    "tile_len": 1023}},
    {"id": "swiglu_f16.source_rank4", "seed": 10, "block_dim": 1,
     "purpose": "Rank 4 and 17472 outputs in 6144-element tiles: the largest case here",
     "parameters": {"variant": "swiglu_f16", "shape": [3, 7, 13, 128], "mode": "random",
                    "beta": 0.5, "tile_len": 6144}},
    {"id": "swiglu_f16.single_value", "seed": 13410, "block_dim": 1,
     "purpose": "Two columns, one output value, through the FP16 widen and narrow",
     "parameters": {"variant": "swiglu_f16", "shape": [1, 2], "mode": "positive", "tile_len": 1,
                    "beta": 0.5}},
    {"id": "swiglu_f16.tile_reuse", "seed": 13411, "block_dim": 1,
     "purpose": "Four tiles through the FP16 operand pair and their FP32 copies",
     "parameters": {"variant": "swiglu_f16", "shape": [5, 314], "mode": "random", "tile_len": 256,
                    "beta": 2.0}},
    {"id": "swiglu_f16.two_cores_tail", "seed": 13412, "block_dim": 2,
     "purpose": "1025 outputs over two cores at 2 bytes per value",
     "parameters": {"variant": "swiglu_f16", "shape": [5, 411], "mode": "random", "tile_len": 256,
                    "beta": 1.5}},
    {"id": "swiglu_f16.small_output", "seed": 13413, "block_dim": 1,
     "purpose": "Products near 2e-3 at FP16, where atol 0.01 admits almost anything",
     "parameters": {"variant": "swiglu_f16", "shape": [3, 257], "mode": "small_output",
                    "tile_len": 256, "beta": 0.5}},
    {"id": "swiglu_f16.beta_zero", "seed": 13414, "block_dim": 1,
     "purpose": "beta = 0 at FP16: the gate is exactly 1/2 and the output is half the product",
     "parameters": {"variant": "swiglu_f16", "shape": [3, 258], "mode": "random", "tile_len": 256,
                    "beta": 0.0}},
    {"id": "swiglu_f16.split_order", "seed": 13415, "block_dim": 1,
     "purpose": "The odd-column crop before the flatten, at FP16",
     "parameters": {"variant": "swiglu_f16", "shape": [3, 515], "mode": "random", "tile_len": 256,
                    "beta": 0.5}},

    {"id": "swiglu_bf16.source_small", "seed": 7, "block_dim": 1,
     "purpose": "The original BF16 SwiGLU case: 512 columns, two 256-element tiles",
     "parameters": {"variant": "swiglu_bf16", "shape": [1, 512], "mode": "random", "beta": 1.0,
                    "tile_len": 256}},
    {"id": "swiglu_bf16.source_odd", "seed": 8, "block_dim": 1,
     "purpose": "4097 columns with beta 0.1: the smallest gate slope in the case list",
     "parameters": {"variant": "swiglu_bf16", "shape": [1, 4097], "mode": "random", "beta": 0.1,
                    "tile_len": 2048}},
    {"id": "swiglu_bf16.single_value", "seed": 13410, "block_dim": 1,
     "purpose": "Two columns, one output value, through the BF16 widen and narrow",
     "parameters": {"variant": "swiglu_bf16", "shape": [1, 2], "mode": "positive", "tile_len": 1,
                    "beta": 0.5}},
    {"id": "swiglu_bf16.tile_reuse", "seed": 13411, "block_dim": 1,
     "purpose": "Four tiles through the BF16 operand pair and their FP32 copies",
     "parameters": {"variant": "swiglu_bf16", "shape": [5, 314], "mode": "random", "tile_len": 256,
                    "beta": 2.0}},
    {"id": "swiglu_bf16.two_cores_tail", "seed": 13412, "block_dim": 2,
     "purpose": "1025 outputs over two cores at BF16",
     "parameters": {"variant": "swiglu_bf16", "shape": [5, 411], "mode": "random", "tile_len": 256,
                    "beta": 1.5}},
    {"id": "swiglu_bf16.small_output", "seed": 13413, "block_dim": 1,
     "purpose": "Products near 2e-3 against a BF16 atol of 0.05: the residual bound is the only "
                "thing left that can fail",
     "parameters": {"variant": "swiglu_bf16", "shape": [3, 257], "mode": "small_output",
                    "tile_len": 256, "beta": 0.5}},
    {"id": "swiglu_bf16.beta_zero", "seed": 13414, "block_dim": 1,
     "purpose": "beta = 0 at BF16: the gate is exactly 1/2",
     "parameters": {"variant": "swiglu_bf16", "shape": [3, 258], "mode": "random", "tile_len": 256,
                    "beta": 0.0}},
    {"id": "swiglu_bf16.split_order", "seed": 13415, "block_dim": 1,
     "purpose": "The odd-column crop before the flatten, at BF16",
     "parameters": {"variant": "swiglu_bf16", "shape": [3, 515], "mode": "random", "tile_len": 256,
                    "beta": 0.5}},
]


def execute(case, inputs, launcher, device):
    """Build this case's variant for this device and launch it. The destination is handed in
    poisoned with NaN and seeded into the launch, so a lane no tile covers reads back as NaN
    rather than as a value the activation could also have produced. SwiGLU takes the two halves
    as separate flattened operands and the negated beta as a scalar, because the kernel's own
    denominator is exp(-beta * x0) + 1."""
    entry = make_kernel(inputs["variant"], device)
    n, tile = inputs["n"], inputs["tile_len"]
    output = torch.full((1, n), float("nan"), dtype=inputs["x"].dtype)
    op = OpExec(entry, launcher=launcher, backend="cce", device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    if settings(inputs["variant"]).mode == "swiglu":
        x0, x1 = split_inputs(inputs)
        actual = op(x0.reshape(1, n), x1.reshape(1, n), output, n, -inputs["beta"], tile)
    else:
        actual = op(inputs["x"].reshape(1, n), output, n, tile)
    return {"output": actual.reshape(inputs["output_shape"])}


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
            print(f"{case['id']:32s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        beta = "" if "beta" not in p else f" beta={p['beta']}"
        print(f"{case['id']}  (shape={p['shape']} tile_len={p['tile_len']}{beta}, "
              f"device={args.device}, launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.device)
        tolerance = TOLERANCE.get(p["variant"], TOLERANCE["default"])
        for name in expected:
            if not compare(name, actual[name], expected[name], tolerance):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
