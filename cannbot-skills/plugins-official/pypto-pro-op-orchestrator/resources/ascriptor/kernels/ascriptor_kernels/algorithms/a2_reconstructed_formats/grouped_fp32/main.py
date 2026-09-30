# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the three grouped FP32 formats through OpExec and check them against the references.

    python main.py                                  # every case, functional simulator
    python main.py --list                           # the case ids, with their purpose
    python main.py --case mbs_mxfp4_macro_boundary  # one case
    python main.py --launcher aclnn                 # cce backend, on this machine's card

Three quantize-dequantize formats share one kernel signature -- (x, y, rows, cols) -- and
are selected per case by `variant`: `plain_mxfp4` is a per-32 E8M0 scale with an E2M1
magnitude grid, `mbs_mxfp4` puts a per-128 E0M8 macro factor on top of that same per-32
step, and `mxfp8_e5m2` replaces the grid with a private-exponent window and a mantissa
step. All three reconstruct FP32 values; none of them publishes the packed payload bytes.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import build_kernel
from reference import make_inputs, reference

# The contract declares these bodies on both A2 and A3, and `build_kernel` binds the selected
# variant to one facade; change this to "a3" to run the identical source against A3.
DEVICE = "a2"

# The reference is not an approximation of this kernel: it reproduces the same grouping, the
# same exponent-bit scales, and even the A2 divider's tie families in `_a2_mbs_div`. What is
# left over is the order the vector unit evaluates FP32 arithmetic in, which is what 1e-5
# covers. The relative L2 ceiling is the part that rejects a vacuous answer -- on a
# tiny-scale group an output of zeros passes atol on its own.
TOLERANCE = {"atol": 1e-05, "rtol": 1e-05, "max_relative_l2": 1e-05}

CASES = [
    {"id": "plain_mxfp4_minimum", "seed": 0, "block_dim": 1,
     "purpose": "The smallest legal shape: one 32-element group on one core, all zeros",
     "parameters": {"variant": "plain_mxfp4", "rows": 1, "cols": 32, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "plain_mxfp4_zero", "seed": 0, "block_dim": 2,
     "purpose": "Every group is zero: the E8M0 floor guard must publish zeros rather than "
                "divide by an amax of zero",
     "parameters": {"variant": "plain_mxfp4", "rows": 4, "cols": 256, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "plain_mxfp4_boundaries", "seed": 1, "block_dim": 2,
     "purpose": "Every E2M1 threshold with both of its FP32 neighbours: which side of each "
                "tie the magnitude selection lands on",
     "parameters": {"variant": "plain_mxfp4", "rows": 2, "cols": 128,
                    "pattern": "e2m1_boundaries", "scale": 1.0}},
    {"id": "plain_mxfp4_source_reuse", "seed": 2, "block_dim": 20,
     "purpose": "3852 groups as 121 tiles over twenty cores: seven tiles per core reuse the "
                "same scale slots, flag buffer and broadcast rows",
     "parameters": {"variant": "plain_mxfp4", "rows": 321, "cols": 384, "pattern": "source",
                    "scale": 8.0}},
    {"id": "plain_mxfp4_tail", "seed": 0, "block_dim": 3,
     "purpose": "60 groups over three cores: the second tile is a 28-of-32 partial and the "
                "third core is handed an empty range",
     "parameters": {"variant": "plain_mxfp4", "rows": 5, "cols": 384, "pattern": "source",
                    "scale": 1.0}},
    {"id": "plain_mxfp4_idle", "seed": 3, "block_dim": 3,
     "purpose": "One group on three cores: two cores must publish nothing",
     "parameters": {"variant": "plain_mxfp4", "rows": 1, "cols": 32, "pattern": "source",
                    "scale": 0.2}},
    {"id": "plain_mxfp4_tiny", "seed": 4, "block_dim": 2,
     "purpose": "Magnitudes at 1e-20, where amax/4 sinks toward the 2^-127 E8M0 minimum and "
                "the clamp decides the answer",
     "parameters": {"variant": "plain_mxfp4", "rows": 1, "cols": 64, "pattern": "source",
                    "scale": 1e-20}},

    {"id": "mbs_mxfp4_minimum", "seed": 0, "block_dim": 1,
     "purpose": "One 128-element macro of zeros: the macro-factor guard returns 1.0 instead "
                "of dividing 6.0 by an amax of zero",
     "parameters": {"variant": "mbs_mxfp4", "rows": 1, "cols": 128, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "mbs_mxfp4_zero", "seed": 0, "block_dim": 2,
     "purpose": "Eight zero macros over two cores: both the macro guard and the inner E8M0 "
                "guard fire in the same launch",
     "parameters": {"variant": "mbs_mxfp4", "rows": 4, "cols": 256, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "mbs_mxfp4_macro_boundary", "seed": 1, "block_dim": 2,
     "purpose": "Three rows built at the macro-factor boundary: amax exactly 6.0 (factor 1), "
                "amax at 2^-126, and a 1.5/1.75/2/3 pattern that exercises the top-8 "
                "mantissa truncation",
     "parameters": {"variant": "mbs_mxfp4", "rows": 3, "cols": 256,
                    "pattern": "macro_boundaries", "scale": 1.0}},
    {"id": "mbs_mxfp4_tiny", "seed": 4, "block_dim": 2,
     "purpose": "One macro at 1e-20: the 6/amax reciprocal is the step that can overflow, so "
                "this is where the macro factor and the inner scale interact worst",
     "parameters": {"variant": "mbs_mxfp4", "rows": 1, "cols": 128, "pattern": "source",
                    "scale": 1e-20}},
    {"id": "mbs_mxfp4_source_reuse", "seed": 2, "block_dim": 40,
     "purpose": "All forty A2/A3 vector cores. The source decorated this body with a default "
                "@kernel(), which declares an AIC+2AIV task with an empty cube side; the "
                "entry here is explicitly mode=\"vec\", which is what makes 40 a vector grid",
     "parameters": {"variant": "mbs_mxfp4", "rows": 321, "cols": 384, "pattern": "source",
                    "scale": 8.0}},
    {"id": "mbs_mxfp4_factor_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Each macro column block is scaled by a different power of two, so the three "
                "macros of one row need three different factors; a factor held across macros "
                "shows up immediately",
     "parameters": {"variant": "mbs_mxfp4", "rows": 65, "cols": 384, "pattern": "factor_reuse",
                    "scale": 64.0}},
    {"id": "mbs_mxfp4_idle", "seed": 5, "block_dim": 3,
     "purpose": "One macro on three cores: two cores must publish nothing",
     "parameters": {"variant": "mbs_mxfp4", "rows": 1, "cols": 128, "pattern": "source",
                    "scale": 0.2}},

    {"id": "mxfp8_e5m2_minimum", "seed": 0, "block_dim": 1,
     "purpose": "The smallest legal shape: one 32-element group on one core, all zeros",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 1, "cols": 32, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "mxfp8_e5m2_zero", "seed": 0, "block_dim": 2,
     "purpose": "Every group is zero: the private exponent collapses to the preserved source "
                "epsilon and the clip bound is zero",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 4, "cols": 256, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "mxfp8_e5m2_epsilon", "seed": 1, "block_dim": 2,
     "purpose": "A 1e-20 to 1e20 logspace sweep in every group: 40 octaves against a 29-octave "
                "private-exponent window, so both the floor and the 1.75*group clip bind",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 2, "cols": 128, "pattern": "e5m2_epsilon",
                    "scale": 1.0}},
    {"id": "mxfp8_e5m2_source_reuse", "seed": 2, "block_dim": 20,
     "purpose": "3852 groups as 121 tiles over twenty cores: seven tiles per core reuse the "
                "same per-group exponent slots",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 321, "cols": 384, "pattern": "source",
                    "scale": 8.0}},
    {"id": "mxfp8_e5m2_tail", "seed": 0, "block_dim": 3,
     "purpose": "60 groups over three cores: the second tile is a 28-of-32 partial and the "
                "third core is handed an empty range",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 5, "cols": 384, "pattern": "source",
                    "scale": 1.0}},
    {"id": "mxfp8_e5m2_idle", "seed": 3, "block_dim": 3,
     "purpose": "One group on three cores: two cores must publish nothing",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 1, "cols": 32, "pattern": "source",
                    "scale": 0.2}},
]


def execute(case, inputs, launcher, backend):
    """Build the format this case names, then launch it. The destination is handed in poisoned
    with NaN and seeded into the launch, so a column the kernel never publishes -- a padded UB
    column or a guard row -- reads back as NaN rather than as a plausible zero."""
    op = OpExec(build_kernel(inputs["variant"], DEVICE), launcher=launcher, backend=backend,
                device=DEVICE, block_dim=case["block_dim"], out_dir=f"tmp/{launcher}",
                seed_outputs=True)
    y = torch.full_like(inputs["x"], float("nan"))
    return {"y": op(inputs["x"], y, inputs["rows"], inputs["cols"])}


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
    print(f"    {name:4s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
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
            print(f"{case['id']:26s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['variant']}, {p['rows']}x{p['cols']}, pattern={p['pattern']}, "
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
