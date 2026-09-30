# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the four FP16 matrix-product normalizations through OpExec and check them against the
independent FP64 reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn         # cce backend, on this machine's card
    python main.py --launcher pypto --case block_absmax_single

Every variant computes `product = X @ Y.T` on the cube and then divides each row by something
it reduces from that row, but they are four different kernels, not one kernel with a flag:

  row_sum_small   one pass. The whole N=256 row fits one L0C tile, so the sum and the division
                  happen in the same vector call and nothing is ever re-read.
  row_sum_large   two passes. N is larger than one tile, so the denominator is not known until
                  every N tile of the row has been produced: pass 1 accumulates the row sum in
                  a persistent UB tensor and parks the raw product in the output tensor, and
                  pass 2 reloads it and divides. `bar_all()` separates them.
  row_l2          the same two-pass shape with sum-of-squares and a sqrt.
  block_absmax    one pass again, because its denominator is local: each 128-column block is
                  divided by its own maximum, so nothing has to wait for the rest of the row.

Reading row_sum_small against row_sum_large is the point of the unit: the same normalization
costs one pass or two depending only on whether the denominator's whole input fits on chip.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import (matmul_chunk_absmax_norm128_kernel, matmul_rowwise_l2_norm_kernel,
                    matmul_rowwise_norm_kernel, matmul_rowwise_norm_large_nk_kernel)
from reference import make_inputs, reference

ENTRIES = {"row_sum_small": matmul_rowwise_norm_kernel,
           "row_sum_large": matmul_rowwise_norm_large_nk_kernel,
           "row_l2": matmul_rowwise_l2_norm_kernel,
           "block_absmax": matmul_chunk_absmax_norm128_kernel}

# The strictest of the four preserved source allowances, applied to all of them: FP16 operands
# reduced in FP32 on the cube and divided in FP32 on the vector unit, against an FP64 product
# and an FP64 denominator rounded once at the end. The relative residual is the part that
# matters here -- a normalized row sums to 1 over N columns, so every element is around 1/N and
# an output of all zeros passes atol=0.002 on its own merits at any N above about 500.
TOLERANCE = {"rtol": 0.002, "atol": 0.002, "max_relative_l2": 0.0001}

CASES = [
    {"id": "row_sum_small_minimum", "seed": 5600, "block_dim": 1,
     "purpose": "One 128-row M tile on one core: the shortest path through the one-pass form",
     "parameters": {"variant": "row_sum_small", "M": 128, "N": 256, "K": 128}},
    {"id": "row_sum_small_source", "seed": 5601, "block_dim": 2,
     "purpose": "The original M=5120 shape: forty M tiles split across two cores",
     "parameters": {"variant": "row_sum_small", "M": 5120, "N": 256, "K": 128}},
    {"id": "row_sum_small_idle", "seed": 5602, "block_dim": 3,
     "purpose": "One M tile on three cores: two cube groups and their vector peers stay idle",
     "parameters": {"variant": "row_sum_small", "M": 128, "N": 256, "K": 128}},
    {"id": "row_sum_small_reuse", "seed": 5603, "block_dim": 1,
     "purpose": "Five M tiles on one core: every L1, L0C and UB double buffer cycles",
     "parameters": {"variant": "row_sum_small", "M": 640, "N": 256, "K": 128}},

    {"id": "row_sum_large_minimum", "seed": 5610, "block_dim": 1,
     "purpose": "One tile in every dimension: the two-pass form with nothing to accumulate over",
     "parameters": {"variant": "row_sum_large", "M": 64, "N": 256, "K": 128}},
    {"id": "row_sum_large_source", "seed": 5611, "block_dim": 2,
     "purpose": "The original large shape: 20 M tiles by 10 N tiles by 8 K tiles on two cores",
     "parameters": {"variant": "row_sum_large", "M": 1280, "N": 2560, "K": 1024}},
    {"id": "row_sum_large_first_k_tail", "seed": 5612, "block_dim": 1,
     "purpose": "K=130 leaves the second K tile 2 rows deep in a never-written L1 slot: an "
                "unbounded contraction reads that slot's poison and every output returns NaN",
     "parameters": {"variant": "row_sum_large", "M": 64, "N": 512, "K": 130}},
    {"id": "row_sum_large_reuse", "seed": 5613, "block_dim": 1,
     "purpose": "Three tiles in every dimension: both passes' buffers and the K tail cycle together",
     "parameters": {"variant": "row_sum_large", "M": 192, "N": 768, "K": 260}},
    {"id": "row_sum_large_idle", "seed": 5614, "block_dim": 3,
     "purpose": "One M tile on three cores, two-pass form: the idle cores must still reach bar_all",
     "parameters": {"variant": "row_sum_large", "M": 64, "N": 256, "K": 128}},
    {"id": "row_sum_large_uneven_cores", "seed": 5615, "block_dim": 2,
     "purpose": "Three M tiles on two cores, so core 0 runs the per-tile bar_all twice and core 1 "
                "once: every other two-pass case is single-core, an even split, or all-or-nothing",
     "parameters": {"variant": "row_sum_large", "M": 192, "N": 512, "K": 256}},

    {"id": "row_l2_minimum", "seed": 5620, "block_dim": 1,
     "purpose": "One tile in every dimension through the sum-of-squares accumulator",
     "parameters": {"variant": "row_l2", "M": 64, "N": 256, "K": 128}},
    {"id": "row_l2_source", "seed": 5621, "block_dim": 2,
     "purpose": "The original M=512 shape across two cores",
     "parameters": {"variant": "row_l2", "M": 512, "N": 256, "K": 128}},
    {"id": "row_l2_reuse", "seed": 5622, "block_dim": 1,
     "purpose": "Three M tiles, three N tiles and two K tiles: the accumulator is re-zeroed per M tile",
     "parameters": {"variant": "row_l2", "M": 192, "N": 768, "K": 256}},
    {"id": "row_l2_idle", "seed": 5623, "block_dim": 3,
     "purpose": "One M tile on three cores, two-pass form with the L2 denominator",
     "parameters": {"variant": "row_l2", "M": 64, "N": 256, "K": 128}},
    {"id": "row_l2_k_unaligned", "seed": 5624, "block_dim": 1,
     "purpose": "K=130, the shape row_sum_large's first_k_tail case exists for: row_l2 bounds every "
                "operand by valid_k too, so it has no K alignment requirement to inherit",
     "parameters": {"variant": "row_l2", "M": 64, "N": 256, "K": 130}},

    {"id": "block_absmax_minimum", "seed": 5630, "block_dim": 1,
     "purpose": "One 128x128 block: one maximum, one division, no tail anywhere",
     "parameters": {"variant": "block_absmax", "M": 128, "N": 128, "K": 128}},
    {"id": "block_absmax_source", "seed": 5631, "block_dim": 2,
     "purpose": "The original shape: ten 128-column blocks per row across two cores",
     "parameters": {"variant": "block_absmax", "M": 512, "N": 1280, "K": 1024}},
    {"id": "block_absmax_tail", "seed": 5632, "block_dim": 2,
     "purpose": "M=65 gives the second vector half exactly one row, and K=130 a 2-deep K tail",
     "parameters": {"variant": "block_absmax", "M": 65, "N": 128, "K": 130}},
    {"id": "block_absmax_single", "seed": 5633, "block_dim": 3,
     "purpose": "One row and one K: the second vector half owns no rows at all, which is what "
                "the `if rows > 0` guard is for -- without it it would address GM[64:64]",
     "parameters": {"variant": "block_absmax", "M": 1, "N": 128, "K": 1}},
    {"id": "block_absmax_reuse", "seed": 5634, "block_dim": 1,
     "purpose": "Four M tiles with a 1-row tail, three column blocks and a K tail on one core",
     "parameters": {"variant": "block_absmax", "M": 385, "N": 384, "K": 260}},
]


def execute(case, inputs, launcher, backend):
    """Launch the variant this case names. The output is handed in poisoned with NaN and seeded
    into the launch, so a row no core writes reads back as NaN rather than as a plausible value.
    The two-pass variants also use this same tensor as their scratch: pass 1 parks the raw
    product in it and pass 2 reloads and overwrites it."""
    p = case["parameters"]
    out = torch.full((p["M"], p["N"]), float("nan"))
    op = OpExec(ENTRIES[p["variant"]], launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(inputs["x"], inputs["y"], out, p["M"], p["N"], p["K"])}


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
            print(f"{case['id']:28s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (M={p['M']} N={p['N']} K={p['K']}, launcher={args.launcher}, "
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
