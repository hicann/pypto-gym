# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run online MX quantization and its matrix product through OpExec and check both.

    python main.py                          # every case, functional simulator
    python main.py --stages                 # also check the raw payload and scale bytes
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn         # cce backend, on this machine's card
    python main.py --launcher pypto --case nd_scale_byte_zero

Two complete paths share one recipe. `nd` takes two FP32 [16, 64] matrices whose rows are the
logical rows; `transposed` takes two FP32 [64, 32] matrices whose columns are, and its VF
interleaves neighbouring row bytes and scatters UINT16 pairs rather than writing bytes one at
a time. Both pick one E8M0 scale per 32 logical K values, encode E4M3 payload online, and feed
the MX cube product, so the FP32 output is the only thing the composition publishes.

`--stages` additionally launches the two leaf kernels, which run the same VFs on one vector
owner and publish the payload and scale bytes the composition keeps inside L1. Those bytes are
fully determined, so they are compared bitwise: a wrong exponent, a midpoint rounded the wrong
way or a scattered byte pair landing in the wrong lane shows up there as an exact mismatch,
while in the product it would only be a small numerical difference.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import (float_to_mxfp8_online_cast_matmul_kernel,
                    float_to_mxfp8_online_cast_transpose_matmul_kernel,
                    quantize_nd_leaf, quantize_transposed_leaf)
from reference import geometry, make_inputs, reference, reference_stages

PRODUCT = {"nd": float_to_mxfp8_online_cast_matmul_kernel,
           "transposed": float_to_mxfp8_online_cast_transpose_matmul_kernel}
LEAF = {"nd": quantize_nd_leaf, "transposed": quantize_transposed_leaf}

STAGE_OUTPUTS = ("payload_x", "scales_x", "payload_y", "scales_y")
OUTPUTS = ("output",)

# The product accumulates quantized operands in the cube while the reference takes an FP64 dot
# product of the decoded ones, so the two differ by the accumulation order alone -- hence the
# source's own pointwise pair. The 0.001 relative residual is what refuses an erased tiny
# product or a dropped group, which atol 0.002 on its own would sign off wherever the true
# product is small. The payload and scale bytes get no tolerance at all: exponent selection and
# finite E4M3 nearest-even rounding determine every one of those bits.
TOLERANCE = {"default": {"rtol": 0.002, "atol": 0.002, "max_relative_l2": 0.001},
             "payload_x": None, "scales_x": None, "payload_y": None, "scales_y": None}

CASES = [
    {"id": "nd_source", "seed": 1, "block_dim": 1,
     "purpose": "The original ND generator verbatim: seed 1, x then y, randn * 16",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "random"}},
    {"id": "nd_midpoints", "seed": 14001, "block_dim": 1,
     "purpose": "E4M3 midpoints of both parities beside a tiny value and a negative zero",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "midpoints"}},
    {"id": "nd_group_exponents", "seed": 14002, "block_dim": 1,
     "purpose": "A different exponent in every row and group, so a shared or stale scale shows",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "group_exponents"}},
    {"id": "nd_last_group_scale", "seed": 14003, "block_dim": 1,
     "purpose": "Only the final row's second group is extreme (2^20 against 2^-12): the last "
                "scale slot is the one most easily left unwritten",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "last_group_scale"}},
    {"id": "nd_zero", "seed": 14004, "block_dim": 1,
     "purpose": "All-zero x with negative zeros in the odd columns: signed zero has to survive "
                "normalization and encoding",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "zero"}},
    {"id": "nd_scale_byte_zero", "seed": 14005, "block_dim": 1,
     "purpose": "x = 2^-127 everywhere, so the exponent bits are 0: this consumer reads scale "
                "byte 0 as 2^-127, not as zero",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "scale_byte_zero"}},
    {"id": "nd_subnormal_payload", "seed": 14006, "block_dim": 1,
     "purpose": "Subnormal and minimum-normal magnitudes against y = 2^40",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "subnormal_payload"}},
    {"id": "nd_small_output", "seed": 14007, "block_dim": 1,
     "purpose": "Both operands at 2^-16: the product is far below atol, so only the relative "
                "residual can tell a real answer from an erased one",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "small_output"}},
    {"id": "nd_idle_groups", "seed": 14008, "block_dim": 3,
     "purpose": "Three MIX groups for one work item: two groups stay idle and must publish nothing",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "random"}},
    {"id": "nd_midpoint_sweep", "seed": 14009, "block_dim": 1,
     "purpose": "Every E4M3 midpoint and its two FP32 neighbours, in both signs: nearest-even "
                "has to break ties both up and down",
     "parameters": {"variant": "nd", "M": 16, "N": 16, "K": 64, "input_rows": 16,
                    "input_cols": 64, "pattern": "midpoint_sweep"}},

    {"id": "transposed_source", "seed": 2, "block_dim": 1,
     "purpose": "The original transposed generator verbatim: seed 2, columns as logical rows",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "random"}},
    {"id": "transposed_midpoints", "seed": 14101, "block_dim": 1,
     "purpose": "The midpoint set through the interleave-and-scatter payload path",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "midpoints"}},
    {"id": "transposed_group_exponents", "seed": 14102, "block_dim": 1,
     "purpose": "Per-row, per-group exponents over 32 logical rows: twice the ND scale row",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "group_exponents"}},
    {"id": "transposed_last_group_scale", "seed": 14103, "block_dim": 1,
     "purpose": "The last of 64 packed scale bytes, which is also the end of the second burst",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "last_group_scale"}},
    {"id": "transposed_zero", "seed": 14104, "block_dim": 1,
     "purpose": "Signed zeros surviving the UINT16 pair scatter, which moves two bytes at once",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "zero"}},
    {"id": "transposed_scale_byte_zero", "seed": 14105, "block_dim": 1,
     "purpose": "Exponent bits 0 on the transposed path: the 2^-127 clamp, not a zero scale",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "scale_byte_zero"}},
    {"id": "transposed_subnormal_payload", "seed": 14106, "block_dim": 1,
     "purpose": "Subnormal and minimum-normal magnitudes against y = 2^40",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "subnormal_payload"}},
    {"id": "transposed_small_output", "seed": 14107, "block_dim": 1,
     "purpose": "Both operands at 2^-16, so only the relative residual bounds the answer",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "small_output"}},
    {"id": "transposed_idle_groups", "seed": 14108, "block_dim": 3,
     "purpose": "Three MIX groups for one work item on the transposed path",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "random"}},
    {"id": "transposed_midpoint_sweep", "seed": 14109, "block_dim": 1,
     "purpose": "Every E4M3 midpoint and its FP32 neighbours through the scatter path",
     "parameters": {"variant": "transposed", "M": 32, "N": 32, "K": 64, "input_rows": 64,
                    "input_cols": 32, "pattern": "midpoint_sweep"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the composition for this case's variant. It quantizes both of its own inputs,
    publishes payload and scales into two rotating L1 slots and runs the MX product. The
    destination is handed in poisoned with NaN and seeded into the launch, so a cell the cube
    never writes reads back as NaN rather than as a plausible zero."""
    rows, _ = geometry(inputs["parameters"])
    entry = PRODUCT[inputs["parameters"]["variant"]]
    op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{entry.name}",
                seed_outputs=True)
    output = torch.full((rows, rows), float("nan"))
    return {"output": op(inputs["x"], inputs["y"], output, 0)}


def execute_stages(case, inputs, launcher, backend):
    """Run the same VF on its own, once per input, to see what the composition keeps in L1.
    The leaf owns one vector core whatever the case's block_dim says, because its whole job is
    to publish one owner's bytes. Both destinations start at byte 255, which is not a legal
    scale here, so a byte no lane writes is visible instead of reading back as a plausible 0.
    Each input gets its own out_dir so a later launch cannot overwrite its peer's artifacts."""
    rows, shape = geometry(inputs["parameters"])
    leaf = LEAF[inputs["parameters"]["variant"]]
    results = {}
    for name in ("x", "y"):
        op = OpExec(leaf, launcher=launcher, backend=backend, device="a5", block_dim=1,
                    out_dir=f"tmp/{launcher}/{leaf.name}/{name}", seed_outputs=True)
        payload = torch.full(shape, 255, dtype=torch.uint8)
        scales = torch.full((1, rows * 2), 255, dtype=torch.uint8)
        codes, packed = op(inputs[name], payload, scales, 0)
        # The leaf transports scales as one packed 32/64-byte row; the host only reshapes that
        # unchanged byte sequence into the logical [row, group] order.
        results[f"payload_{name}"] = codes
        results[f"scales_{name}"] = packed.reshape(rows, 2)
    return results


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
    parser.add_argument("--stages", action="store_true",
                        help="also launch the leaf kernels and compare the raw payload and "
                             "scale bytes, not only the product")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:30s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['variant']}, {p['input_rows']}x{p['input_cols']} -> "
              f"{p['M']}x{p['N']}, launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        if args.stages:
            expected.update(reference_stages(inputs))
            actual.update(execute_stages(case, inputs, args.launcher, args.backend))
        for name in (STAGE_OUTPUTS + OUTPUTS if args.stages else OUTPUTS):
            tolerance = TOLERANCE.get(name, TOLERANCE["default"])
            if not compare(name, actual[name], expected[name], tolerance):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
