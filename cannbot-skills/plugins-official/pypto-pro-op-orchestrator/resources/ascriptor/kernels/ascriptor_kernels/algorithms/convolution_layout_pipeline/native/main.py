# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the four native convolution bodies through OpExec and check them against FP64 Torch.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case large_two_batches
    python main.py --backend pto_isa --launcher board --case basic_channel_tail

This is the native level of the convolution_layout_pipeline family: the cube body alone, fed
image and weights the host has already packed into NC1HWC0 and the Cout/C1/Kh/Kw/C0 fractal
order. `layout` is the packing stage on its own, `stepwise` runs the packing and this cube as
four separate launches, and `integrated` fuses all four into one MIX launch. Nothing here
transposes anything: this is the convolution and only the convolution.

Four bodies, selected per case by `mode`: `basic` is plain FP16 accumulation, `bias` adds the
BT bias on the first K tile, `dilation` combines stride 2 with dilation 2, and `large` keeps a
4x4 asymmetric filter resident in L1 while reloading the feature map once per batch.

Every case checks two tensors. `output` is the logical NCHW result. `physical_nz` is every cell
the cube actually wrote, including the M_PAD rows past HO*WO and the zero-padded channels up to
the rounded Cout -- the window keeps sliding into that M padding (D-223) and those rows are real
computed values, not garbage to be ignored.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

# pypto_pro: `conv2d` desugars, once per K tile, into `dma.l1_to_l0.img2col` -- the im2col
#   feature-map move from L1 into L0A (see the k_tile helper in the desugar pass). That opcode
#   is outside the pypto surface, so the backend has no instruction to emit for the A operand
#   of every mmad in all four bodies. The cce and pto_isa backends run every case; --launcher
#   pypto fails at the board stage, on every case, with the upstream gap A5-UP-005, "op
#   dma.l1_to_l0.img2col is outside the pypto surface". It is upstream-owned and there is
#   nothing to route around in this kernel: any convolution that reaches the cube through
#   `conv2d` lands on the same opcode.

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import kernel_for
from reference import geometry, make_inputs, reference

# FP16 operands summed in an FP32 accumulator cannot match an FP64 convolution exactly, and the
# device chooses its own K-tile order, which the reference deliberately does not reproduce. The
# elementwise bounds are the source's own. The relative-L2 limit is what an empty result fails:
# a kernel that writes zeros, or that stops after the first K tile, stays inside atol over most
# of the output. The integer and bias-only cases are the stricter controls -- their arithmetic
# is exact, so they should land on 0.
TOLERANCE = {"rtol": 0.003, "atol": 0.003, "max_relative_l2": 0.001}

CASES = [
    {"id": "basic_channel_tail", "seed": 0, "block_dim": 1,
     "purpose": "C20 into C1=2 and COUT40 into N=3: both tails are zero-padded on the host",
     "parameters": {"mode": "basic", "C": 20, "H": 8, "W": 8, "COUT": 40, "B": 1,
                    "M_PAD": 64, "HO": 8, "WO": 8, "pattern": "random"}},
    {"id": "basic_full_channels", "seed": 101, "block_dim": 1,
     "purpose": "C32 and COUT48 fill the capacities exactly; integers make the sum exact",
     "parameters": {"mode": "basic", "C": 32, "H": 8, "W": 8, "COUT": 48, "B": 1,
                    "M_PAD": 64, "HO": 8, "WO": 8, "pattern": "integer"}},
    {"id": "basic_spatial_tail", "seed": 102, "block_dim": 1,
     "purpose": "H7: 56 real output positions inside M_PAD 64, so 8 padded-M rows are computed",
     "parameters": {"mode": "basic", "C": 20, "H": 7, "W": 8, "COUT": 40, "B": 1,
                    "M_PAD": 64, "HO": 7, "WO": 8, "pattern": "integer"}},
    {"id": "bias_channel_tail", "seed": 1, "block_dim": 1,
     "purpose": "C31/COUT47: the bias must stop at 47 and leave the 48th channel at zero",
     "parameters": {"mode": "bias", "C": 31, "H": 8, "W": 8, "COUT": 47, "B": 1,
                    "M_PAD": 64, "HO": 8, "WO": 8, "pattern": "random"}},
    {"id": "bias_two_k_tiles", "seed": 104, "block_dim": 1,
     "purpose": "Zero image, C32 so K spans two 144 tiles: the bias must be added on the first "
                "tile only, and the answer is the bias alone",
     "parameters": {"mode": "bias", "C": 32, "H": 8, "W": 8, "COUT": 48, "B": 1,
                    "M_PAD": 64, "HO": 8, "WO": 8, "pattern": "bias_only"}},
    {"id": "bias_spatial_tail", "seed": 105, "block_dim": 1,
     "purpose": "H7 with a bias: the padded-M rows carry the bias too, not zeros",
     "parameters": {"mode": "bias", "C": 17, "H": 7, "W": 8, "COUT": 33, "B": 1,
                    "M_PAD": 64, "HO": 7, "WO": 8, "pattern": "random"}},
    {"id": "dilation_stride_tail", "seed": 2, "block_dim": 1,
     "purpose": "12x12 under stride 2 and dilation 2 with pad 2: a 6x6 plane, C13 and COUT23 "
                "both tails",
     "parameters": {"mode": "dilation", "C": 13, "H": 12, "W": 12, "COUT": 23, "B": 1,
                    "M_PAD": 48, "HO": 6, "WO": 6, "pattern": "random"}},
    {"id": "dilation_full_channels", "seed": 107, "block_dim": 1,
     "purpose": "C16 and COUT32 exactly fill the dilated body's capacities",
     "parameters": {"mode": "dilation", "C": 16, "H": 12, "W": 12, "COUT": 32, "B": 1,
                    "M_PAD": 48, "HO": 6, "WO": 6, "pattern": "integer"}},
    {"id": "large_two_batches", "seed": 7, "block_dim": 1,
     "purpose": "4x4 filter, 14x16 plane, two batches: the weights stay in L1 while the feature "
                "map is reloaded per batch",
     "parameters": {"mode": "large", "C": 32, "H": 14, "W": 16, "COUT": 64, "B": 2,
                    "M_PAD": 224, "HO": 14, "WO": 16, "pattern": "random"}},
    {"id": "large_asymmetric_image", "seed": 109, "block_dim": 1,
     "purpose": "16x14, the other valid 224-position valuation, against the asymmetric (1,2,1,2) "
                "pad an even filter needs",
     "parameters": {"mode": "large", "C": 20, "H": 16, "W": 14, "COUT": 64, "B": 2,
                    "M_PAD": 224, "HO": 16, "WO": 14, "pattern": "integer"}},
    {"id": "basic_original_seed0", "seed": 0, "block_dim": 1,
     "purpose": "The source's own shape and direct-FP16 generator for basic",
     "parameters": {"mode": "basic", "C": 20, "H": 8, "W": 8, "COUT": 40, "B": 1,
                    "M_PAD": 64, "HO": 8, "WO": 8, "pattern": "source_random"}},
    {"id": "bias_original_seed1", "seed": 1, "block_dim": 1,
     "purpose": "The source's own shape and direct-FP16 generator for bias, random FP32 bias",
     "parameters": {"mode": "bias", "C": 31, "H": 8, "W": 8, "COUT": 47, "B": 1,
                    "M_PAD": 64, "HO": 8, "WO": 8, "pattern": "source_random"}},
    {"id": "dilation_original_seed2", "seed": 2, "block_dim": 1,
     "purpose": "The source's own C12/COUT24 shape and direct-FP16 generator for dilation",
     "parameters": {"mode": "dilation", "C": 12, "H": 12, "W": 12, "COUT": 24, "B": 1,
                    "M_PAD": 48, "HO": 6, "WO": 6, "pattern": "source_random"}},
    {"id": "basic_minimum_channels", "seed": 13, "block_dim": 1,
     "purpose": "C1 and COUT1: fifteen of the sixteen lanes in the only computed N block are "
                "padding, and must still be what the reference says",
     "parameters": {"mode": "basic", "C": 1, "H": 8, "W": 8, "COUT": 1, "B": 1,
                    "M_PAD": 64, "HO": 8, "WO": 8, "pattern": "integer"}},
    {"id": "bias_minimum_channels", "seed": 13, "block_dim": 1,
     "purpose": "The same minimum with a bias and a zero image: only channel 0 carries it",
     "parameters": {"mode": "bias", "C": 1, "H": 8, "W": 8, "COUT": 1, "B": 1,
                    "M_PAD": 64, "HO": 8, "WO": 8, "pattern": "bias_only"}},
    {"id": "dilation_minimum_channels", "seed": 13, "block_dim": 1,
     "purpose": "The same minimum under stride 2 and dilation 2",
     "parameters": {"mode": "dilation", "C": 1, "H": 12, "W": 12, "COUT": 1, "B": 1,
                    "M_PAD": 48, "HO": 6, "WO": 6, "pattern": "integer"}},
]

OUTPUTS = ("output", "physical_nz")


def execute(case, inputs, launcher, backend):
    """Launch the body this case names. The scalars are the runtime shape -- the L1 and L0C
    tensors are always allocated at the mode's static worst case, and h/w/c/cout/m_pad mask the
    tail inside them. `large` takes the batch count as a sixth scalar because it loops over
    batches itself. The destination is poisoned with NaN and seeded into the launch, so a cell
    the M loop never reaches reads back as NaN rather than as the zero a padded channel
    legitimately holds."""
    p = case["parameters"]
    g = geometry(p)
    out = torch.full((p["B"] * (g["cout_pad"] // 16) * p["M_PAD"], 16), float("nan"))
    tensors = [inputs["data"], inputs["packed_weights"]]
    if p["mode"] == "bias":
        tensors.append(inputs["packed_bias"])
    tensors.append(out)
    scalars = [p[name] for name in ("H", "W", "C", "COUT", "M_PAD")]
    if p["mode"] == "large":
        scalars.append(p["B"])
    op = OpExec(kernel_for(p["mode"], "a5"), launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    actual = op(*tensors, *scalars)

    nchw = actual.reshape(p["B"], g["cout_pad"] // 16, p["M_PAD"], 16).permute(0, 1, 3, 2)
    logical = nchw.reshape(p["B"], g["cout_pad"], p["M_PAD"])[:, :p["COUT"], :p["HO"] * p["WO"]]
    computed_n = (p["COUT"] + 15) // 16
    physical = actual.reshape(p["B"], g["cout_pad"] // 16, p["M_PAD"], 16)[:, :computed_n]
    return {"output": logical.contiguous().reshape(p["B"], p["COUT"], p["HO"], p["WO"]),
            "physical_nz": physical.contiguous().reshape(-1, 16)}


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
            print(f"{case['id']:26s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['mode']}, B={p['B']} C={p['C']} {p['H']}x{p['W']} "
              f"COUT={p['COUT']}, launcher={args.launcher}, block_dim={case['block_dim']})")
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
