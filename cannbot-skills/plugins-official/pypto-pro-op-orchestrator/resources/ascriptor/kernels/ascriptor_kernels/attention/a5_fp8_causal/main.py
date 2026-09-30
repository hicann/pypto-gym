# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the causal E5M2 attention kernel through OpExec and check it against the reference.

    python main.py                              # every case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --launcher aclnn --case all  # cce backend on this machine's card, every case
    python main.py --launcher pypto --case tail

Q, K, V and the probability matrix handed to PV are E5M2; the scores, the running maximum,
the exponentials, the rescale and the output stay FP32. The row sum is accumulated from the
FP32 probabilities BEFORE they are cast, so the denominator never sees the low-precision
carrier -- which is why this demo publishes `rowmax` and `rowsum` as outputs of their own
instead of only the attention result.

The causal boundary is never stored: a diagonal tile rebuilds its mask from `cols.arange(...)`
and one `compare`, and each vector sub-block owns a different 64-row band, so the two bodies
`apply_causal_diagonal_sb0` and `apply_causal_diagonal_sb1` differ only in the offset they
compare against.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import flash_attn_full_fp8_causal_kernel
from reference import make_inputs, reference

# `out` is wide because every probability entering PV is an E5M2 number with two mantissa bits,
# so a single element can move several percent while the answer as a whole is right; the
# relative L2 ceiling is what rejects a vacuous output that atol alone would accept.
# `rowmax` and `rowsum` are FP32 quantities defined before any cast, so they get a rule three
# orders of magnitude tighter: at this width a wrong causal mask or a wrong score scale shows
# up as a rowmax defect instead of hiding inside the quantization noise of `out`, and a kernel
# that summed the probabilities after the E5M2 cast fails on rowsum alone.
TOLERANCE = {"default": {"rtol": 0.2, "atol": 0.2, "max_relative_l2": 0.03},
             "rowmax": {"rtol": 1e-05, "atol": 0.0001, "max_relative_l2": 1e-05},
             "rowsum": {"rtol": 1e-05, "atol": 0.0001, "max_relative_l2": 1e-05}}

CASES = [
    {"id": "aligned", "seed": 42, "block_dim": 1,
     "purpose": "One exact 128x128 tile on one core: the diagonal path with no tail of any kind",
     "parameters": {"BH": 1, "S1": 128, "S2": 128, "D": 128}},
    {"id": "tail", "seed": 43, "block_dim": 2,
     "purpose": "129 queries and 133 keys: a one-row M tail tile and a five-column N tail, so "
                "the diagonal mask and the separate N-tail mask both run",
     "parameters": {"BH": 1, "S1": 129, "S2": 133, "D": 128}},
    {"id": "rectangular", "seed": 44, "block_dim": 2,
     "purpose": "More query tiles than key tiles: the later M tiles have no diagonal at all and "
                "see every key, while the first one is still masked",
     "parameters": {"BH": 1, "S1": 257, "S2": 133, "D": 128}},
    {"id": "heads_idle", "seed": 45, "block_dim": 3,
     "purpose": "Two heads over three cores leaves one idle, and 65 query rows put exactly one "
                "row in the second sub-block's 64-row band",
     "parameters": {"BH": 2, "S1": 65, "S2": 73, "D": 128}},
    {"id": "source_long_tail", "seed": 42, "block_dim": 2,
     "purpose": "The original source shape: nine full M tiles and a one-row tenth against nine "
                "full N tiles and a 33-column tail",
     "parameters": {"BH": 1, "S1": 1153, "S2": 1185, "D": 128}},
]


def execute(case, inputs, launcher, backend):
    """Launch the kernel. Every destination is handed in poisoned with NaN and seeded into the
    launch, so a row no core writes reads back as NaN rather than as a plausible number.

    The private row statistics are allocated with an eight-float head stride, so that every head
    and every 64-row sub-block starts on a 32-byte GM transaction boundary; without it two heads
    share a transaction and race. The padding belongs to the output, and the host slices the
    declared rows back out here rather than the kernel packing them."""
    p = case["parameters"]
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    out = torch.full(q.shape, float("nan"))
    stats_stride = (p["S1"] + 7) // 8 * 8
    rowmax = torch.full((p["BH"] * stats_stride,), float("nan"))
    rowsum = torch.full_like(rowmax, float("nan"))
    op = OpExec(flash_attn_full_fp8_causal_kernel, launcher=launcher, backend=backend,
                device="a5", block_dim=case["block_dim"], out_dir=f"tmp/{launcher}",
                seed_outputs=True)
    values = op(q, k, v, out, rowmax, rowsum, p["BH"], p["S1"], p["S2"], 128, 128 ** -0.5)
    return {"out": values[0],
            **{name: value.reshape(p["BH"], stats_stride)[:, :p["S1"]].contiguous().reshape(-1)
               for name, value in zip(("rowmax", "rowsum"), values[1:], strict=True)}}


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
            print(f"{case['id']:18s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (BH={p['BH']} S1={p['S1']} S2={p['S2']}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            rule = TOLERANCE.get(name, TOLERANCE["default"])
            if not compare(name, actual[name], expected[name], rule):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
