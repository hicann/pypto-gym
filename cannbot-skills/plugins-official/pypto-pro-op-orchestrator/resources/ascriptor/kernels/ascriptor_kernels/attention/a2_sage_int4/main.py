# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the A2/A3 signed-int4 SAGE decode kernel through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator, a2 facade
    python main.py --device a3              # the same source through the a3 facade
    python main.py --launcher pipesim
    python main.py --launcher aclnn --case source_2044

Decode-only, non-causal attention over a quantized KV cache. Q and K arrive as signed int4
values packed eight to an int32 carrier, V as int8, and the smoothing correction `qm @
k_smooth^T` as FP16. The online maximum and denominator stay FP32; the probabilities are
scaled by 127 and narrowed float32 -> float16 -> int8 before PV. Three outputs are published:
the FP32 row of attention output, and the two FP32 row statistics.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import build_kernel
from reference import make_inputs, reference

# Every narrowing boundary in this kernel is reproduced exactly by the reference -- the int4
# unpack, the FP16 smoothed-score store, the float32->float16->int8 probability rounding, the
# int8 PV -- so the only difference left is the order in which the FP32 tile results are
# reduced. That is why the bounds are tight rather than generous.
#
# rowmax and rowsum get their own rule with no residual bound: they are single scalars per
# head, computed before probability quantization, so a relative L2 over a [BH, 1] tensor would
# say nothing that the elementwise bound has not already said.
TOLERANCE = {"default": {"atol": 1e-05, "rtol": 1e-05, "max_relative_l2": 1e-05},
             "rowmax": {"atol": 1e-05, "rtol": 1e-05},
             "rowsum": {"atol": 1e-05, "rtol": 1e-05}}

CASES = [
    {"id": "signed_aligned", "seed": 7, "block_dim": 2,
     "purpose": "Exactly one full 512-key tile: no tail anywhere, signed carrier controls",
     "parameters": {"BH": 2, "S2": 512, "D": 128, "TILES": 1, "pattern": "signed"}},
    {"id": "signed_tail_reuse", "seed": 11, "block_dim": 2,
     "purpose": "513 keys: a second tile holding one key, and two heads per core reusing workspaces",
     "parameters": {"BH": 4, "S2": 513, "D": 128, "TILES": 2, "pattern": "signed"}},
    {"id": "idle_small", "seed": 17, "block_dim": 3,
     "purpose": "17 keys over three cores: a very short tail, and a third core that owns no head",
     "parameters": {"BH": 2, "S2": 17, "D": 128, "TILES": 1, "pattern": "signed"}},
    {"id": "source_453", "seed": 42, "block_dim": 20,
     "purpose": "The original 32-head source length 453, at the source quantization order",
     "parameters": {"BH": 32, "S2": 453, "D": 128, "TILES": 1, "pattern": "source"}},
    {"id": "source_2044", "seed": 42, "block_dim": 20,
     "purpose": "The original source length 2044: four tiles, so the online rescale runs three times",
     "parameters": {"BH": 32, "S2": 2044, "D": 128, "TILES": 4, "pattern": "source"}},
    {"id": "signed_453", "seed": 23, "block_dim": 2,
     "purpose": "The 453-key tail with signed-carrier controls, two heads on two cores",
     "parameters": {"BH": 2, "S2": 453, "D": 128, "TILES": 1, "pattern": "signed"}},
    {"id": "signed_453_bd20", "seed": 29, "block_dim": 20,
     "purpose": "32 heads over 20 cores: uneven ownership, twelve cores taking a second head",
     "parameters": {"BH": 32, "S2": 453, "D": 128, "TILES": 1, "pattern": "signed"}},
    {"id": "signed_453_bd20_full", "seed": 31, "block_dim": 20,
     "purpose": "40 heads over 20 cores: every core owns exactly two and none is idle",
     "parameters": {"BH": 40, "S2": 453, "D": 128, "TILES": 1, "pattern": "signed"}},
    {"id": "signed_453_bd16", "seed": 37, "block_dim": 16,
     "purpose": "The same 32 heads at a different launch width: two per core over 16 cores",
     "parameters": {"BH": 32, "S2": 453, "D": 128, "TILES": 1, "pattern": "signed"}},
    {"id": "signed_513_bd20", "seed": 41, "block_dim": 20,
     "purpose": "A tile boundary and uneven ownership together: 513 keys, 32 heads, 20 cores",
     "parameters": {"BH": 32, "S2": 513, "D": 128, "TILES": 2, "pattern": "signed"}},
]

INPUTS = ("q", "k", "v", "scale_q", "scale_k", "scale_v", "qm", "k_smooth")
OUTPUTS = ("out", "rowmax", "rowsum")


def execute(case, inputs, launcher, device):
    """Launch the kernel. All three destinations are handed in poisoned with NaN and seeded
    into the launch: only sub-block 0 publishes the single decode row, so a head that no core
    claimed must read back as NaN rather than as a zero row with a zero denominator."""
    bh = inputs["BH"]
    outputs = (torch.full((bh, 1, 128), float("nan"), dtype=torch.float32),
               torch.full((bh, 1), float("nan"), dtype=torch.float32),
               torch.full((bh, 1), float("nan"), dtype=torch.float32))
    op = OpExec(build_kernel(device), launcher=launcher, backend="cce", device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    actual = op(*(inputs[name] for name in INPUTS), *outputs, bh, inputs["S2"])
    return dict(zip(OUTPUTS, actual, strict=True))


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
            print(f"{case['id']:24s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (BH={p['BH']} S2={p['S2']}, {p['pattern']}, device={args.device}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.device)
        for name in OUTPUTS:
            rule = TOLERANCE.get(name, TOLERANCE["default"])
            if not compare(name, actual[name], expected[name], rule):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
