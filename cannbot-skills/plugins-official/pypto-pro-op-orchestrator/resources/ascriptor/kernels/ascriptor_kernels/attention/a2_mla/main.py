# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the A2/A3 MLA attention kernel through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator, a2 facade
    python main.py --device a3              # the same source through the a3 facade
    python main.py --launcher pipesim
    python main.py --launcher aclnn --case source_tail

The geometry is fixed: one batch, eight query heads, four KV heads, and a score built from
both feature groups -- 512 non-rotary plus 64 rotary columns -- so query head `h` reads KV
head `h // 2`. The kernel streams 128-column KV tiles with online softmax state, narrows P
to FP16 before PV, and keeps the maximum, denominator, numerator and public output in FP32.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import build_kernel
from reference import make_inputs, reference

# Not bitwise: P is narrowed to FP16 before PV, and the FP32 numerator and denominator are
# rescaled once per 128-column KV tile, so the kernel's summation order is the tiling and the
# reference reproduces that tiling but not the on-chip reduction tree inside a tile. The
# relative L2 bound is what rejects a kernel that dropped the rotary term or the head mapping
# and still landed inside the elementwise bounds on most rows.
TOLERANCE = {"atol": 0.0002, "rtol": 0.001, "max_relative_l2": 0.001}

CASES = [
    {"id": "partial", "seed": 42, "block_dim": 3,
     "purpose": "S=16, SKV=32 on three cores: one partial KV tile and a short M tail",
     "parameters": {"B": 1, "HQ": 8, "HKV": 4, "Dn": 512, "Dr": 64, "S": 16, "SKV": 32,
                    "v_mode": "k_nope"}},
    {"id": "one_key", "seed": 1, "block_dim": 20,
     "purpose": "A single query against a single key on 20 cores: the online state never updates",
     "parameters": {"B": 1, "HQ": 8, "HKV": 4, "Dn": 512, "Dr": 64, "S": 1, "SKV": 1,
                    "v_mode": "independent"}},
    {"id": "source_aligned", "seed": 2, "block_dim": 20,
     "purpose": "S=128, SKV=256: exactly one M tile and two full KV tiles, no tail anywhere",
     "parameters": {"B": 1, "HQ": 8, "HKV": 4, "Dn": 512, "Dr": 64, "S": 128, "SKV": 256,
                    "v_mode": "k_nope"}},
    {"id": "source_tail", "seed": 3, "block_dim": 20,
     "purpose": "S=257, SKV=260: tails on both axes, and M tiles owned more than once per core",
     "parameters": {"B": 1, "HQ": 8, "HKV": 4, "Dn": 512, "Dr": 64, "S": 257, "SKV": 260,
                    "v_mode": "independent"}},
]


def execute(case, inputs, launcher, device):
    """Launch the kernel. The FP32 output is handed in poisoned with NaN and seeded into the
    launch, so a row no M tile claimed reads back as NaN rather than as a plausible zero."""
    out = torch.full((8 * inputs["S"], 512), float("nan"), dtype=torch.float32)
    op = OpExec(build_kernel(device), launcher=launcher, backend="cce", device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(*(inputs[name] for name in ("q_nope", "q_rope", "k_nope", "k_rope", "v")),
                      out, inputs["S"], inputs["SKV"], 576 ** -0.5)}


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
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (S={p['S']} SKV={p['SKV']}, v={p['v_mode']}, "
              f"device={args.device}, launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.device)
        for name in expected:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
