# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the scaled E4M3 decode kernel through OpExec and check it against the independent reference.

    python main.py                              # every case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --launcher aclnn --case all  # cce backend on this machine's card, every case
    python main.py --launcher pypto --case source_long_tail

One query row per head against the whole key history, streamed 256 keys at a time. The three
scales are runtime scalars, not constants: `scale_q * scale_k / sqrt(128)` folds into the score
scale and `scale_v / 16` into the output scale, which is where the x16 probability headroom is
paid back. The row sum is accumulated from the FP32 probabilities BEFORE they are cast to E4M3,
so the denominator never sees the low-precision carrier.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import mha_ifa_fp8_scale_256
from reference import make_inputs, reference

# Kernel and reference agree on the precision order -- x16, cast to E4M3, correct by scale_v/16 --
# but the kernel rounds each 256-key tile on chip while the reference rounds in Torch, so a long
# key tail accumulates more than a ULP of difference. The relative L2 ceiling is what rejects a
# vacuous answer: every element is divided by the row sum, so atol on its own would accept an
# output that is merely uniformly small.
TOLERANCE = {"atol": 0.05, "rtol": 0.05, "max_relative_l2": 0.02}

CASES = [
    {"id": "aligned", "seed": 42, "block_dim": 2,
     "purpose": "Two heads on two cores, one exact 256-key tile each: no tail anywhere",
     "parameters": {"BH": 2, "S": 256, "L": 1, "D": 128}},
    {"id": "tail_idle", "seed": 43, "block_dim": 3,
     "purpose": "257 keys is a full tile plus a one-key tail; one head on three cores leaves two idle",
     "parameters": {"BH": 1, "S": 257, "L": 1, "D": 128}},
    {"id": "reuse_heads", "seed": 44, "block_dim": 2,
     "purpose": "Three heads over two cores: one core runs two heads, so the row state and the "
                "on-chip buffers must be re-initialised between them",
     "parameters": {"BH": 3, "S": 769, "L": 1, "D": 128}},
    {"id": "source_long_tail", "seed": 7, "block_dim": 1,
     "purpose": "The original source shape: eight full tiles and a one-key ninth on a single core, "
                "which is the longest rescale chain the online row state has to survive",
     "parameters": {"BH": 1, "S": 2049, "L": 1, "D": 128}},
]


def execute(case, inputs, launcher, backend):
    """Launch the kernel. The output is handed in poisoned with NaN and seeded into the launch,
    so a row the kernel never writes reads back as NaN instead of as a plausible zero."""
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    out = torch.full(q.shape, float("nan"))
    op = OpExec(mha_ifa_fp8_scale_256, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(q, k, v, out, q.shape[0], 1, k.shape[1], 128,
                      *(inputs[name] for name in ("scale_q", "scale_k", "scale_v")))}


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
        print(f"{case['id']}  (BH={p['BH']} S={p['S']}, launcher={args.launcher}, "
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
