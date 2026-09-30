# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the four streamed decode variants through OpExec and check them against one reference.

    python main.py                              # every case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --launcher aclnn --case all  # cce backend on this machine's card, every case
    python main.py --launcher pypto --case nz128_matched

This is a controlled 2x2, not four kernels that happen to live together: {ND, NZ} probability
publication crossed with a {128, 256} streamed key width, over bodies that are otherwise the
same. The entire structural difference between `mha_ifa_v2` and `mha_ifa_nz` is the publish
line -- `l1p[...] <<= ub_p_half[...]` against `<<= ub_p_nz[...].nz()` -- and the NZ pair pays
for it with a `pack_p_to_nz_row_*` pass and a staging buffer exactly `rows` tall.

Every variant gets the same inputs from the same seed and is compared against the same
full-FP32 softmax, so the four are directly comparable: `..._matched` is the same shape in
all four, and only the variant changes.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import mha_ifa_256, mha_ifa_nz, mha_ifa_nz_256, mha_ifa_v2
from reference import make_inputs, reference

# The kernel streams the softmax and rounds the probability tile to FP16 before PV, while the
# reference takes one FP32 softmax over the whole key range; only P is rounded, and the row
# sum stays FP32 on both sides. The relative L2 ceiling is what rejects a vacuous output: the
# result is a convex combination of V rows, so atol alone accepts an answer that is merely
# plausible everywhere.
TOLERANCE = {"rtol": 0.001, "atol": 0.001, "max_relative_l2": 0.005}

# variant name -> the kernel entry and the key tile it streams. S must be a multiple of the
# tile: none of these bodies has a partial-tile path.
VARIANTS = {"nd128": (mha_ifa_v2, 128), "nz128": (mha_ifa_nz, 128),
            "nd256": (mha_ifa_256, 256), "nz256": (mha_ifa_nz_256, 256)}

CASES = [
    {"id": "nd128_matched", "seed": 42, "block_dim": 2,
     "purpose": "Two heads on two cores, four 128-key tiles each: the shared shape every "
                "variant runs, with an ND probability publish",
     "parameters": {"variant": "nd128", "BH": 2, "S": 512, "L": 1, "D": 128}},
    {"id": "nd128_idle", "seed": 43, "block_dim": 3,
     "purpose": "One head on three cores: two cores are assigned nothing and must write nothing",
     "parameters": {"variant": "nd128", "BH": 1, "S": 256, "L": 1, "D": 128}},
    {"id": "nd128_reuse", "seed": 44, "block_dim": 2,
     "purpose": "Three heads over two cores: one core runs two heads, so the row state and the "
                "on-chip buffers are re-initialised between them",
     "parameters": {"variant": "nd128", "BH": 3, "S": 1024, "L": 1, "D": 128}},

    {"id": "nz128_matched", "seed": 42, "block_dim": 2,
     "purpose": "The NZ counterpart of nd128_matched: same inputs, same schedule, compact-NZ "
                "publish one fractal row tall",
     "parameters": {"variant": "nz128", "BH": 2, "S": 512, "L": 1, "D": 128}},
    {"id": "nz128_idle", "seed": 43, "block_dim": 3,
     "purpose": "One head on three cores under the NZ publish: two idle cores write nothing",
     "parameters": {"variant": "nz128", "BH": 1, "S": 256, "L": 1, "D": 128}},
    {"id": "nz128_reuse", "seed": 44, "block_dim": 2,
     "purpose": "Three heads over two cores under the NZ publish: the one-row staging buffer "
                "is reused tile after tile and must not publish a fractal column it never wrote",
     "parameters": {"variant": "nz128", "BH": 3, "S": 1024, "L": 1, "D": 128}},

    {"id": "nd256_matched", "seed": 42, "block_dim": 2,
     "purpose": "The same shape at a 256-key tile: two tiles instead of four, with splitk=64 "
                "on QK and splitn=64 on PV",
     "parameters": {"variant": "nd256", "BH": 2, "S": 512, "L": 1, "D": 128}},
    {"id": "nd256_idle", "seed": 43, "block_dim": 3,
     "purpose": "A single 256-key tile on three cores: one tile, one head, two idle cores",
     "parameters": {"variant": "nd256", "BH": 1, "S": 256, "L": 1, "D": 128}},
    {"id": "nd256_reuse", "seed": 44, "block_dim": 2,
     "purpose": "Three heads over two cores at the wider tile: buffer reuse between heads with "
                "half as many rescale steps as the 128 variant",
     "parameters": {"variant": "nd256", "BH": 3, "S": 1024, "L": 1, "D": 128}},

    {"id": "nz256_matched", "seed": 42, "block_dim": 2,
     "purpose": "The fourth corner of the 2x2: 256-key tile with the NZ publish, whose pack "
                "writes two half-rows and has to know where block 8 starts",
     "parameters": {"variant": "nz256", "BH": 2, "S": 512, "L": 1, "D": 128}},
    {"id": "nz256_idle", "seed": 43, "block_dim": 3,
     "purpose": "One 256-key tile, NZ publish, two idle cores",
     "parameters": {"variant": "nz256", "BH": 1, "S": 256, "L": 1, "D": 128}},
    {"id": "nz256_reuse", "seed": 44, "block_dim": 2,
     "purpose": "Three heads over two cores with the two-register NZ pack repeated per tile",
     "parameters": {"variant": "nz256", "BH": 3, "S": 1024, "L": 1, "D": 128}},
]


def execute(case, inputs, launcher, backend):
    """Launch the variant this case names. The output is handed in poisoned with NaN and
    seeded into the launch, so a head no core claims reads back as NaN rather than as a
    plausible number -- which is what the `_idle` cases are for."""
    entry, _tile = VARIANTS[case["parameters"]["variant"]]
    q, k, v = (inputs[name] for name in ("q", "k", "v"))
    out = torch.full_like(q, float("nan"), dtype=torch.float32)
    op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(q, k, v, out, q.shape[0], 1, k.shape[1], 128)}


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
            print(f"{case['id']:16s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (variant={p['variant']}, BH={p['BH']} S={p['S']}, "
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
