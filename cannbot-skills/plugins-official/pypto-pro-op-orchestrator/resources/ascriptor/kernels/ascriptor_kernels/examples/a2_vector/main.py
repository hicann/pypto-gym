# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the A2/A3 masked-scale kernel through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator, a2 facade
    python main.py --device a3              # the same source through the a3 facade
    python main.py --launcher pipesim
    python main.py --launcher aclnn --case multi_core

One kernel body serves both devices: `build_kernel` binds it to `ascriptor.a2` or
`ascriptor.a3` at call time, so the algorithm is written once and the facade chooses the
device profile. Import exactly one facade per process — the module cache does not restore
an earlier target — which is why a device switch means a new `python main.py --device ...`.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import build_kernel
from reference import make_inputs, reference

# Bitwise. The kernel multiplies elementwise in float32 and then by a scalar, in that order,
# and the reference performs the same two operations in the same order; there is no reduction
# and no atomic, so nothing can legitimately reorder and any difference is a defect.
TOLERANCE = None

CASES = [
    {"id": "tail_one", "seed": 8200, "block_dim": 1,
     "purpose": "n = 1: a single live lane in a 128-element tile",
     "parameters": {"n": 1, "tile_len": 128, "scale": 0.75, "dtype": "float32"}},
    {"id": "tail_63", "seed": 8201, "block_dim": 1,
     "purpose": "n = 63: a tail just under the tile",
     "parameters": {"n": 63, "tile_len": 128, "scale": 0.75, "dtype": "float32"}},
    {"id": "vector_64", "seed": 8202, "block_dim": 1,
     "purpose": "n = 64: exactly one vector register, no tail",
     "parameters": {"n": 64, "tile_len": 128, "scale": 0.75, "dtype": "float32"}},
    {"id": "tail_65", "seed": 8203, "block_dim": 1,
     "purpose": "n = 65: one element past a register, the cheapest way to break a tail",
     "parameters": {"n": 65, "tile_len": 128, "scale": 0.75, "dtype": "float32"}},
    {"id": "multi_tile", "seed": 8204, "block_dim": 1,
     "purpose": "n = 257: three tiles on one core, the last one partial",
     "parameters": {"n": 257, "tile_len": 128, "scale": 0.75, "dtype": "float32"}},
    {"id": "multi_core", "seed": 8205, "block_dim": 2,
     "purpose": "n = 515 over two cores: the tile split must not overlap or leave a gap",
     "parameters": {"n": 515, "tile_len": 128, "scale": 0.75, "dtype": "float32"}},
    {"id": "idle_cores", "seed": 8206, "block_dim": 4,
     "purpose": "n = 129 over four cores: two cores get a tile, two get nothing and must exit cleanly",
     "parameters": {"n": 129, "tile_len": 128, "scale": 0.75, "dtype": "float32"}},
    {"id": "tile_capacity", "seed": 8207, "block_dim": 1,
     "purpose": "tile_len at the full 8192-element UB buffer, plus one element of tail",
     "parameters": {"n": 8193, "tile_len": 8192, "scale": 0.75, "dtype": "float32"}},
]


def execute(case, inputs, launcher, device):
    """The output tensor is handed in poisoned with NaN and seeded into the launch, so a lane
    the kernel skips reads back as NaN instead of as the zero a masked lane also produces."""
    entry = build_kernel(device)
    op = OpExec(entry, launcher=launcher, backend="cce", device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"y": op(inputs["x"], inputs["mask"], inputs["y"],
                    inputs["n"], inputs["scale"], inputs["tile_len"])}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want) if tolerance is None else torch.allclose(got, want, **tolerance)
    if tolerance is None:
        detail = "  bitwise"
    else:
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (tolerance.get("atol", 0.0) + tolerance.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        detail = f"  allclose={margin:.2f}x ({tolerance})"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:6s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok:
        # Destinations arrive NaN-poisoned, so an element still NaN was never written.
        outside = (got != want) if tolerance is None else ~torch.isclose(got.float(), want.float(), **tolerance)
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
        print(f"{case['id']}  (n={case['parameters']['n']}, device={args.device}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
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
