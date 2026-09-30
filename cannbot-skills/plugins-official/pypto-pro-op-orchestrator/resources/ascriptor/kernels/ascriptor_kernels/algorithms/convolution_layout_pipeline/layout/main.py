# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the NCHW -> NC1HWC0 conversion through OpExec and check every output bit.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn --case ub_pad_channel_spatial_tail

This is the layout level of the convolution_layout_pipeline family: the first stage of that
convolution, on its own, with nothing after it. Two bodies do the same job and differ only in
who zeroes the channel tail. `host_pad` is handed a grid the host already padded to C1*16
channels, so the kernel only loads and transposes. `ub_pad` is handed the C real channels and
zeroes the tail on chip with `dup`, which means a V-pipe write and an MTE2-pipe load both
target src_buf; the source's `t5_dup_ready` / `t5_load_valid` SEvent pair is what keeps that
from being a WAW race. Each case runs in both variants so the two are directly comparable.

Spatial padding is a separate matter: HW is rounded up to 16 on the way in and those rows are
dropped on the way out, so the result has exactly HW rows per plane in both variants.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import kernel_for
from reference import geometry, make_inputs, reference

# Bitwise. Nothing is computed here -- every output half is some input half moved to a new
# address, or a zero the channel tail is defined to hold. There is no arithmetic to round, so
# a tolerance could only hide a misplaced element, a dropped spatial pad row, or a tail the
# kernel forgot to clear.
TOLERANCE = None

CASES = [
    {"id": "host_pad_source", "seed": 4, "block_dim": 1,
     "purpose": "The source shape B2/C20/HW36 with the host supplying the channel padding",
     "parameters": {"variant": "host_pad", "B": 2, "C": 20, "C1": 2, "HW": 36}},
    {"id": "host_pad_full_channels", "seed": 2, "block_dim": 1,
     "purpose": "C = 16 exactly: one full C0 block, no channel tail to zero",
     "parameters": {"variant": "host_pad", "B": 2, "C": 16, "C1": 1, "HW": 36}},
    {"id": "host_pad_idle_owner", "seed": 3, "block_dim": 1,
     "purpose": "One (b, c1) block of work over two lanes, so vector 1 must exit writing nothing",
     "parameters": {"variant": "host_pad", "B": 1, "C": 1, "C1": 1, "HW": 17}},
    {"id": "host_pad_channel_spatial_tail", "seed": 7, "block_dim": 1,
     "purpose": "C31 and HW45 together: a one-channel tail and three dropped spatial pad rows",
     "parameters": {"variant": "host_pad", "B": 1, "C": 31, "C1": 2, "HW": 45}},

    {"id": "ub_pad_source", "seed": 5, "block_dim": 1,
     "purpose": "The source shape B2/C20/HW36 with the kernel zeroing the channel tail on chip",
     "parameters": {"variant": "ub_pad", "B": 2, "C": 20, "C1": 2, "HW": 36}},
    {"id": "ub_pad_full_channels", "seed": 2, "block_dim": 1,
     "purpose": "C = 16 exactly: the on-chip dup runs but has no tail left to clear",
     "parameters": {"variant": "ub_pad", "B": 2, "C": 16, "C1": 1, "HW": 36}},
    {"id": "ub_pad_idle_owner", "seed": 3, "block_dim": 1,
     "purpose": "One block over two lanes, with the dup/load ready-valid handshake in the loop",
     "parameters": {"variant": "ub_pad", "B": 1, "C": 1, "C1": 1, "HW": 17}},
    {"id": "ub_pad_channel_spatial_tail", "seed": 7, "block_dim": 1,
     "purpose": "C31 and HW45: the fifteen tail lanes come from dup, not from the host grid",
     "parameters": {"variant": "ub_pad", "B": 1, "C": 31, "C1": 2, "HW": 45}},

    {"id": "host_pad_original_seed4", "seed": 4, "block_dim": 1,
     "purpose": "The original driver's direct-FP16 draws at seed 4, host_pad",
     "parameters": {"variant": "host_pad", "B": 2, "C": 20, "C1": 2, "HW": 36,
                    "sampling": "source_fp16"}},
    {"id": "ub_pad_original_seed5", "seed": 5, "block_dim": 1,
     "purpose": "The original driver's direct-FP16 draws at seed 5, ub_pad",
     "parameters": {"variant": "ub_pad", "B": 2, "C": 20, "C1": 2, "HW": 36,
                    "sampling": "source_fp16"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the variant this case names. The two bodies take a different fourth argument:
    host_pad counts C1 blocks because the host already rounded the channels up, ub_pad counts
    the C real channels because it derives C1 and the tail width itself. The [144, 16]
    destination is poisoned with NaN and seeded into the launch, so a row the kernel never
    writes reads back as NaN rather than as the zero a channel tail legitimately holds -- and
    only the declared B*C1*HW rows are taken from it."""
    p = case["parameters"]
    c1, padded_hw, _, _ = geometry(p)
    output = torch.full((144, 16), float("nan"), dtype=torch.float16)
    op = OpExec(kernel_for(p["variant"], "a2"), launcher=launcher, backend=backend, device="a2",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    channels = c1 if p["variant"] == "host_pad" else p["C"]
    actual = op(inputs["data"], output, p["B"], channels, p["HW"], padded_hw)
    return {"output": actual[:p["B"] * c1 * p["HW"]].clone()}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail = torch.equal(got, want), "  bitwise"
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (bounds.get("atol", 0.0) + bounds.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok, detail = torch.allclose(got.float(), want.float(), **bounds), f"  allclose={margin:.2f}x ({bounds})"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
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
            print(f"{case['id']:30s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['variant']}, B={p['B']} C={p['C']} HW={p['HW']}, "
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
