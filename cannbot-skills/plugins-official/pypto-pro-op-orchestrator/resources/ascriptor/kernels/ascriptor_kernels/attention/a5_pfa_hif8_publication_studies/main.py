# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the seven HiFloat8 publication and protocol studies through OpExec and check them.

    python main.py                              # every case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --launcher aclnn --case all  # cce backend on this machine's card
    python main.py --launcher pypto --case nz3_fd

Seven complete attention kernels share one unscaled-HiFloat8 body, so what differs between any
two of them is stated and nothing else moves. Four are publication studies -- `nz1`-`nz4` are
four ways to get a HiFloat8 P tile into L1 in NZ, differing only in the softmax `@vf` pair the
shared body is handed. Three are synchronisation studies over ONE identical ND schedule:
`nd_fused`, `nd_mutexse` and `nd_splitevent` compute the same numbers under three protocols.
Every variant runs the same six geometries, which is what makes the rows comparable.

This unit records what each variant does, not which one wins: nothing here ranks them.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import (cprobe_base_kernel, cprobe_mutexse_kernel, cprobe_splitevent_kernel,
                    nzp_full_nz1_kernel, nzp_full_nz2_kernel, nzp_full_nz3_kernel,
                    nzp_full_nz4_kernel)
from reference import make_inputs, reference

KERNELS = {"nz1": nzp_full_nz1_kernel, "nz2": nzp_full_nz2_kernel,
           "nz3": nzp_full_nz3_kernel, "nz4": nzp_full_nz4_kernel,
           "nd_fused": cprobe_base_kernel, "nd_mutexse": cprobe_mutexse_kernel,
           "nd_splitevent": cprobe_splitevent_kernel}

# The oracle quantizes P to HiFloat8 through the same independent codec the kernels cast with,
# and `tile_sum` replays each variant's own addition order, so neither of those differences is
# hiding inside the budget. What is left is FP32 matrix rounding and the final BF16 cast -- a
# tolerance, not bit equality. The relative L2 residual is the part that still rejects a
# vacuous answer: zeros, or probabilities that were never quantized, pass atol on their own.
TOLERANCE = {"atol": 0.001, "rtol": 0.01, "max_relative_l2": 0.004}

CASES = [
    # nz1: MULTI4 squeeze, then a masked strided NZ store at pitch 129
    {"id": "nz1_aligned", "seed": 0, "block_dim": 1,
     "purpose": "Two aligned key tiles on one core: the publication path with no tail and no "
                "split",
     "parameters": {"variant": "nz1", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nz1_batch_tail", "seed": 1, "block_dim": 2,
     "purpose": "Two independent batches with partial query and key tiles; the storage a batch "
                "does not use stays poisoned",
     "parameters": {"variant": "nz1", "B": 2, "MQ": 65, "N": 133, "D": 128}},
    {"id": "nz1_fd", "seed": 2, "block_dim": 2,
     "purpose": "One query tile split between two core intervals, so the merge of the "
                "two owners' partial states runs",
     "parameters": {"variant": "nz1", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nz1_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Seven query tiles cross the three-slot state ring, and one of them is also "
                "split",
     "parameters": {"variant": "nz1", "B": 1, "MQ": 769, "N": 133, "D": 128}},
    {"id": "nz1_one_key_idle", "seed": 4, "block_dim": 3,
     "purpose": "A single query and a single key, with two idle launch cores that still join "
                "the barrier",
     "parameters": {"variant": "nz1", "B": 1, "MQ": 1, "N": 1, "D": 128}},
    {"id": "nz1_source_shape", "seed": 0, "block_dim": 4,
     "purpose": "The source's own default query/key shape on four cores",
     "parameters": {"variant": "nz1", "B": 1, "MQ": 2056, "N": 256, "D": 128}},

    # nz2: a uint8 gather at indices 0,4,...,252 feeding the same pitch-129 store
    {"id": "nz2_aligned", "seed": 0, "block_dim": 1,
     "purpose": "Two aligned key tiles on one core: the publication path with no tail and no "
                "split",
     "parameters": {"variant": "nz2", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nz2_batch_tail", "seed": 1, "block_dim": 2,
     "purpose": "Two independent batches with partial query and key tiles; the storage a batch "
                "does not use stays poisoned",
     "parameters": {"variant": "nz2", "B": 2, "MQ": 65, "N": 133, "D": 128}},
    {"id": "nz2_fd", "seed": 2, "block_dim": 2,
     "purpose": "One query tile split between two core intervals, so the merge of the "
                "two owners' partial states runs",
     "parameters": {"variant": "nz2", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nz2_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Seven query tiles cross the three-slot state ring, and one of them is also "
                "split",
     "parameters": {"variant": "nz2", "B": 1, "MQ": 769, "N": 133, "D": 128}},
    {"id": "nz2_one_key_idle", "seed": 4, "block_dim": 3,
     "purpose": "A single query and a single key, with two idle launch cores that still join "
                "the barrier",
     "parameters": {"variant": "nz2", "B": 1, "MQ": 1, "N": 1, "D": 128}},
    {"id": "nz2_source_shape", "seed": 0, "block_dim": 4,
     "purpose": "The source's own default query/key shape on four cores",
     "parameters": {"variant": "nz2", "B": 1, "MQ": 2056, "N": 256, "D": 128}},

    # nz3: squeeze, even-byte zero interleave, LOWHALF predicate, uint16 scatter (M10-031)
    {"id": "nz3_aligned", "seed": 0, "block_dim": 1,
     "purpose": "Two aligned key tiles on one core: the publication path with no tail and no "
                "split",
     "parameters": {"variant": "nz3", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nz3_batch_tail", "seed": 1, "block_dim": 2,
     "purpose": "Two independent batches with partial query and key tiles; the storage a batch "
                "does not use stays poisoned",
     "parameters": {"variant": "nz3", "B": 2, "MQ": 65, "N": 133, "D": 128}},
    {"id": "nz3_fd", "seed": 2, "block_dim": 2,
     "purpose": "One query tile split between two core intervals, so the merge of the "
                "two owners' partial states runs",
     "parameters": {"variant": "nz3", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nz3_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Seven query tiles cross the three-slot state ring, and one of them is also "
                "split",
     "parameters": {"variant": "nz3", "B": 1, "MQ": 769, "N": 133, "D": 128}},
    {"id": "nz3_one_key_idle", "seed": 4, "block_dim": 3,
     "purpose": "A single query and a single key, with two idle launch cores that still join "
                "the barrier",
     "parameters": {"variant": "nz3", "B": 1, "MQ": 1, "N": 1, "D": 128}},
    {"id": "nz3_source_shape", "seed": 0, "block_dim": 4,
     "purpose": "The source's own default query/key shape on four cores",
     "parameters": {"variant": "nz3", "B": 1, "MQ": 2056, "N": 256, "D": 128}},

    # nz4: a four-key native HiFloat8 deinterleave into four 32-key slabs at pitch 33
    {"id": "nz4_aligned", "seed": 0, "block_dim": 1,
     "purpose": "Two aligned key tiles on one core: the publication path with no tail and no "
                "split",
     "parameters": {"variant": "nz4", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nz4_batch_tail", "seed": 1, "block_dim": 2,
     "purpose": "Two independent batches with partial query and key tiles; the storage a batch "
                "does not use stays poisoned",
     "parameters": {"variant": "nz4", "B": 2, "MQ": 65, "N": 133, "D": 128}},
    {"id": "nz4_fd", "seed": 2, "block_dim": 2,
     "purpose": "One query tile split between two core intervals, so the merge of the "
                "two owners' partial states runs",
     "parameters": {"variant": "nz4", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nz4_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Seven query tiles cross the three-slot state ring, and one of them is also "
                "split",
     "parameters": {"variant": "nz4", "B": 1, "MQ": 769, "N": 133, "D": 128}},
    {"id": "nz4_one_key_idle", "seed": 4, "block_dim": 3,
     "purpose": "A single query and a single key, with two idle launch cores that still join "
                "the barrier",
     "parameters": {"variant": "nz4", "B": 1, "MQ": 1, "N": 1, "D": 128}},
    {"id": "nz4_source_shape", "seed": 0, "block_dim": 4,
     "purpose": "The source's own default query/key shape on four cores",
     "parameters": {"variant": "nz4", "B": 1, "MQ": 2056, "N": 256, "D": 128}},

    # nd_fused: ND pack4 with fragmented ND2NZ; end-pipe-only C/V mutexes
    {"id": "nd_fused_aligned", "seed": 0, "block_dim": 1,
     "purpose": "Two aligned key tiles on one core: the publication path with no tail and no "
                "split",
     "parameters": {"variant": "nd_fused", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nd_fused_batch_tail", "seed": 1, "block_dim": 2,
     "purpose": "Two independent batches with partial query and key tiles; the storage a batch "
                "does not use stays poisoned",
     "parameters": {"variant": "nd_fused", "B": 2, "MQ": 65, "N": 133, "D": 128}},
    {"id": "nd_fused_fd", "seed": 2, "block_dim": 2,
     "purpose": "One query tile split between two core intervals, so the merge of the "
                "two owners' partial states runs",
     "parameters": {"variant": "nd_fused", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nd_fused_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Seven query tiles cross the three-slot state ring, and one of them is also "
                "split",
     "parameters": {"variant": "nd_fused", "B": 1, "MQ": 769, "N": 133, "D": 128}},
    {"id": "nd_fused_one_key_idle", "seed": 4, "block_dim": 3,
     "purpose": "A single query and a single key, with two idle launch cores that still join "
                "the barrier",
     "parameters": {"variant": "nd_fused", "B": 1, "MQ": 1, "N": 1, "D": 128}},
    {"id": "nd_fused_source_shape", "seed": 0, "block_dim": 4,
     "purpose": "The source's own default query/key shape on four cores",
     "parameters": {"variant": "nd_fused", "B": 1, "MQ": 2056, "N": 256, "D": 128}},

    # nd_mutexse: the same ND schedule, with the PV and P start pipes equal to their end pipes
    {"id": "nd_mutexse_aligned", "seed": 0, "block_dim": 1,
     "purpose": "Two aligned key tiles on one core: the publication path with no tail and no "
                "split",
     "parameters": {"variant": "nd_mutexse", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nd_mutexse_batch_tail", "seed": 1, "block_dim": 2,
     "purpose": "Two independent batches with partial query and key tiles; the storage a batch "
                "does not use stays poisoned",
     "parameters": {"variant": "nd_mutexse", "B": 2, "MQ": 65, "N": 133, "D": 128}},
    {"id": "nd_mutexse_fd", "seed": 2, "block_dim": 2,
     "purpose": "One query tile split between two core intervals, so the merge of the "
                "two owners' partial states runs",
     "parameters": {"variant": "nd_mutexse", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nd_mutexse_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Seven query tiles cross the three-slot state ring, and one of them is also "
                "split",
     "parameters": {"variant": "nd_mutexse", "B": 1, "MQ": 769, "N": 133, "D": 128}},
    {"id": "nd_mutexse_one_key_idle", "seed": 4, "block_dim": 3,
     "purpose": "A single query and a single key, with two idle launch cores that still join "
                "the barrier",
     "parameters": {"variant": "nd_mutexse", "B": 1, "MQ": 1, "N": 1, "D": 128}},
    {"id": "nd_mutexse_source_shape", "seed": 0, "block_dim": 4,
     "purpose": "The source's own default query/key shape on four cores",
     "parameters": {"variant": "nd_mutexse", "B": 1, "MQ": 2056, "N": 256, "D": 128}},

    # nd_splitevent: the same ND schedule, plus two M-to-FIX SEvents inside the cube side
    {"id": "nd_splitevent_aligned", "seed": 0, "block_dim": 1,
     "purpose": "Two aligned key tiles on one core: the publication path with no tail and no "
                "split",
     "parameters": {"variant": "nd_splitevent", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nd_splitevent_batch_tail", "seed": 1, "block_dim": 2,
     "purpose": "Two independent batches with partial query and key tiles; the storage a batch "
                "does not use stays poisoned",
     "parameters": {"variant": "nd_splitevent", "B": 2, "MQ": 65, "N": 133, "D": 128}},
    {"id": "nd_splitevent_fd", "seed": 2, "block_dim": 2,
     "purpose": "One query tile split between two core intervals, so the merge of the "
                "two owners' partial states runs",
     "parameters": {"variant": "nd_splitevent", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "nd_splitevent_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Seven query tiles cross the three-slot state ring, and one of them is also "
                "split",
     "parameters": {"variant": "nd_splitevent", "B": 1, "MQ": 769, "N": 133, "D": 128}},
    {"id": "nd_splitevent_one_key_idle", "seed": 4, "block_dim": 3,
     "purpose": "A single query and a single key, with two idle launch cores that still join "
                "the barrier",
     "parameters": {"variant": "nd_splitevent", "B": 1, "MQ": 1, "N": 1, "D": 128}},
    {"id": "nd_splitevent_source_shape", "seed": 0, "block_dim": 4,
     "purpose": "The source's own default query/key shape on four cores",
     "parameters": {"variant": "nd_splitevent", "B": 1, "MQ": 2056, "N": 256, "D": 128}}
]


def execute(case, inputs, launcher, backend):
    """Pick the kernel this case's `variant` names and launch it. The output is handed in
    poisoned with NaN and seeded into the launch, so a query row no core claims reads back as
    NaN rather than as a plausible zero."""
    entry = KERNELS[case["parameters"]["variant"]]
    out = torch.full((inputs["B"] * inputs["MQ"], 128), float("nan"), dtype=torch.bfloat16)
    op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(inputs["q"], inputs["k"], inputs["v"], out,
                      inputs["B"], inputs["MQ"], inputs["N"], 128)}


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
    parser.add_argument("--variant", default="all",
                        help="run every case of one variant: " + ", ".join(KERNELS))
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:28s} {case['purpose']}")
        return 0

    selected = [case for case in CASES
                if args.case in ("all", case["id"])
                and args.variant in ("all", case["parameters"]["variant"])]
    if not selected:
        parser.error(f"no case named {args.case!r} for variant {args.variant!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (B={p['B']} MQ={p['MQ']} N={p['N']}, launcher={args.launcher}, "
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
