# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the all-HiFloat8 PFA variants through OpExec and check them against the oracle.

    python main.py                              # every case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --launcher aclnn --case all  # cce backend on this machine's card
    python main.py --variant v9                 # one variant's cases only

The variant list is the lesson index, and every variant is a complete attention kernel:

    nd     P published key-major in ND
    nz     the promoted V6 -- NZ built directly in UB with a four-key deinterleave at
           physical stride 33, with the exponent biased by ln16 so P and its denominator
           both carry a factor of 16
    v8     three physical 128-key score tiles under one maximum update, so three P beats
           accumulate into a single L0C PV result
    v9     the score exported to FP16 through two SINGLE drains with a FixP19 scale, and
           an FP16 online state

Four answers to "how does P get from the vector side back to the cube". The five `study_*`
cases are the other axis: one schedule held fixed while only the place the softmax scale is
applied and the precision the state carries move.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

# pypto_pro: `dma.l0c_to_ub` with an FP32->FP32 scale. PyPTO emits TMOV's scalar overload,
#   whose PTO dtype selector picks NoQuant for FP32 (and vector quantization for E4M3), and the
#   Python move API has no parameter for asking it to quantize -- so the scale is silently
#   dropped and the FP32 board control returns the unscaled result. That is upstream's gap
#   A5-UP-008, not a restriction of ours. It costs exactly three of the forty cases here:
#   study_v1_matched, study_v1_split and study_v1_tail, the only cases that apply the FixP19
#   scale at a drain whose destination is also FP32. study_v2 applies the same scale into an
#   FP16 destination and v9 exports through FP16 too, so both take a different overload and are
#   unaffected. The cce and pto_isa backends run all forty; --launcher pypto runs the other
#   thirty-seven and, on those three, comes back with a result that is off by the scale factor
#   rather than failing to emit.

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import (pfa_fd_v6_allhif8_kernel, pfa_fd_v9_allhif8_kernel,
                    qk_softmax_pv_hif8_kernel, qk_softmax_pv_v1_kernel, qk_softmax_pv_v2_kernel,
                    qk_softmax_pv_v3_kernel, qk_softmax_pv_v3_splitn_kernel,
                    qk_softmax_pv_v4_kernel, v8_allhif8_kernel)
from reference import make_inputs, reference

KERNELS = {"nd": qk_softmax_pv_hif8_kernel, "nz": pfa_fd_v6_allhif8_kernel,
           "v8": v8_allhif8_kernel, "v9": pfa_fd_v9_allhif8_kernel,
           "study_v1": qk_softmax_pv_v1_kernel, "study_v2": qk_softmax_pv_v2_kernel,
           "study_v3": qk_softmax_pv_v3_kernel, "study_v4": qk_softmax_pv_v4_kernel,
           "study_v3_splitn": qk_softmax_pv_v3_splitn_kernel}

# kernel.py also carries five `*_nocv_kernel` entrypoints. They are the same five study bodies
# with their cube-to-vector mutexes replaced by a no-op, so the missing handoff shows up as a
# wrong value instead of a hang; they exist to measure the unsynchronised parallel ceiling and
# are incorrect by construction. No case runs them, and a run of one proves nothing.

# The oracle replays each variant's own arithmetic -- the FixP19 truncation, the FP16 state and
# reduction order, the ln16 probability bias, the HiFloat8 cast -- so none of those differences
# is hiding inside the budget. What is left is FP32 matrix rounding and the final BF16 cast, so
# this is a tolerance rather than bit equality. The relative L2 residual is the part that still
# rejects a vacuous answer: zeros, or an unquantized probability, pass atol on their own.
TOLERANCE = {"atol": 0.001, "rtol": 0.01, "max_relative_l2": 0.004}

CASES = [
    # nd: key-major ND probability, FP32 state, unscaled P
    {"id": "nd_matched", "seed": 0, "block_dim": 2,
     "purpose": "Eight whole tasks over two cores: no ragged tile, no shared query tile",
     "parameters": {"variant": "nd", "B": 1, "MQ": 256, "N": 512, "D": 128}},
    {"id": "nd_split", "seed": 1, "block_dim": 2,
     "purpose": "Four key tiles of one query tile split over two cores, so the two-owner merge "
                "produces every output row",
     "parameters": {"variant": "nd", "B": 1, "MQ": 128, "N": 512, "D": 128}},
    {"id": "nd_batch_tail", "seed": 2, "block_dim": 2,
     "purpose": "Two batches with a one-row query tail and a five-key tail: the masked tail "
                "softmax and the PV that reads only the valid V rows",
     "parameters": {"variant": "nd", "B": 2, "MQ": 129, "N": 133, "D": 128}},
    {"id": "nd_reuse", "seed": 3, "block_dim": 3,
     "purpose": "Twenty tasks over three cores: the three-slot cache ring wraps and two "
                "different query tiles each end up with two owners",
     "parameters": {"variant": "nd", "B": 1, "MQ": 513, "N": 385, "D": 128}},

    # nz: promoted V6 -- NZ built in UB by four-key deinterleave at stride 33, P and denominator x16
    {"id": "nz_matched", "seed": 0, "block_dim": 2,
     "purpose": "The four-key deinterleave publication with no tail and no shared query tile",
     "parameters": {"variant": "nz", "B": 1, "MQ": 256, "N": 512, "D": 128}},
    {"id": "nz_split", "seed": 1, "block_dim": 2,
     "purpose": "One query tile split over two cores: the x16 probability scale has to survive "
                "the merge, where it cancels against the denominator",
     "parameters": {"variant": "nz", "B": 1, "MQ": 128, "N": 512, "D": 128}},
    {"id": "nz_batch_tail", "seed": 2, "block_dim": 2,
     "purpose": "Two batches with a one-row query tail and a five-key tail: the corrected "
                "single NEG fill of the invalid score rows",
     "parameters": {"variant": "nz", "B": 2, "MQ": 129, "N": 133, "D": 128}},
    {"id": "nz_reuse", "seed": 3, "block_dim": 3,
     "purpose": "Twenty tasks over three cores: cache ring wrap with two split query tiles",
     "parameters": {"variant": "nz", "B": 1, "MQ": 513, "N": 385, "D": 128}},

    # v8: logical 384-key groups, one shared maximum per group, final group must hold 129..384 keys
    {"id": "v8_matched", "seed": 0, "block_dim": 2,
     "purpose": "Two full 384-key groups per query tile over two cores: three P beats retire "
                "into one L0C PV result per group",
     "parameters": {"variant": "v8", "B": 1, "MQ": 256, "N": 768, "D": 128}},
    {"id": "v8_split", "seed": 1, "block_dim": 2,
     "purpose": "One query tile's two groups owned by different cores, so the shared per-group "
                "maximum still has to merge correctly across the boundary",
     "parameters": {"variant": "v8", "B": 1, "MQ": 128, "N": 768, "D": 128}},
    {"id": "v8_batch_tail", "seed": 2, "block_dim": 2,
     "purpose": "A single 133-key group, just inside V8's 129-key floor, with a one-row query "
                "tail: the second tile is empty and must not retire a beat",
     "parameters": {"variant": "v8", "B": 2, "MQ": 129, "N": 133, "D": 128}},
    {"id": "v8_tail3_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Three groups ending in a 259-key final group, over two cores with a split "
                "query tile: the credit ring rotates and the tail group is partial",
     "parameters": {"variant": "v8", "B": 1, "MQ": 257, "N": 1027, "D": 128}},

    # study_v1: FixP19 scale taken at the fixpipe drain, FP32 score and state
    {"id": "study_v1_matched", "seed": 0, "block_dim": 1,
     "purpose": "Two aligned key tiles on one core: the FP32 baseline for the scale-placement "
                "comparison",
     "parameters": {"variant": "study_v1", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "study_v1_split", "seed": 1, "block_dim": 2,
     "purpose": "The same arithmetic with one query tile owned by two cores",
     "parameters": {"variant": "study_v1", "B": 1, "MQ": 128, "N": 512, "D": 128}},
    {"id": "study_v1_tail", "seed": 2, "block_dim": 1,
     "purpose": "A 65-row query tile and a five-key tail on the fixpipe-scaled FP32 path",
     "parameters": {"variant": "study_v1", "B": 1, "MQ": 65, "N": 133, "D": 128}},

    # study_v2: the same FixP19 fixpipe scale, but FP16 score and FP16 online state
    {"id": "study_v2_matched", "seed": 0, "block_dim": 1,
     "purpose": "Same scale placement as study_v1 with FP16 state: what the narrower state "
                "costs is the only thing that moved",
     "parameters": {"variant": "study_v2", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "study_v2_split", "seed": 1, "block_dim": 2,
     "purpose": "FP16 state across a two-owner merge, which is computed in FP32",
     "parameters": {"variant": "study_v2", "B": 1, "MQ": 128, "N": 512, "D": 128}},
    {"id": "study_v2_tail", "seed": 2, "block_dim": 1,
     "purpose": "The FP16 tail score load: a half-register UNPK reads exactly 64 half elements "
                "per allocated row, then deinterleaves back into lane order",
     "parameters": {"variant": "study_v2", "B": 1, "MQ": 65, "N": 133, "D": 128}},

    # study_v3: the same schedule with the scale applied in the vector function, FP32 state
    {"id": "study_v3_matched", "seed": 0, "block_dim": 1,
     "purpose": "The scale moved from the fixpipe into the vector function, FP32 throughout: "
                "the control for study_v1",
     "parameters": {"variant": "study_v3", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "study_v3_split", "seed": 1, "block_dim": 2,
     "purpose": "VF-scaled FP32 with one query tile owned by two cores",
     "parameters": {"variant": "study_v3", "B": 1, "MQ": 128, "N": 512, "D": 128}},
    {"id": "study_v3_tail", "seed": 2, "block_dim": 1,
     "purpose": "A 65-row query tile and a five-key tail on the VF-scaled FP32 path",
     "parameters": {"variant": "study_v3", "B": 1, "MQ": 65, "N": 133, "D": 128}},

    # study_v4: raw score rounded to FP16 first, then scaled in the vector function, FP16 state
    {"id": "study_v4_matched", "seed": 0, "block_dim": 1,
     "purpose": "The raw score rounds to FP16 before it is scaled, so the rounding happens on "
                "the larger number: the counterpart of study_v2",
     "parameters": {"variant": "study_v4", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "study_v4_split", "seed": 1, "block_dim": 2,
     "purpose": "The FP16-first path across a two-owner merge",
     "parameters": {"variant": "study_v4", "B": 1, "MQ": 128, "N": 512, "D": 128}},
    {"id": "study_v4_tail", "seed": 2, "block_dim": 1,
     "purpose": "The FP16-first path through the half-register UNPK tail load",
     "parameters": {"variant": "study_v4", "B": 1, "MQ": 65, "N": 133, "D": 128}},

    # study_v3_splitn: study_v3's arithmetic through ONE SPLITN fixpipe bridge, not dual-SINGLE
    {"id": "study_v3_splitn_matched", "seed": 0, "block_dim": 1,
     "purpose": "study_v3's numbers through a single SPLITN drain instead of two SINGLE ones; "
                "SPLITN cannot requantize, which is why this control needs scale 1.0",
     "parameters": {"variant": "study_v3_splitn", "B": 1, "MQ": 128, "N": 256, "D": 128}},
    {"id": "study_v3_splitn_split", "seed": 1, "block_dim": 2,
     "purpose": "The SPLITN bridge with one query tile owned by two cores",
     "parameters": {"variant": "study_v3_splitn", "B": 1, "MQ": 128, "N": 512, "D": 128}},
    {"id": "study_v3_splitn_tail", "seed": 2, "block_dim": 1,
     "purpose": "The SPLITN bridge on a 65-row query tile with a five-key tail",
     "parameters": {"variant": "study_v3_splitn", "B": 1, "MQ": 65, "N": 133, "D": 128}},

    # v9: FP16 FixP-scaled score exports and FP16 online state; three P beats for every tail
    {"id": "v9_matched", "seed": 0, "block_dim": 2,
     "purpose": "Two full 384-key groups over two cores: the ordinary three-beat rotation",
     "parameters": {"variant": "v9", "B": 1, "MQ": 256, "N": 768, "D": 128}},
    {"id": "v9_split", "seed": 1, "block_dim": 2,
     "purpose": "One query tile's two groups owned by different cores, with FP16 partial state "
                "merged in FP32",
     "parameters": {"variant": "v9", "B": 1, "MQ": 128, "N": 768, "D": 128}},
    {"id": "v9_single_tail", "seed": 2, "block_dim": 1,
     "purpose": "A single 73-key group under a 65-row query tile: both tails at once",
     "parameters": {"variant": "v9", "B": 1, "MQ": 65, "N": 73, "D": 128}},
    {"id": "v9_tail3_reuse", "seed": 3, "block_dim": 2,
     "purpose": "Three groups ending in a 259-key final group, over two cores with a split "
                "query tile",
     "parameters": {"variant": "v9", "B": 1, "MQ": 257, "N": 1027, "D": 128}},
    {"id": "v9_one_key", "seed": 7, "block_dim": 1,
     "purpose": "A final group of exactly one key: P1 and P2 are phase-only and still retire "
                "their beats through the same event pair a real beat uses",
     "parameters": {"variant": "v9", "B": 1, "MQ": 129, "N": 1, "D": 128}},
    {"id": "v9_p0_tail127", "seed": 8, "block_dim": 1,
     "purpose": "P0 stops one key short of a full tile, so the tail predicate is exercised on "
                "the first beat rather than the last",
     "parameters": {"variant": "v9", "B": 1, "MQ": 129, "N": 127, "D": 128}},
    {"id": "v9_p0_full128", "seed": 9, "block_dim": 1,
     "purpose": "P0 exactly full, P1 and P2 phase-only: the boundary either side of "
                "v9_p0_tail127",
     "parameters": {"variant": "v9", "B": 1, "MQ": 129, "N": 128, "D": 128}},
    {"id": "v9_short_phase_reuse", "seed": 10, "block_dim": 1,
     "purpose": "A 129-key group repeated over three query tiles: a one-key P1 and a "
                "phase-only P2 recur, which is where a leaked beat would accumulate",
     "parameters": {"variant": "v9", "B": 1, "MQ": 257, "N": 129, "D": 128}},
    {"id": "v9_p1_tail255", "seed": 11, "block_dim": 1,
     "purpose": "P1 ends one key short of its tile, with P2 phase-only",
     "parameters": {"variant": "v9", "B": 1, "MQ": 129, "N": 255, "D": 128}},
    {"id": "v9_two_tiles256", "seed": 12, "block_dim": 1,
     "purpose": "P0 and P1 full, P2 phase-only: the boundary either side of v9_p1_tail255",
     "parameters": {"variant": "v9", "B": 1, "MQ": 129, "N": 256, "D": 128}},
    {"id": "v9_p2_one257", "seed": 13, "block_dim": 1,
     "purpose": "P2 carries exactly one key: the shortest real third beat there is",
     "parameters": {"variant": "v9", "B": 1, "MQ": 129, "N": 257, "D": 128}},
    {"id": "v9_p2_tail383", "seed": 14, "block_dim": 1,
     "purpose": "P2 ends one key short of a full 384-key group: the longest partial tail",
     "parameters": {"variant": "v9", "B": 1, "MQ": 129, "N": 383, "D": 128}},
    {"id": "v9_phase_after_full", "seed": 15, "block_dim": 1,
     "purpose": "A 129-key final group after two full ones: the short group has to follow a "
                "full one without inheriting its slot state",
     "parameters": {"variant": "v9", "B": 1, "MQ": 129, "N": 897, "D": 128}},
]


def execute(case, inputs, launcher, backend):
    """Pick the kernel this case's `variant` names and launch it. The output is handed in
    poisoned with NaN and seeded into the launch, so a query row no core claims reads back as
    NaN rather than as a plausible zero."""
    entry = KERNELS[case["parameters"]["variant"]]
    out = torch.full(inputs["q"].shape, float("nan"), dtype=torch.bfloat16)
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
            print(f"{case['id']:24s} {case['purpose']}")
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
