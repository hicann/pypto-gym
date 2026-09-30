# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the per-task metadata attention studies through OpExec and check them against the oracle.

    python main.py                              # every case, functional simulator
    python main.py --list                       # the case ids, with their purpose
    python main.py --launcher aclnn --case all  # cce backend on this machine's card
    python main.py --case meta11_split_tail

Two kernels share one unscaled-HiFloat8 body and differ only in how much of the per-task
geometry they read from GM: `meta8` loads eight int32 fields per (query tile, key tile) task
and recomputes the cache slot, the q-buffer parity and the first/last-key tests in core;
`meta11` reads those four as fields too. Every case runs both ABIs over the same geometry, so
the pair is the comparison.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

# pypto_pro: the L0/L1 staging extents here are read out of the `meta` tensor at run time
#   (`valid_n`, `v_rows` and `valid_m` are GetValueFrom loads), so no constant typed capacity is
#   proven for the reinterpreted HiFloat8 carriers and the emitter stops at `mem.reinterpret`
#   with "dynamic L0 staging extent has no proven typed capacity". The cce and pto_isa backends
#   run all twelve cases; --launcher pypto fails during emission, before anything reaches a
#   board, on all twelve. This refusal is ours and deliberate -- RFC-0013 requires a statically
#   proven local capacity -- not an upstream defect waiting on a backend fix, so do not read it
#   as a missing feature. An older board pass exists from before the guard, taken when the
#   emitter accepted the raw byte-carrier axis as the typed capacity; it is history and does not
#   qualify this source. Closing it means giving the kernel a specialised or statically branched
#   extent instead of a run-time one.

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import qk_softmax_pv_v6_meta_ext_kernel, qk_softmax_pv_v6_meta_kernel
from reference import make_inputs, reference

KERNELS = {"meta8": qk_softmax_pv_v6_meta_kernel, "meta11": qk_softmax_pv_v6_meta_ext_kernel}

# The oracle quantizes P to HiFloat8 through the same independent codec the kernel's cast
# implements, so what is left between them is FP32 matrix and reduction order plus the final
# BF16 rounding -- a tolerance, not bit equality. The relative L2 residual is the part that
# still rejects a wrong answer: a kernel that skipped the probability quantization, or wrote
# zeros, passes atol on its own.
TOLERANCE = {"atol": 0.001, "rtol": 0.01, "max_relative_l2": 0.004}

CASES = [
    {"id": "meta8_aligned", "seed": 0, "block_dim": 2,
     "purpose": "Eight whole tasks over two cores: no ragged tile and no split query tile",
     "parameters": {"variant": "meta8", "B": 1, "MQ": 256, "N": 512, "D": 128}},
    {"id": "meta8_split", "seed": 1, "block_dim": 2,
     "purpose": "Three key tiles over two cores cut one query tile in half, so both owners "
                "publish partial state and the flash-decoding merge produces the row",
     "parameters": {"variant": "meta8", "B": 1, "MQ": 128, "N": 384, "D": 128}},
    {"id": "meta8_batch_tail", "seed": 2, "block_dim": 2,
     "purpose": "Two batches with a one-row query tail and a five-key tail: the tail softmax "
                "path and the D-214 contracted tail PV, which reads only valid V rows",
     "parameters": {"variant": "meta8", "B": 2, "MQ": 129, "N": 133, "D": 128}},
    {"id": "meta8_split_tail", "seed": 3, "block_dim": 4,
     "purpose": "Two split boundaries and ragged tiles at once: a merged query tile whose "
                "owners each saw a partial key tile",
     "parameters": {"variant": "meta8", "B": 1, "MQ": 129, "N": 257, "D": 128}},
    {"id": "meta8_idle", "seed": 4, "block_dim": 2,
     "purpose": "One task on two cores: the core with no work must write nothing at all",
     "parameters": {"variant": "meta8", "B": 1, "MQ": 65, "N": 73, "D": 128}},
    {"id": "meta8_reuse", "seed": 5, "block_dim": 3,
     "purpose": "Twenty tasks over three cores: the three softmax-state cache slots wrap and "
                "the two-step preload keeps running across query tiles",
     "parameters": {"variant": "meta8", "B": 1, "MQ": 513, "N": 385, "D": 128}},

    {"id": "meta11_aligned", "seed": 0, "block_dim": 2,
     "purpose": "The eleven-field ABI on the aligned geometry: slot, parity and key flags are "
                "GM loads instead of in-core arithmetic",
     "parameters": {"variant": "meta11", "B": 1, "MQ": 256, "N": 512, "D": 128}},
    {"id": "meta11_split", "seed": 1, "block_dim": 2,
     "purpose": "The split query tile again, with last_k driving the merge publication instead "
                "of a kt comparison",
     "parameters": {"variant": "meta11", "B": 1, "MQ": 128, "N": 384, "D": 128}},
    {"id": "meta11_batch_tail", "seed": 2, "block_dim": 2,
     "purpose": "Ragged query and key tails on the eleven-field ABI, over two batches",
     "parameters": {"variant": "meta11", "B": 2, "MQ": 129, "N": 133, "D": 128}},
    {"id": "meta11_split_tail", "seed": 3, "block_dim": 4,
     "purpose": "Two split boundaries and ragged tiles, with first_k selecting the accumulator "
                "reset that meta8 derives from kt",
     "parameters": {"variant": "meta11", "B": 1, "MQ": 129, "N": 257, "D": 128}},
    {"id": "meta11_idle", "seed": 4, "block_dim": 2,
     "purpose": "One task on two cores on the eleven-field ABI: an idle core reads no metadata",
     "parameters": {"variant": "meta11", "B": 1, "MQ": 65, "N": 73, "D": 128}},
    {"id": "meta11_reuse", "seed": 5, "block_dim": 3,
     "purpose": "Twenty tasks over three cores where the cache slot comes from the table, so a "
                "wrong slot field would alias two query tiles' softmax state",
     "parameters": {"variant": "meta11", "B": 1, "MQ": 513, "N": 385, "D": 128}},
]


def execute(case, inputs, launcher, backend):
    """Pick the kernel this case's `variant` names and launch it. The output is handed in
    poisoned with NaN and seeded into the launch, so a query row no core claims reads back as
    NaN rather than as the zero a missing store would be indistinguishable from."""
    entry = KERNELS[case["parameters"]["variant"]]
    out = torch.full(inputs["q"].shape, float("nan"), dtype=torch.bfloat16)
    op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(inputs["q"], inputs["k"], inputs["v"], inputs["meta"], out,
                      *(inputs[name] for name in ("B", "MQ", "N", "D")))}


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
            print(f"{case['id']:20s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
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
