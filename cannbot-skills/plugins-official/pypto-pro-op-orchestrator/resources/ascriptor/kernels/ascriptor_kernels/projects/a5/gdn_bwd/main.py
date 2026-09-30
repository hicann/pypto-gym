# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the five-stage GDN (gated delta-net) backward through OpExec and check it against
the reference.

    python main.py                          # every case, functional simulator
    python main.py --stages                 # also check each stage's own outputs
    python main.py --launcher aclnn         # cce backend on this machine's card
    python main.py --launcher pypto --case recurrence

This is a pipeline, not a single kernel: five kernels launch in order and each reads what the
previous ones wrote. Stage order is scan_local -> scan_state -> wu -> inverse_preprocess ->
finalize; `--stages` publishes every intermediate and compares it with the matching reference
stage, which is how you find which launch is wrong instead of only that the final gradient is.

The gate is what separates this from delta_rule_bwd: every chunk carries a per-token log gate
`g`, so the intra-chunk keep is `exp(g_cumsum_i - g_cumsum_j)` under the causal mask instead of
a constant 1, the state recurrence decays by `exp(g_cumsum)` between chunks, and there is a
fifth gradient `d_g`. Set g to zero and this unit is delta_rule_bwd.

A stage that is exact on its own can still be wrong after a predecessor has run on the same
core, because on-chip storage is not cleared between launches. That is why the default here
runs the whole sequence rather than each kernel alone, and why a green run of one stage in
isolation proves less than it looks.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import (finalize_bwd_kernel, inverse_preprocess_bwd_kernel, scan_local_bwd_kernel,
                    scan_state_bwd_kernel, wu_bwd_kernel)
from reference import make_inputs, reference, reference_stages

# The source's 5e-4 elementwise budget for FP32 reductions across the explicit BF16 handoffs,
# plus a 2% relative residual norm. The residual is what rejects an all-zero or missing small
# gradient: several leaf gradients here are small enough that atol alone accepts a kernel that
# writes nothing, and a 25% perturbation of a nonzero one also has to fail.
TOLERANCE = {"rtol": 0.0005, "atol": 0.0005, "max_relative_l2": 0.02}

CASES = [
    {"id": "single_chunk", "seed": 9101, "block_dim": 1,
     "purpose": "One chunk with the original log-sigmoid gate: no recurrence and no core split",
     "parameters": {"B": 1, "H": 1, "C": 1, "gate": "logsigmoid", "scale": 0.05}},
    {"id": "recurrence", "seed": 9102, "block_dim": 1,
     "purpose": "Two chunks under slow decay: the reverse-time state actually carries",
     "parameters": {"B": 1, "H": 1, "C": 2, "gate": "slow", "scale": 0.05}},
    {"id": "partition_tail", "seed": 9103, "block_dim": 2,
     "purpose": "BHC=6 on two cores: not a multiple of four, so finalize takes its partial batch-tile path",
     "parameters": {"B": 1, "H": 2, "C": 3, "gate": "slow", "scale": 0.05}},
    {"id": "zero_gradient", "seed": 9104, "block_dim": 1,
     "purpose": "Zero grad_output and zero grad_final_state: every input gradient must come back zero",
     "parameters": {"B": 1, "H": 1, "C": 2, "gate": "slow", "scale": 0.05, "gradient": "zero"}},
    {"id": "bh_reuse", "seed": 9105, "block_dim": 1,
     "purpose": "Three heads on one core: on-chip buffers are reused between BH groups",
     "parameters": {"B": 1, "H": 3, "C": 2, "gate": "slow", "scale": 0.05}},
]

OUTPUTS = ("d_query", "d_key", "d_value", "d_beta", "d_g")
STAGE_OUTPUTS = ("d_score_tmp", "d_v_attn_tmp", "d_decay_core", "d_decay_core_masked",
                 "d_q_core", "d_k_core", "d_g_core", "d_value_wu", "d_k_cumdecay",
                 "d_wu", "d_v_beta", "d_k_beta_wu", "d_g_wu",
                 "d_k_pre", "d_k_beta_pre", "d_decay_pre", "d_decay_pre_masked")


def execute(case, inputs, launcher, backend):
    """Launch the five stages in order. Every destination is handed in poisoned with NaN and
    seeded into the launch, so an element no stage writes reads back as NaN rather than as a
    zero the reference may also produce."""
    q, k, v, beta = (inputs[name] for name in ("query", "key", "value", "beta"))
    B, H, C = q.shape[:3]
    gm = (B, H, C, 64, 64)
    vec = (B, H, C, 64)
    bhc = B * H * C

    def empty(shape=None, dtype=torch.float32):
        return torch.full(q.shape if shape is None else tuple(shape), float("nan"), dtype=dtype)

    def launch(entry, args):
        op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{entry.name}",
                    seed_outputs=True)
        return op(*args)

    local = launch(scan_local_bwd_kernel, (
        q, k, inputs["grad_output"], inputs["decay_mask"], inputs["v_new_history"],
        empty(gm, torch.bfloat16), empty(), empty(gm), empty(gm), B, H, C))
    state = launch(scan_state_bwd_kernel, (
        q, k, inputs["grad_output"], inputs["g_cumsum"], inputs["k_cumdecay"],
        inputs["state_after_history"], inputs["grad_final_state"], local[0],
        inputs["v_new_history"], inputs["k_weighted_history"], local[1], inputs["exp_delta_history"],
        empty(), empty(), empty(vec), empty(dtype=torch.bfloat16), empty(dtype=torch.bfloat16),
        B, H, C))
    wu = launch(wu_bwd_kernel, (
        k, v, beta, inputs["g_cumsum"], inputs["wu_attn_bf16"], state[3], state[4],
        empty(gm, torch.bfloat16), empty(), empty(), empty(vec), B, H, C))
    inverse = launch(inverse_preprocess_bwd_kernel, (
        k, beta, inputs["decay_mask"], inputs["wu_attn_bf16"], wu[0],
        empty(), empty(), empty(gm), empty(gm), B, H, C))

    # finalize batches four BHC rows of the gate gradient at a time, so its d_g inputs and its
    # d_g destination are flattened to [BHC, ...]. The four constant operands are the row
    # selectors of that batching, the identity and the reverse-causal keep.
    selectors = torch.zeros(16, 64)
    for index in range(4):
        selectors[index * 4 + index] = 1
    final = launch(finalize_bwd_kernel, (
        k, v, beta, state[1], state[2].reshape(bhc, 64), local[3].reshape(bhc, 64, 64),
        wu[1], wu[2], wu[3].reshape(bhc, 64), inverse[0], inverse[1],
        inverse[3].reshape(bhc, 64, 64),
        selectors, -selectors, torch.eye(64), torch.ones(64, 64).tril(),
        empty(), empty(), empty(vec), empty((bhc, 64)), B, H, C, bhc))
    final = (*final[:3], final[3].reshape(vec))

    result = dict(zip(OUTPUTS, (state[0], *final), strict=True))
    result.update(zip(STAGE_OUTPUTS, (*local, *state, *wu, *inverse), strict=True))
    return result


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
        if "max_relative_l2" in tolerance:
            # float64: this pipeline's row-scaled tensors reach 1e25, whose squares overflow a
            # float32 accumulator and turn the ratio into a nan that reads as a failure.
            norm = torch.linalg.vector_norm(want.double().flatten())
            residual = torch.linalg.vector_norm((got.double() - want.double()).flatten())
            relative = (residual / norm).item() if norm > 0 else residual.item()
            ok = ok and relative <= tolerance["max_relative_l2"]
            detail += f"  rel_l2={relative:.3e}/{tolerance['max_relative_l2']}"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:20s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
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
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--stages", action="store_true",
                        help="also compare each stage's intermediates, not only the five gradients")
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
        print(f"{case['id']}  (B={p['B']} H={p['H']} C={p['C']} gate={p['gate']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference_stages(inputs) if args.stages else reference(inputs)
        if args.stages:
            # finalize forwards the query gradient untouched, so the public `d_query` is
            # scan_state's `d_q_core`; the stage list names it once and the gradient list once.
            expected["d_query"] = expected["d_q_core"]
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in (STAGE_OUTPUTS + OUTPUTS if args.stages else OUTPUTS):
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
