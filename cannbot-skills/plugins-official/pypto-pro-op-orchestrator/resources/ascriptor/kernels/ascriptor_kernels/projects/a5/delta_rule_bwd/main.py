# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the five-stage Delta Rule backward through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator
    python main.py --stages                 # also check each stage's own outputs
    python main.py --launcher aclnn         # cce backend on this machine's card
    python main.py --launcher pypto --case recurrence

This is a pipeline, not a single kernel: five kernels launch in order and each reads what
the previous ones wrote. Stage order is scan_local -> scan_state -> wu -> inverse_preprocess
-> finalize; `--stages` publishes every intermediate and compares it with the matching
reference stage, which is how you find which launch is wrong instead of only that the
final gradient is.

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

# The source's elementwise budget, plus a 5% relative residual norm. The residual is what
# rejects an all-zero or missing small gradient: atol alone accepts a kernel that writes
# nothing wherever the true gradient is already small.
TOLERANCE = {"rtol": 0.0005, "atol": 0.0005, "max_relative_l2": 0.05}

CASES = [
    {"id": "smoke", "seed": 4201, "block_dim": 1,
     "purpose": "One chunk: the whole pipeline with no recurrence and no core split",
     "parameters": {"B": 1, "H": 1, "C": 1, "input_scale": 0.05}},
    {"id": "recurrence", "seed": 4202, "block_dim": 1,
     "purpose": "Three chunks in sequence: the reverse-time state actually carries",
     "parameters": {"B": 1, "H": 1, "C": 3, "input_scale": 0.05}},
    {"id": "reuse", "seed": 4203, "block_dim": 1,
     "purpose": "Three heads on one core: on-chip buffers are reused between heads",
     "parameters": {"B": 1, "H": 3, "C": 2, "input_scale": 0.05}},
    {"id": "multi_core", "seed": 4204, "block_dim": 2,
     "purpose": "Two cores: the batch split must not let one core read the other's state",
     "parameters": {"B": 2, "H": 1, "C": 2, "input_scale": 0.05}},
    {"id": "zero_final_gradient", "seed": 4205, "block_dim": 1,
     "purpose": "d(final state) = 0: the boundary term drops out and only the chunk path remains",
     "parameters": {"B": 1, "H": 1, "C": 2, "final_gradient_scale": 0.0}},
]

OUTPUTS = ("d_query", "d_key", "d_value", "d_beta")
STAGE_OUTPUTS = ("d_score_tmp", "d_v_attn_tmp", "d_q_core", "d_k_core", "d_value_wu",
                 "d_k_cumdecay", "d_wu", "d_v_beta", "d_k_beta_wu", "d_k_pre", "d_k_beta_pre")


def execute(case, inputs, launcher, backend):
    """Launch the five stages in order. Every destination is handed in poisoned with NaN and
    seeded into the launch, so an element no stage writes reads back as NaN rather than as a
    zero the reference may also produce."""
    q, k, v, beta, do = (inputs[name] for name in ("query", "key", "value", "beta", "grad_output"))
    B, H, C = q.shape[:3]
    ashape = (B, H, C, 64, 64)

    def empty(shape=None, dtype=torch.float32):
        return torch.full(q.shape if shape is None else shape, float("nan"), dtype=dtype)

    def launch(entry, args):
        op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{entry.name}",
                    seed_outputs=True)
        return op(*args)

    local = launch(scan_local_bwd_kernel, (
        q, k, do, inputs["v_new_history"], empty(ashape, torch.bfloat16), empty(), B, H, C))
    scan = launch(scan_state_bwd_kernel, (
        q, k, do, inputs["k_cumdecay"], inputs["state_after_history"], inputs["grad_final_state"],
        local[0], inputs["v_new_history"], local[1], empty(), empty(),
        empty(dtype=torch.bfloat16), empty(dtype=torch.bfloat16), B, H, C))
    wu = launch(wu_bwd_kernel, (
        k, v, beta, inputs["wu_attn_bf16"], scan[2], scan[3],
        empty(ashape, torch.bfloat16), empty(), empty(), B, H, C))
    inverse = launch(inverse_preprocess_bwd_kernel, (
        k, beta, inputs["wu_attn_bf16"], wu[0],
        empty(dtype=torch.bfloat16), empty(dtype=torch.bfloat16), B, H, C))
    final = launch(finalize_bwd_kernel, (
        k, v, beta, scan[1], wu[1], wu[2], inverse[0], inverse[1],
        empty(), empty(), empty(beta.shape), B, H, C))

    result = dict(zip(OUTPUTS, (scan[0], *final), strict=True))
    result.update(zip(STAGE_OUTPUTS, (*local, *scan, *wu, *inverse), strict=True))
    return result


def compare(name, got, want, tolerance):
    got, want = got.cpu().float(), want.cpu().float()
    if tolerance is None:
        ok, detail = torch.equal(got, want), ""
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (bounds.get("atol", 0.0) + bounds.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok, detail = torch.allclose(got, want, **bounds), f"  allclose={margin:.2f}x ({bounds})"
        if "max_relative_l2" in tolerance:
            norm = torch.linalg.vector_norm(want.flatten())
            residual = torch.linalg.vector_norm((got - want).flatten())
            relative = (residual / norm).item() if norm > 0 else residual.item()
            ok = ok and relative <= tolerance["max_relative_l2"]
            detail += f"  rel_l2={relative:.3e}/{tolerance['max_relative_l2']}"
    worst = (got - want).abs().max().item()
    print(f"    {name:14s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
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
                        help="also compare each stage's intermediates, not only the four gradients")
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
        print(f"{case['id']}  (B={p['B']} H={p['H']} C={p['C']}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference_stages(inputs) if args.stages else reference(inputs)
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
