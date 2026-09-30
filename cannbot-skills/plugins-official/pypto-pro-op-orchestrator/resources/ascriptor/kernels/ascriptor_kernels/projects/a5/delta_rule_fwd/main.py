# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the four-stage Delta Rule forward through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator
    python main.py --stages                 # also check every intermediate the launches publish
    python main.py --launcher aclnn         # cce backend on this machine's card
    python main.py --launcher pypto --case neumann

This is a pipeline, not a single kernel: four kernels launch in order and each reads what
the previous ones wrote. Stage order is preprocess -> triangular inverse -> recompute W/U ->
fused chunk recurrence. The seam out of preprocess is fp32; everything after the inverse is
bf16.

`--stages` publishes every intermediate and compares it with the matching reference stage,
which is how you find which launch is wrong instead of only that the output is. It launches
two more kernels than the production path: `recurrence_saved_kernel` in place of `sub2_kernel`,
because that is the variant that publishes the state and v_new histories, and the standalone
`sub1_kernel` score leaf, whose computation the production recurrence fuses rather than
launching. Running that leaf is what stops a change to the fused causal-mask epilogue from
being checked only against itself.

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

from kernel import (delta_preprocess_kernel, delta_recompute_wu_kernel, recurrence_saved_kernel,
                    sub1_kernel, sub2_kernel, tril_inverse64_neumann_strict_bf16_kernel,
                    tril_inverse64_neumann_strict_bf16_serial_kernel,
                    tril_inverse64_v2_strict_bf16_kernel)
from reference import OUTPUTS, STAGE_OUTPUTS, make_inputs, reference, reference_stages

# The source's elementwise budget, plus a 5% relative residual norm. The residual is what
# rejects an all-zero or missing small output: at an input scale of 0.05 most of these
# tensors sit inside 2e-3 of zero on their own, so atol alone accepts a kernel that writes
# nothing wherever the true value is already small.
TOLERANCE = {"rtol": 0.002, "atol": 0.002, "max_relative_l2": 0.05}

CASES = [
    {"id": "smoke", "seed": 4201, "block_dim": 1,
     "purpose": "One chunk: the whole pipeline with no recurrence and no core split",
     "parameters": {"B": 1, "H": 1, "C": 1, "input_scale": 0.05}},
    {"id": "recurrence", "seed": 4202, "block_dim": 1,
     "purpose": "Three chunks in sequence: the bf16 state actually carries",
     "parameters": {"B": 1, "H": 1, "C": 3, "input_scale": 0.05}},
    {"id": "reuse", "seed": 4203, "block_dim": 1,
     "purpose": "Three heads on one core: on-chip buffers are reused between heads",
     "parameters": {"B": 1, "H": 3, "C": 2, "input_scale": 0.05}},
    {"id": "multi_core", "seed": 4204, "block_dim": 2,
     "purpose": "Two cores: the batch split must not let one core read the other's state",
     "parameters": {"B": 2, "H": 1, "C": 2, "input_scale": 0.05}},
    {"id": "saved_state", "seed": 4205, "block_dim": 1,
     "purpose": "The four-output variant at the source's 1/sqrt(128) query scale: the same "
                "four launches also publish the histories the backward consumes",
     "parameters": {"B": 1, "H": 1, "C": 2, "variant": "saved_state",
                    "query_scale": 0.08838834764831845}},
    {"id": "neumann", "seed": 4206, "block_dim": 1,
     "purpose": "The Neumann inverse instead of the block inverse: bf16 cube operands and "
                "bf16 intermediate publishes, against the same fp32 reference",
     "parameters": {"B": 1, "H": 1, "C": 2, "inverse_variant": "neumann"}},
    {"id": "neumann_serial", "seed": 4207, "block_dim": 1,
     "purpose": "The single-parity serial Neumann inverse, which keeps the source's operation order",
     "parameters": {"B": 1, "H": 1, "C": 1, "inverse_variant": "neumann_serial"}},
]

# Three source variants of one stage. Changing the default needs its own precision and
# performance evidence; that is why all three stay here and are all exercised by a case.
INVERSE_KERNELS = {"block": tril_inverse64_v2_strict_bf16_kernel,
                   "neumann": tril_inverse64_neumann_strict_bf16_kernel,
                   "neumann_serial": tril_inverse64_neumann_strict_bf16_serial_kernel}


def execute(case, inputs, launcher, backend, checkpoints=False):
    """Launch the four stages in order. Every destination is handed in poisoned with NaN and
    seeded into the launch, so an element no stage writes reads back as NaN rather than as a
    zero the reference may also produce. The initial state is the exception: it is real zeros,
    because this unit implements the zero-initial-state path only and the first chunk reads it.
    """
    q = (inputs["query"] * inputs["scale"]).bfloat16()
    k, v, beta = inputs["key"], inputs["value"], inputs["beta"]
    B, H, C = q.shape[:3]
    qshape, ashape, sshape = q.shape, (B, H, C, 64, 64), (B, H, 128, 128)

    def empty(shape, dtype=torch.bfloat16):
        return torch.full(shape, float("nan"), dtype=dtype)

    def launch(entry, args):
        op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{entry.name}",
                    seed_outputs=True)
        return op(*args)

    a = launch(delta_preprocess_kernel, (k, beta, empty(ashape, torch.float32), B, H, C, 64, 128))
    w = launch(INVERSE_KERNELS[inputs["inverse_variant"]], (a, empty(ashape), B, H, C))
    value_wu, key_wu = launch(delta_recompute_wu_kernel,
                              (k, v, beta, w, empty(qshape), empty(qshape), B, H, C, 64, 128))
    state0 = torch.zeros(sshape, dtype=torch.bfloat16)
    args = (q, k, value_wu, key_wu, state0, empty(qshape), empty(sshape))
    # Stage checks observe the named saved-state variant; the default public path keeps its
    # original two outputs and four production launches.
    if checkpoints or inputs["variant"] == "saved_state":
        values = launch(recurrence_saved_kernel,
                        args + (empty((B, H, C, 128, 128)), empty(qshape), B, H, C))
        result = dict(zip(OUTPUTS["saved_state"], values, strict=True))
    else:
        result = dict(zip(OUTPUTS["default"], launch(sub2_kernel, args + (B, H, C)), strict=True))
    if checkpoints:
        result.update(preprocess_attn=a, wu_attn_bf16=w, value_wu=value_wu, k_cumdecay=key_wu)
        result["standalone_scores"] = launch(sub1_kernel, (q, k, empty(ashape), B, H, C))
    return result


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
    parser.add_argument("--stages", action="store_true",
                        help="also compare every published intermediate, not only the outputs")
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
        print(f"{case['id']}  (B={p['B']} H={p['H']} C={p['C']}, "
              f"inverse={p.get('inverse_variant', 'block')}, "
              f"variant={p.get('variant', 'default')}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference_stages(inputs) if args.stages else reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend, checkpoints=args.stages)
        names = STAGE_OUTPUTS + OUTPUTS["saved_state"] if args.stages else OUTPUTS[inputs["variant"]]
        for name in names:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
