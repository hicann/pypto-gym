# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the five-stage GDN forward through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator
    python main.py --stages                 # also check each kernel on its own
    python main.py --launcher aclnn         # cce backend on this machine's card
    python main.py --launcher pypto --case recurrence

This is a pipeline, not a single kernel: five kernels launch in order and each reads what
the previous ones wrote. Stage order is preprocess -> strict triangular inverse -> recompute
WU -> attention scores -> chunk recurrence. The seams are fp32 `g_cumsum`, `decay_mask` and
`strict_lower`, and bf16 `wu_attn`, `value_wu`, `k_cumdecay` and `attention`.

The recurrent tail exists in two kernels, which is why `--stages` launches six. `plain`
returns only the output and the final state; `saved` returns those plus the four histories
gdn_bwd consumes, from the same recurrence. Checking both is how a difference gets attributed
to the saved kernel's extra stores rather than to the maths they share.

The two modes answer different questions. The default hands every kernel its predecessor's
actual output, which is the only arrangement that can catch a stage that is exact on its own
and wrong once the stage before it has run on the same core. `--stages` instead hands each
kernel the reference's own upstream values and compares all fifteen checkpoints, so a wrong
`output` can be attributed to one launch rather than to the pipeline as a whole.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import (gdn_preprocess_v2_kernel, gdn_recompute_wu_v2_kernel, gdn_recurrent_plain,
                    gdn_recurrent_saved, sub1_kernel, tril_inverse64_v2_strict_bf16_kernel)
from reference import OUTPUTS, make_inputs, reference, reference_stages

# The source's elementwise budget for fp32 reductions across the explicit bf16 boundaries,
# plus a 2% relative residual norm. The residual is what rejects an all-zero or 25%-perturbed
# tensor: at an input scale of 0.05 most of these outputs sit well inside 2e-3 of zero on
# their own, so the elementwise bound alone would accept a kernel that wrote nothing.
TOLERANCE = {"rtol": 0.002, "atol": 0.002, "max_relative_l2": 0.02}

CASES = [
    {"id": "single_chunk", "seed": 8101, "block_dim": 1,
     "purpose": "One chunk, log-sigmoid gates: all five seams with no chunk-to-chunk recurrence",
     "parameters": {"B": 1, "H": 1, "C": 1, "gate": "logsigmoid", "scale": 0.05}},
    {"id": "recurrence", "seed": 8102, "block_dim": 1,
     "purpose": "Two chunks, slow gates: the bf16 state has to carry from one chunk to the next",
     "parameters": {"B": 1, "H": 1, "C": 2, "gate": "slow", "scale": 0.05}},
    {"id": "partition_tail", "seed": 8103, "block_dim": 2,
     "purpose": "Six BHC tiles over two cores: the producer partition ends on a partial batch tile",
     "parameters": {"B": 1, "H": 2, "C": 3, "gate": "slow", "scale": 0.05}},
    {"id": "bh_reuse", "seed": 8104, "block_dim": 1,
     "purpose": "Three BH groups on one core: on-chip buffers are reused between heads",
     "parameters": {"B": 1, "H": 3, "C": 2, "gate": "slow", "scale": 0.05}},
]

STAGE_OUTPUTS = ("preprocess.g_cumsum", "preprocess.decay_mask", "preprocess.strict_lower",
                 "inverse.wu_attn", "recompute.value_wu", "recompute.k_cumdecay",
                 "scores.attention", "plain.output", "plain.final_state",
                 *("saved." + name for name in OUTPUTS))


def poison(shape, dtype=torch.bfloat16):
    """Every destination is handed in full of NaN and seeded into the launch, so an element no
    kernel writes reads back as NaN rather than as a zero the reference may also produce."""
    return torch.full(tuple(shape), float("nan"), dtype=dtype)


def stage_launcher(case, launcher, backend):
    """One OpExec per stage: these are five separate launches, not one kernel in five parts."""

    def launch(entry, args):
        op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{entry.name}",
                    seed_outputs=True)
        return op(*args)

    return launch


def shapes(values):
    b, h, c, _, _ = values["query"].shape
    return (b, h, c), (b, h, c, 64), (b, h, c, 64, 64), (b, h, 128, 128)


def preprocess(values, launch):
    """g_cumsum, decay_mask and strict_lower. The prefix sum is the source's fp32 triangular
    matmul `g @ triu`, not a cumsum primitive, and beta stays out of the cube operand."""
    (b, h, c), scalar, matrix, _ = shapes(values)
    return launch(gdn_preprocess_v2_kernel,
                  (values["key"], values["beta"], values["g"], torch.triu(torch.ones(64, 64)),
                   poison(scalar, torch.float32), poison(matrix, torch.float32),
                   poison(matrix, torch.float32), b, h, c, 64, 128))


def inverse(values, launch):
    (b, h, c), _, matrix, _ = shapes(values)
    return launch(tril_inverse64_v2_strict_bf16_kernel,
                  (values["strict_lower"], poison(matrix), b, h, c))


def recompute(values, launch):
    (b, h, c), _, _, _ = shapes(values)
    shape = values["query"].shape
    return launch(gdn_recompute_wu_v2_kernel,
                  (values["key"], values["value"], values["beta"], values["g_cumsum"],
                   values["wu_attn"], poison(shape), poison(shape), b, h, c, 64, 128))


def scores(values, launch):
    (b, h, c), _, matrix, _ = shapes(values)
    return launch(sub1_kernel,
                  (values["query"], values["key"], values["decay_mask"], poison(matrix), b, h, c))


def recurrent(values, launch, saved):
    """The chunk recurrence, in its output-only or its history-publishing form.

    The initial state is zeros rather than NaN: this unit implements the zero-initial-state
    path only, and the kernel reads that tensor before it writes anything into it.
    """
    (b, h, c), scalar, _, state = shapes(values)
    shape = values["query"].shape
    out = [poison(shape), poison(state)]
    if saved:
        out += [poison((b, h, c, 128, 128)), poison(shape), poison(shape),
                poison(scalar, torch.float32)]
    return launch(gdn_recurrent_saved if saved else gdn_recurrent_plain,
                  (values["attention"], values["query"], values["key"], values["value_wu"],
                   values["k_cumdecay"], values["g_cumsum"], torch.zeros(state, dtype=torch.bfloat16),
                   *out, b, h, c))


def execute(case, inputs, launcher, backend):
    """The composition: each stage reads what the stage before it actually wrote."""
    launch = stage_launcher(case, launcher, backend)
    g_cumsum, decay_mask, strict_lower = preprocess(inputs, launch)
    values = dict(inputs, g_cumsum=g_cumsum, decay_mask=decay_mask, strict_lower=strict_lower)
    values["wu_attn"] = inverse(values, launch)
    values["value_wu"], values["k_cumdecay"] = recompute(values, launch)
    values["attention"] = scores(values, launch)
    return dict(zip(OUTPUTS, recurrent(values, launch, saved=True), strict=True))


def execute_stages(case, inputs, launcher, backend):
    """The same kernels, each fed the reference's upstream values instead of its predecessor's.
    A failure here names one kernel; `execute` above is what checks that they still agree when
    they are chained."""
    launch = stage_launcher(case, launcher, backend)
    expected = reference_stages(inputs)
    values = dict(inputs, g_cumsum=expected["preprocess.g_cumsum"],
                  decay_mask=expected["preprocess.decay_mask"],
                  strict_lower=expected["preprocess.strict_lower"],
                  wu_attn=expected["inverse.wu_attn"], value_wu=expected["recompute.value_wu"],
                  k_cumdecay=expected["recompute.k_cumdecay"],
                  attention=expected["scores.attention"])
    pre = preprocess(values, launch)
    wu = recompute(values, launch)
    plain = recurrent(values, launch, saved=False)
    saved = recurrent(values, launch, saved=True)
    result = {"preprocess.g_cumsum": pre[0], "preprocess.decay_mask": pre[1],
              "preprocess.strict_lower": pre[2], "inverse.wu_attn": inverse(values, launch),
              "recompute.value_wu": wu[0], "recompute.k_cumdecay": wu[1],
              "scores.attention": scores(values, launch),
              "plain.output": plain[0], "plain.final_state": plain[1]}
    result.update({"saved." + name: value for name, value in zip(OUTPUTS, saved, strict=True)})
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
        idx, unwritten = outside.nonzero(), int((outside & torch.isnan(got.float())).sum())
        note = f", {unwritten} still NaN-poisoned (never written)" if unwritten else ""
        print(f"      {len(idx)}/{outside.numel()} elements outside{note}; "
              f"first {' '.join(str(tuple(i.tolist())) for i in idx[:3])}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce", "pto_isa"))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--stages", action="store_true",
                        help="also check each kernel alone against its own reference checkpoint")
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
        if args.stages:
            expected = reference_stages(inputs)
            actual = execute_stages(case, inputs, args.launcher, args.backend)
            for name in STAGE_OUTPUTS:
                if not compare(name, actual[name], expected[name], TOLERANCE):
                    failed.append(f"{case['id']}/{name}")
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
