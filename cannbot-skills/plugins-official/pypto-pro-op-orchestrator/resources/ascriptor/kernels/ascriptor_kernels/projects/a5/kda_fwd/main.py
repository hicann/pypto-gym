# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the five-stage KDA forward through OpExec and check it against the reference.

    python main.py                          # every case, functional simulator
    python main.py --stages                 # also check each kernel on its own
    python main.py --launcher aclnn         # cce backend on this machine's card
    python main.py --launcher pypto --case grouped_heads

This is a pipeline, not a single kernel: five kernels launch in order and each reads what
the previous ones wrote. Stage order is gate -> score -> triangular inverse -> WY ->
fused recurrent tail, and the seams between them are fp32 `g`, an fp32 strictly-lower
score matrix, and bf16 `Aqk/Akk/w/u/kg`.

The two modes answer different questions. The default hands every kernel its predecessor's
actual output, which is the only arrangement that can catch a stage that is exact on its own
and wrong once the stage before it has run on the same core. `--stages` instead hands each
kernel the reference's own upstream values and compares all eleven checkpoints, so a wrong
`o` can be attributed to one launch rather than to the pipeline as a whole. Run both: the
first says whether the composition works, the second says which kernel broke it.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import (kda_sub1_gate_kernel, kda_sub2_score_kernel, kda_sub3_wy_kernel,
                    kda_sub45_fused_kernel, tril_inverse64_v2_strict_bf16_kernel)
from reference import make_inputs, reference, reference_stages

# The source's 0.02 elementwise budget is what the bf16 cube operands and bf16 stores cost at
# this input scale, plus a 5% relative residual norm. The residual is what rejects an all-zero
# output: with stddev 0.04 inputs the true `o` peaks near 1e-3 and `final_state` near 2e-2, so
# a kernel that writes nothing would sit inside 0.02 on its own.
# The gate is held far tighter because nothing in it is a cube operand: it is an fp32 running
# sum and an exp on the vector unit against the same maths in Torch, and only the accumulation
# order of the sequential scan separates the two.
TOLERANCE = {"default": {"rtol": 0.02, "atol": 0.02, "max_relative_l2": 0.05},
             "gate.g": {"rtol": 1e-6, "atol": 1e-6, "max_relative_l2": 1e-6}}

CASES = [
    {"id": "single_chunk", "seed": 2026, "block_dim": 1,
     "purpose": "One chunk on one core: all five seams, with no chunk-to-chunk recurrence",
     "parameters": {"B": 1, "H": 1, "HV": 1, "C": 1, "L": 64, "K": 128, "V": 128,
                    "initial_state": "random"}},
    {"id": "multi_chunk", "seed": 2027, "block_dim": 1,
     "purpose": "Two chunks: the fp32 recurrent state has to carry from one chunk to the next",
     "parameters": {"B": 1, "H": 1, "HV": 1, "C": 2, "L": 64, "K": 128, "V": 128,
                    "initial_state": "random"}},
    {"id": "grouped_heads", "seed": 2028, "block_dim": 2,
     "purpose": "HV=2 over H=1: two value heads read one shared query/key head, split over two cores",
     "parameters": {"B": 1, "H": 1, "HV": 2, "C": 2, "L": 64, "K": 128, "V": 128,
                    "initial_state": "random"}},
    {"id": "uneven_cores", "seed": 2029, "block_dim": 3,
     "purpose": "Three cores for two (B,HV) pairs, from a zero initial state: one core draws no work",
     "parameters": {"B": 1, "H": 1, "HV": 2, "C": 1, "L": 64, "K": 128, "V": 128,
                    "initial_state": "zero"}},
]

OUTPUTS = ("o", "final_state")
STAGE_OUTPUTS = ("gate.g", "score.Aqk", "score.strict", "inverse.Akk", "inverse.strict_lower",
                 "wy.w", "wy.u", "wy.qg", "wy.kg", "recurrent.o", "recurrent.final_state")


def poison(shape, dtype):
    """Every destination is handed in full of NaN and seeded into the launch, so an element no
    kernel writes reads back as NaN rather than as a zero the reference may also produce."""
    return torch.full(shape, float("nan"), dtype=dtype)


def chunked(inputs):
    """Public token-major tensors to the BHCLK / BHVCLK layout the kernels index.

    The permutation is host work on purpose: the kernels take chunks as a real axis, and
    folding the reshape into them would hide the one thing the tiling depends on.
    """
    q, k, v, raw, beta = (inputs[name] for name in ("q", "k", "v", "g_raw", "beta"))
    b, t, h, _ = q.shape
    hv, c = v.shape[2], t // 64

    def transform(x, heads):
        return x.reshape(b, c, 64, heads, 128).permute(0, 3, 1, 2, 4).contiguous()

    return {"q": transform(q, h), "k": transform(k, h), "v": transform(v, hv),
            "g_raw": transform(raw, hv),
            "beta": beta.reshape(b, c, 64, hv).permute(0, 3, 1, 2).contiguous(),
            "initial_state": inputs["initial_state"], "dims": (b, h, hv, c)}


def stage_launcher(case, launcher, backend):
    """One OpExec per stage: these are five separate launches, not one kernel in five parts."""

    def launch(entry, args):
        op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                    block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{entry.name}",
                    seed_outputs=True)
        return op(*args)

    return launch


def gate(data, launch):
    b, _, hv, c = data["dims"]
    return launch(kda_sub1_gate_kernel,
                  (data["g_raw"], poison((b, hv, c, 64, 128), torch.float32), b, hv, c, 64, 128))


def scores(data, g, launch):
    b, h, hv, c = data["dims"]
    shape = (b, hv, c, 64, 64)
    return launch(kda_sub2_score_kernel,
                  (data["q"], data["k"], g, data["beta"], poison(shape, torch.bfloat16),
                   poison(shape, torch.float32), b, h, hv, c, 64, 128, 128 ** -0.5))


def inverse(data, strict, launch):
    b, _, hv, c = data["dims"]
    return launch(tril_inverse64_v2_strict_bf16_kernel,
                  (strict, poison(strict.shape, torch.bfloat16), b, hv, c))


def wy(data, g, akk, launch):
    b, h, hv, c = data["dims"]
    outputs = tuple(poison((b, hv, c, 64, 128), torch.bfloat16) for _ in range(4))
    return launch(kda_sub3_wy_kernel,
                  (data["q"], data["k"], data["v"], data["beta"], akk, g, *outputs,
                   b, h, hv, c, 64, 128, 128))


def recurrent(data, g, aqk, w, u, kg, launch):
    b, h, hv, c = data["dims"]
    return launch(kda_sub45_fused_kernel,
                  (data["q"], aqk, kg, w, u, g, data["initial_state"],
                   poison((b, hv, c, 64, 128), torch.bfloat16),
                   poison((b, hv, 128, 128), torch.float32),
                   b, h, hv, c, 64, 128, 128, 128 ** -0.5))


def execute(case, inputs, launcher, backend):
    """The composition: each stage reads what the stage before it actually wrote."""
    launch = stage_launcher(case, launcher, backend)
    data = chunked(inputs)
    g = gate(data, launch)
    aqk, strict = scores(data, g, launch)
    akk = inverse(data, strict, launch)
    w, u, _qg, kg = wy(data, g, akk, launch)
    out, final_state = recurrent(data, g, aqk, w, u, kg, launch)
    b, _, hv, c = data["dims"]
    return {"o": out.permute(0, 2, 3, 1, 4).reshape(b, c * 64, hv, 128).contiguous(),
            "final_state": final_state}


def execute_stages(case, inputs, launcher, backend):
    """The same five launches, each fed the reference's upstream values instead of its
    predecessor's. A failure here names one kernel; `execute` above is what checks that the
    five of them still agree when they are chained."""
    launch = stage_launcher(case, launcher, backend)
    data = chunked(inputs)
    expected = reference_stages(inputs)
    result = {"gate.g": gate(data, launch)}
    aqk, strict = scores(data, expected["gate.g"], launch)
    result.update({"score.Aqk": aqk, "score.strict": strict})
    result["inverse.Akk"] = inverse(data, expected["score.strict"], launch)
    # The identity diagonal is fixed and large; comparing only the strictly-lower part stops an
    # identity matrix from hiding a missing inversion behind it.
    result["inverse.strict_lower"] = torch.tril(result["inverse.Akk"], diagonal=-1)
    result.update(zip(("wy.w", "wy.u", "wy.qg", "wy.kg"),
                      wy(data, expected["gate.g"], expected["inverse.Akk"], launch), strict=True))
    out, state = recurrent(data, expected["gate.g"], expected["score.Aqk"], expected["wy.w"],
                           expected["wy.u"], expected["wy.kg"], launch)
    result.update({"recurrent.o": out, "recurrent.final_state": state})
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
        print(f"{case['id']}  (B={p['B']} H={p['H']} HV={p['HV']} C={p['C']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        if args.stages:
            expected = reference_stages(inputs)
            actual = execute_stages(case, inputs, args.launcher, args.backend)
            for name in STAGE_OUTPUTS:
                tolerance = TOLERANCE.get(name, TOLERANCE["default"])
                if not compare(name, actual[name], expected[name], tolerance):
                    failed.append(f"{case['id']}/{name}")
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name], TOLERANCE.get(name, TOLERANCE["default"])):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
