# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run NsaCompress through OpExec and check it against the Torch reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher board         # cce backend, on this machine's card
    python main.py --launcher pypto --case ut_a2_bf16

Every output is compared BITWISE. A port reproduces an operator, so the comparison that
belongs here is identity, and this operator has two ways to be almost right. Upstream
rounds ties away from zero (AscendC CAST_ROUND) where `torch.Tensor.to` rounds them to
even; and fp32 addition is not associative, so the ReduceBlock fold order is observable
too. Each costs about one ULP on a handful of elements -- which is exactly what a
tolerance wide enough to "pass" would have absorbed. The second one was found only by
running the vendor operator on an A2 card and disagreeing with it.

What this delivers is the PyPTO-Pro path: `--launcher pypto`. `--launcher board --backend cce`
is a control -- it adjudicates a disagreement and catches a form only one backend prints -- and
a CCE pass on its own is not this demo passing.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import column_tile, make_kernel
from reference import make_inputs, reference

# Bitwise, for the reason in the module docstring. `None` selects exact comparison.
TOLERANCE = {"output": None}

CASES = [
    {"id": "ut_a1_fp16", "seed": 30101, "block_dim": 28,
     "purpose": "Upstream UT Case_0: 48 batches of 100, one head, L == d so no window overlaps",
     "parameters": {"dtype": "fp16", "N": 1, "D": 32, "L": 16, "stride": 16,
                    "seq_lens": [100] * 48}},
    {"id": "ut_a2_bf16", "seed": 30102, "block_dim": 28,
     "purpose": "Upstream UT Case_1 in bf16 with nine heads: N*D = 288 is not a multiple of "
                "the 64 fp32 lanes, so every row's last chunk overruns into the spare row",
     "parameters": {"dtype": "bf16", "N": 9, "D": 32, "L": 16, "stride": 16,
                    "seq_lens": [100] * 48}},
    {"id": "ut_a3_fp16_small", "seed": 30103, "block_dim": 4,
     "purpose": "Upstream aclnn UT Case_0: one whole window per batch, the shortest sequence "
                "that produces any output at all",
     "parameters": {"dtype": "fp16", "N": 32, "D": 16, "L": 16, "stride": 16,
                    "seq_lens": [16, 16, 16]}},
    {"id": "aclnn_example_overlap", "seed": 30104, "block_dim": 4,
     "purpose": "The upstream example's L = 32 > d = 16: overlapping windows, which is the "
                "regime its overlap state machine exists for and this kernel needs no state for",
     "parameters": {"dtype": "fp16", "N": 1, "D": 32, "L": 32, "stride": 16,
                    "seq_lens": [128]}},
    {"id": "ragged_batches_bf16", "seed": 30105, "block_dim": 8,
     "purpose": "Unequal sequence lengths including one shorter than L, which contributes no "
                "output row at all and must not shift the rows after it",
     "parameters": {"dtype": "bf16", "N": 4, "D": 64, "L": 32, "stride": 16,
                    "seq_lens": [100, 33, 64, 17, 200]}},
    {"id": "widest_head_sub_split", "seed": 30107, "block_dim": 8,
     "purpose": "The largest corner the constraints admit, L = 128 with D = 256: one whole "
                "head does not fit the UB budget, so the group is half a head and the "
                "window arrives in two transfers per output token",
     "parameters": {"dtype": "bf16", "N": 1, "D": 256, "L": 128, "stride": 16,
                    "seq_lens": [256, 200]}},
    {"id": "wide_head_multi_chunk", "seed": 30108, "block_dim": 8,
     "purpose": "A head wider than one register (D = 128) taken as a whole head: the weight "
                "splat needs several registers per head, where a head of 64 or fewer columns "
                "needs one that overruns into the next",
     "parameters": {"dtype": "bf16", "N": 1, "D": 128, "L": 16, "stride": 16,
                    "seq_lens": [256, 200]}},
    {"id": "many_head_groups", "seed": 30109, "block_dim": 8,
     "purpose": "Four column groups per token (N = 8 heads, two per group): the only case "
                "where one group's software pipeline is followed by another's, which is "
                "where a prefetch past the end of a group has a second writer to race",
     "parameters": {"dtype": "bf16", "N": 8, "D": 64, "L": 128, "stride": 16,
                    "seq_lens": [256, 160]}},
    {"id": "more_cores_than_tokens", "seed": 30106, "block_dim": 16,
     "purpose": "Fewer output tokens than vector cores: the tail cores must own an empty range "
                "rather than another core's rows",
     "parameters": {"dtype": "fp16", "N": 2, "D": 32, "L": 16, "stride": 16,
                    "seq_lens": [48, 48]}},
]


def execute(case, inputs, launcher, backend):
    parameters = case["parameters"]
    N, D, L = parameters["N"], parameters["D"], parameters["L"]
    token_count = inputs["win_start"].numel()

    entry = make_kernel(parameters["dtype"], T=inputs["T"], TC=token_count,
                        L=L, N=N, D=D, block_dim=case["block_dim"])
    op = OpExec(entry, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)

    # The ABI is the upstream's TND bytes seen two-dimensionally: [T, N, D] and [L, N] are
    # contiguous, so these are views and move nothing. The reference keeps the 3-D shape.
    destination = torch.full((token_count, N * D), float("nan"), dtype=inputs["input"].dtype)
    got = op(inputs["input"].reshape(inputs["T"], N * D),
             inputs["weight"].reshape(1, L * N),
             inputs["win_start"].reshape(1, token_count),
             destination)
    return {"output": got.reshape(token_count, N, D)}


def compare(name, got, want, tolerance):
    got, want = got.cpu(), want.cpu()
    if tolerance is None:
        ok, detail, outside = torch.equal(got, want), "  bitwise", got != want
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        atol, rtol = bounds.get("atol", 0.0), bounds.get("rtol", 0.0)
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element. Print the worst element as a fraction of its own allowance, so the
        # number has a bound of 1 and a passing line cannot read as a failing one.
        room = (atol + rtol * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        outside = ~torch.isclose(got.float(), want.float(), **bounds)
        ok = not outside.any().item()
        detail = f"  allclose={margin:.2f}x (atol={atol:g} rtol={rtol:g})"
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:12s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok and outside.any():
        # The destination arrives NaN-poisoned, so an element that is still NaN was never
        # written. Saying which, and where, is the difference between "something is nan" and
        # "output row 127 is".
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
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:24s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        group_width, heads = column_tile(p["L"], p["N"], p["D"], 2)
        shape = f"{heads} head(s)" if heads else f"{p['D'] // group_width} slices/head"
        print(f"{case['id']}  (dtype={p['dtype']} N={p['N']} D={p['D']} L={p['L']} "
              f"stride={p['stride']} batches={len(p['seq_lens'])}, group={group_width} cols "
              f"= {shape}, launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name], TOLERANCE.get(name)):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
