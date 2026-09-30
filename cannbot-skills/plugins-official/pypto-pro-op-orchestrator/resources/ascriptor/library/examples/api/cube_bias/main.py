# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One cube tile with a bias, computed three ways, and each way catches a different mistake.

    python main.py                       # every case, functional simulator
    python main.py --list                # the case ids, with their purpose
    python main.py --case splitk         # one of them
    python main.py --launcher pipesim    # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn      # the cce backend, on this machine's card

The same 32x64 product with the same bias, staged three ways by `matmul`'s shortcut parameters: no
split, `splitn=32`, `splitk=16`. All three must produce identical bytes, and the host reference is
one matmul plus one bias addition -- it does not know which split ran.

The bias is `(arange(64) - 31) / 4`: non-zero, non-uniform, and a multiple of a quarter. Each of
those three properties buys a specific failure. Non-zero catches a bias that was never applied.
Non-uniform catches a split-N pass that took its bias from the wrong column half. And because the
values are exact in FP32 -- as are the small integer products -- a split-K pass that applied the
bias on both of its K accumulations instead of once shows up as an exact doubling rather than as
rounding.

The bias lives in L1 as a contiguous row moved by `gm_to_l1_pad`, not as ND2NZ-fractalized storage.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_bias
from reference import make_inputs, reference

DEVICE = "a5"

M, N, K = 32, 64, 32

OUTPUTS = ("o",)

CASES = [
    {"id": "none", "seed": 8901, "block_dim": 1,
     "purpose": "No split: the whole tile in one matmul. The baseline the other two have to match "
                "byte for byte, and the only case where the bias is applied exactly once by "
                "construction",
     "parameters": {"mode": "none"}},
    {"id": "splitn", "seed": 8901, "block_dim": 1,
     "purpose": "splitn=32: the 64 output columns are computed in two halves, so each half needs "
                "its own slice of the bias row. A pass that took the first slice twice is what the "
                "non-uniform bias is here to catch",
     "parameters": {"mode": "splitn"}},
    {"id": "splitk", "seed": 8901, "block_dim": 1,
     "purpose": "splitk=16: the K dimension is accumulated in two passes, and the bias belongs on "
                "the init tile only. Applied on both passes it doubles exactly, which a zero or "
                "uniform bias could not have shown",
     "parameters": {"mode": "splitk"}},
]


def check_domain(inputs, expected):
    """The three properties of the bias the cases depend on, and the exactness the comparison
    depends on. All of them are easy to break while editing the generator and none of them would
    announce itself -- the run would simply stop being able to fail."""
    x, y, bias = inputs["x"], inputs["y"], inputs["bias"]
    if x.shape != (M, K) or y.shape != (N, K) or bias.shape != (1, N):
        raise ValueError(f"the operands must be [{M}, {K}], [{N}, {K}] and a [1, {N}] bias row")
    # `(arange(64) - 31) / 4` crosses zero at column 31, so that one column cannot distinguish a
    # bias that was applied from one that was not. Every other column can, and no column at all
    # could if the row were uniform.
    if int((bias == 0).sum()) > 1:
        raise ValueError("more than one zero bias lane: those columns cannot detect a missing bias")
    if len(set(bias.flatten().tolist())) != N:
        raise ValueError("a uniform bias could not detect the wrong split-N slice")
    for name, tensor in (("bias", bias), ("x", x), ("y", y)):
        if not torch.equal(tensor * 4, (tensor * 4).round()):
            raise ValueError(f"{name} must be a multiple of a quarter, or the comparison cannot "
                             f"be bitwise")


def execute(case, inputs, launcher, backend):
    """One launch. The destination arrives NaN-poisoned and is seeded into the launch, so an output
    element the fixpipe never wrote reads back as NaN rather than as a plausible product."""
    op = OpExec(make_bias(inputs["mode"]), launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    return {"o": op(inputs["x"], inputs["y"], inputs["bias"],
                    torch.full((M, N), float("nan")))}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise. Small integer products and quarter-valued bias are exact in FP32, so every bit of
    the three split paths has to agree -- a tolerance here would absorb a doubled bias."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {M}x{N} FP32 elements")
    if not ok:
        got, want = got.cpu(), want.cpu()
        outside = (got != want)
        index = outside.nonzero()
        # A doubled bias differs from the reference by exactly the bias; a wrong slice does not.
        delta = (got - want)[outside]
        bias_like = bool(torch.equal(delta, delta.flatten()[0].expand_as(delta))) and delta.numel() > 1
        poisoned = int((outside & torch.isnan(got)).sum())
        print(f"      {len(index)}/{got.numel()} elements differ; first "
              f"{[tuple(i.tolist()) for i in index[:3]]}, deltas "
              f"{[round(float(d), 4) for d in delta[:3]]}"
              + ("  (a single constant delta everywhere: look for a bias applied twice)"
                 if bias_like else "")
              + (f"; {poisoned} still NaN-poisoned (never written)" if poisoned else ""))
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
            print(f"{case['id']:8s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        print(f"{case['id']}  (staging={case['parameters']['mode']}, {M}x{N}x{K}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
