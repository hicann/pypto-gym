# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Three A2/A3 cube output transfers: a half round trip, an atomic add, and a restored mode.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case seeded_add_restore  # one of them
    python main.py --device a3              # the same source against the A3 facade
    python main.py --launcher pipesim       # lowered pipeline: events, hazards, deadlock
    python main.py --launcher aclnn         # the cce backend, on this machine's card

  half_reuse   the FP32 L0C product is written into an **FP16** L1 tensor and fed back through a
               second matmul against an FP16 identity. `y`'s last column is 2^-12, so the exact FP32
               product carries fractions below an FP16 ULP -- the reference rounds explicitly and
               asserts the case actually observes the conversion.
  overwrite    a plain fixpipe store followed by `atomic_add()` of the same product, which is
               `2 * product`.
  seeded       `atomic_add()` onto a destination that arrives non-zero, then a plain store to a
               second non-zero destination -- so a lost initial value and a leaked atomic mode are
               two different failures on two different planes.

A note the source this came from earned: its header claimed to retain a non-zero prior output, while
its body stored the product and then added it. `overwrite` preserves that actual behaviour and
therefore establishes nothing about seeding; `seeded` is the case that does.

The seeded case only works because the destinations are seeded into the launch. A direct `OpExec`
caller has to ask for `seed_outputs=True`, as `execute` does here.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import make_kernels
from reference import make_inputs, reference

DEVICES = ("a2", "a3")

SHAPE = (32, 16)

OUTPUTS = ("o",)

CASES = [
    {"id": "half_conversion", "seed": 8741, "block_dim": 1,
     "purpose": "The FP32 product rounds into FP16 L1 before an identity second matmul. The last "
                "column of y is 2^-12, so the exact product has fractions an FP16 tensor cannot "
                "hold -- skipping the conversion changes the answer, and the reference refuses to "
                "run a case where it would not",
     "parameters": {"variant": "half_reuse", "planes": 1}},
    {"id": "half_conversion_second_seed", "seed": 8742, "block_dim": 1,
     "purpose": "The same conversion over a different draw, so the observable rounding is not a "
                "property of one input",
     "parameters": {"variant": "half_reuse", "planes": 1}},
    {"id": "overwrite_then_add", "seed": 8743, "block_dim": 1,
     "purpose": "A plain store of the product followed by an atomic add of the same product: the "
                "answer is 2 * product. This is the behaviour the original source had, as against "
                "the behaviour its header described, and it cannot establish output seeding at all",
     "parameters": {"variant": "overwrite", "planes": 1}},
    {"id": "seeded_add_restore", "seed": 8744, "block_dim": 1,
     "purpose": "What the previous case cannot show: the atomic destination arrives non-zero and "
                "the add accumulates onto it, and a following plain store to a second non-zero "
                "destination checks the atomic mode did not leak past its block. A lost initial "
                "value fails plane 0; a leaked mode fails plane 1",
     "parameters": {"variant": "seeded", "planes": 2}},
]


def check_domain(inputs, expected):
    """Each variant's own precondition. The half case needs a product FP16 cannot represent, or the
    conversion it is about would be invisible; the seeded case needs both destinations non-zero, or
    a lost initial value would look like a correct run."""
    variant = inputs["variant"]
    dtype = torch.float16 if variant == "half_reuse" else torch.float32
    if inputs["x"].shape != SHAPE or inputs["y"].shape != (SHAPE[1], SHAPE[1]):
        raise ValueError(f"this family declares x{SHAPE} and y{(SHAPE[1], SHAPE[1])}")
    if inputs["x"].dtype != dtype or inputs["y"].dtype != dtype:
        raise ValueError(f"the {variant} variant takes {dtype} operands")
    product = inputs["x"].float() @ inputs["y"].float().T
    if variant == "half_reuse" and torch.equal(product, product.half().float()):
        raise ValueError("this case's product is representable in FP16, so the intermediate "
                         "conversion would not be observable")
    if variant == "seeded":
        for name in ("initial_z", "initial_after"):
            if not bool((inputs[name] != 0).all()):
                raise ValueError(f"{name} must be non-zero for the seeded case to mean anything")


def execute(case, inputs, launcher, backend, device):
    """One launch per case, with `seed_outputs=True` so the seeded case's non-zero destinations reach
    the device. Each execution hands in fresh clones, so a case that accumulates cannot accumulate
    across runs."""
    variant = inputs["variant"]
    entry = make_kernels(device)[variant]
    op = OpExec(entry, launcher=launcher, backend=backend, device=device,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}/{case['id']}",
                seed_outputs=True)
    if variant == "half_reuse":
        args = (inputs["x"], inputs["y"], torch.eye(SHAPE[1]).half(),
                torch.full(SHAPE, float("nan")))
    elif variant == "overwrite":
        args = (inputs["x"], inputs["y"], inputs["initial_z"].clone())
    else:
        args = (inputs["x"], inputs["y"], inputs["initial_z"].clone(),
                inputs["initial_after"].clone())
    produced = op(*args)
    planes = (torch.stack(list(produced)) if isinstance(produced, (tuple, list))
              else produced.unsqueeze(0))
    return {"o": planes}


def raw(tensor):
    """The tensor's stored bytes, flattened, so a comparison sees what was written."""
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def compare(name, got, want):
    """Bitwise over every plane: the products are exact FP32 and the intermediate FP16 rounding is
    explicit in the reference."""
    if got.dtype != want.dtype or got.shape != want.shape:
        print(f"    {name:9s} FAIL  {got.dtype}{tuple(got.shape)} != "
              f"{want.dtype}{tuple(want.shape)}")
        return False
    ok = torch.equal(raw(got), raw(want))
    print(f"    {name:9s} {'ok  ' if ok else 'FAIL'}  bitwise over {got.shape[0]} plane(s) of "
          f"{got.shape[1]}x{got.shape[2]}")
    if not ok:
        got, want = got.cpu(), want.cpu()
        for plane in range(got.shape[0]):
            outside = got[plane] != want[plane]
            if not outside.any():
                continue
            index = outside.nonzero()
            # A doubled or missing accumulation is a constant ratio; a leaked atomic mode adds the
            # destination's own prior value.
            ratios = sorted({round(float(got[plane][outside][i] / want[plane][outside][i]), 4)
                             for i in range(min(4, int(outside.sum())))
                             if want[plane][outside][i] != 0})
            print(f"      plane {plane}: {int(outside.sum())} elements differ, first "
                  f"{[tuple(v.tolist()) for v in index[:3]]}, got/want ratios {ratios}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--device", default=DEVICES[0], choices=DEVICES)
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:28s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['variant']}, {p['planes']} plane(s), device={args.device}, "
              f"launcher={args.launcher})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        check_domain(inputs, expected)
        actual = execute(case, inputs, args.launcher, args.backend, args.device)
        for name in OUTPUTS:
            if not compare(name, actual[name], expected[name]):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
