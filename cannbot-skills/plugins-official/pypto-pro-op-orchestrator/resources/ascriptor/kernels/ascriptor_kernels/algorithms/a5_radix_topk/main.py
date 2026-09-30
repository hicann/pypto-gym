# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the MSD radix top-k through OpExec and check it against a Torch top-k.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn         # cce backend, on this machine's card
    python main.py --launcher pypto --case ties_2048_64

The kernel writes two buffers: the selected values and the input indices they came from.
Its output order is unspecified, and at a tied threshold several index sets are equally
correct top-k answers, so what is compared here is the sorted value multiset. The indices
come back from the launch and are deliberately not compared: pinning them would make a
legal tie-break look like a defect.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import radix_kernel
from reference import make_inputs, reference

# Bitwise. Selection moves values, it never computes them: every returned value is a copy of
# an input word, so the sorted multiset has to match Torch's top-k to the last bit. A
# tolerance here would buy nothing and would hide the one defect that matters -- a value
# taken from the wrong element. Signed zeros are the reason the comparison is over values
# rather than over indices: +0.0 and -0.0 are numerically tied, so either may be selected.
TOLERANCE = None

CASES = [
    {"id": "random_1000_10", "seed": 31010, "block_dim": 1,
     "purpose": "The source's own case: 1000 Gaussian keys and a small k",
     "parameters": {"count": 1000, "k": 10, "flavour": "random"}},
    {"id": "random_4096_512", "seed": 127488, "block_dim": 1,
     "purpose": "Both maxima at once: the full 4096-key buffer with no padding left, and the "
                "largest k the output holds",
     "parameters": {"count": 4096, "k": 512, "flavour": "random"}},
    {"id": "random_257_33", "seed": 8000, "block_dim": 1,
     "purpose": "Neither the count nor k lands on a 64-lane boundary",
     "parameters": {"count": 257, "k": 33, "flavour": "random"}},
    {"id": "narrow_1000_10", "seed": 31010, "block_dim": 1,
     "purpose": "Keys within 1e-6 of 1.0: the first three radix bytes are identical, so the "
                "whole decision falls to the last level",
     "parameters": {"count": 1000, "k": 10, "flavour": "narrow"}},
    {"id": "ties_2048_64", "seed": 63552, "block_dim": 1,
     "purpose": "Five distinct values over 2048 keys: the threshold is heavily tied and "
                "compaction has to stop at exactly k",
     "parameters": {"count": 2048, "k": 64, "flavour": "ties"}},
    {"id": "signed_600_7", "seed": 18607, "block_dim": 1,
     "purpose": "Mixed signs: descending FP32 order is not the order of the raw words, so the "
                "sign bit has to be folded before the byte levels run",
     "parameters": {"count": 600, "k": 7, "flavour": "signed"}},
    {"id": "random_1_1", "seed": 32, "block_dim": 1,
     "purpose": "One key, k = 1: the other 4095 buffer entries are -inf padding and must not win",
     "parameters": {"count": 1, "k": 1, "flavour": "random"}},
    {"id": "ascending_63_63", "seed": 2016, "block_dim": 1,
     "purpose": "k = count one lane below the 64-wide vector: every key is selected",
     "parameters": {"count": 63, "k": 63, "flavour": "ascending"}},
    {"id": "ascending_64_64", "seed": 2048, "block_dim": 1,
     "purpose": "k = count exactly on the 64-lane boundary",
     "parameters": {"count": 64, "k": 64, "flavour": "ascending"}},
    {"id": "ascending_65_65", "seed": 2080, "block_dim": 1,
     "purpose": "k = count one lane past the boundary: a second vector holding a single key",
     "parameters": {"count": 65, "k": 65, "flavour": "ascending"}},
    {"id": "random_512_512", "seed": 16384, "block_dim": 1,
     "purpose": "k = count = the output capacity: the candidate buffer fills exactly and "
                "nothing is rejected",
     "parameters": {"count": 512, "k": 512, "flavour": "random"}},
    {"id": "all_equal_257_33", "seed": 8000, "block_dim": 1,
     "purpose": "Every key equal: all 257 sit at the threshold, so any 33 distinct indices are "
                "a correct answer and only the value multiset is decidable",
     "parameters": {"count": 257, "k": 33, "flavour": "all_equal"}},
    {"id": "signed_zero_257_33", "seed": 8000, "block_dim": 1,
     "purpose": "+0.0 and -0.0 are equal numbers with different words: either may be selected, "
                "and a radix pass that compares raw words would order them",
     "parameters": {"count": 257, "k": 33, "flavour": "signed_zero"}},
    {"id": "ascending_129_7", "seed": 4006, "block_dim": 1,
     "purpose": "A strict ramp: the threshold is unique, so exactly one selection is legal",
     "parameters": {"count": 129, "k": 7, "flavour": "ascending"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the kernel. Both destinations are handed in poisoned -- NaN for the values and
    -1 for the indices -- and seeded into the launch, so a candidate slot the kernel never
    writes reads back as poison rather than as a plausible key. The source's full 512-slot
    DMA is kept, so only the first k entries of either buffer mean anything."""
    op = OpExec(radix_kernel, launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    values = torch.full((1, 512), float("nan"), dtype=torch.float32)
    indices = torch.full((1, 512), -1, dtype=torch.int32)
    out_values, out_indices = op(inputs["keys"], values, indices, inputs["count"], inputs["k"])
    selected = out_values.flatten()[:inputs["k"]]
    return {"values_descending": torch.sort(selected, descending=True).values}


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
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:18s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
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
        print(f"{case['id']}  (count={p['count']} k={p['k']} {p['flavour']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
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
