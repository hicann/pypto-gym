# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the cube product plus SIMT transpose through OpExec and check it against the
independent FP64 reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --launcher pipesim       # lowered pipeline, events and hazards
    python main.py --launcher aclnn         # cce backend, on this machine's card
    python main.py --case contiguous_read_loop_reuse

This is the `@simt` surface, not `@vf`. A vector function is written once and applies to a
whole register; a SIMT body is written from one thread's point of view and the launch runs
1024 of them, so the transpose is spelled as `for i in range(simt_thread_id(), total,
simt_thread_num())` and correctness is a statement about which thread owns which element.

The two variants differ only in how that flat index `i` is decomposed, and the choice decides
which side of the copy is contiguous:

  contiguous_read   `row = i // N, col = i % N`, so `i` walks the UB tile in row order and the
                    GM write is strided by M.
  contiguous_write  `col = i // M, row = i % M`, so `i` walks the GM output in row order and
                    the UB read is strided by N.

Neither is ranked here. What the pair is for is that the same transpose has two legitimate
index decompositions, and only one of them can have a contiguous destination.

Ownership is the other half of the demo. The cube result is published to both vector peers
(`l0c_to_ub` twice, `sub_block_id=0` then `1`) under one `CvMutex`, but only sub-block 0 runs
the SIMT body, and only cube group 0 runs at all. The original source let both vector peers
write the same GM elements; the values agreed, which is exactly why it looked safe and was
not -- two non-atomic writers of one element is a write-write hazard whatever they write.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import (simt_matmul_transpose_contig_read_kernel,
                    simt_matmul_transpose_contig_write_kernel)
from reference import make_inputs, reference

ENTRIES = {"contiguous_read": simt_matmul_transpose_contig_read_kernel,
           "contiguous_write": simt_matmul_transpose_contig_write_kernel}

# The transpose itself moves values without touching them, so the whole budget belongs to the
# FP32 cube accumulation that produced them, against an FP64 product rounded once. The relative
# residual is what makes the zero case mean something: a kernel that writes nothing where the
# answer is small passes an absolute bound on its own.
TOLERANCE = {"rtol": 0.0001, "atol": 0.0001, "max_relative_l2": 1e-05}

CASES = [
    {"id": "contiguous_write_source", "seed": 0, "block_dim": 1,
     "purpose": "The original 32x16x8 shape and seed-0 stream, contiguous-GM-write order",
     "parameters": {"variant": "contiguous_write", "M": 32, "N": 16, "K": 8,
                    "distribution": "random"}},
    {"id": "contiguous_read_source", "seed": 0, "block_dim": 1,
     "purpose": "The identical input through the contiguous-UB-read order: the two index "
                "decompositions must agree element for element",
     "parameters": {"variant": "contiguous_read", "M": 32, "N": 16, "K": 8,
                    "distribution": "random"}},
    {"id": "contiguous_read_draft_seed2026", "seed": 2026, "block_dim": 1,
     "purpose": "The released draft's own seed on the source shape, kept as a separate case",
     "parameters": {"variant": "contiguous_read", "M": 32, "N": 16, "K": 8,
                    "distribution": "random"}},
    {"id": "contiguous_read_rectangular", "seed": 2027, "block_dim": 1,
     "purpose": "M=16, N=48 with exact integer values: a non-square output where a swapped "
                "index is visibly wrong rather than merely close",
     "parameters": {"variant": "contiguous_read", "M": 16, "N": 48, "K": 16,
                    "distribution": "integer"}},
    {"id": "contiguous_read_thread_tail", "seed": 2028, "block_dim": 1,
     "purpose": "256 elements for 1024 threads: three quarters of them never enter the loop",
     "parameters": {"variant": "contiguous_read", "M": 16, "N": 16, "K": 8,
                    "distribution": "random"}},
    {"id": "contiguous_read_loop_reuse", "seed": 2029, "block_dim": 1,
     "purpose": "3072 elements for 1024 threads: every thread makes three strided loop visits",
     "parameters": {"variant": "contiguous_read", "M": 64, "N": 48, "K": 32,
                    "distribution": "random"}},
    {"id": "contiguous_read_idle_cores", "seed": 2030, "block_dim": 3,
     "purpose": "Three cube groups launched and only group 0 computes: the other two must not "
                "reach the output",
     "parameters": {"variant": "contiguous_read", "M": 32, "N": 16, "K": 8,
                    "distribution": "random"}},
    {"id": "contiguous_read_zero", "seed": 2031, "block_dim": 2,
     "purpose": "An all-zero product into a NaN-poisoned output: zero is the right answer, so "
                "only the poison distinguishes a correct kernel from one that wrote nothing",
     "parameters": {"variant": "contiguous_read", "M": 16, "N": 32, "K": 16,
                    "distribution": "zero"}},

    {"id": "contiguous_write_draft_seed2036", "seed": 2036, "block_dim": 1,
     "purpose": "The write-order draft seed on the source shape",
     "parameters": {"variant": "contiguous_write", "M": 32, "N": 16, "K": 8,
                    "distribution": "random"}},
    {"id": "contiguous_write_rectangular", "seed": 2037, "block_dim": 1,
     "purpose": "The non-square integer case through the write-order decomposition",
     "parameters": {"variant": "contiguous_write", "M": 16, "N": 48, "K": 16,
                    "distribution": "integer"}},
    {"id": "contiguous_write_thread_tail", "seed": 2038, "block_dim": 1,
     "purpose": "256 elements for 1024 threads, write-order decomposition",
     "parameters": {"variant": "contiguous_write", "M": 16, "N": 16, "K": 8,
                    "distribution": "random"}},
    {"id": "contiguous_write_loop_reuse", "seed": 2039, "block_dim": 1,
     "purpose": "3072 elements, three loop visits per thread, write-order decomposition",
     "parameters": {"variant": "contiguous_write", "M": 64, "N": 48, "K": 32,
                    "distribution": "random"}},
    {"id": "contiguous_write_idle_cores", "seed": 2040, "block_dim": 3,
     "purpose": "Three cube groups with two idle, write-order decomposition",
     "parameters": {"variant": "contiguous_write", "M": 32, "N": 16, "K": 8,
                    "distribution": "random"}},
    {"id": "contiguous_write_zero", "seed": 2041, "block_dim": 2,
     "purpose": "The zero-into-poison control for the write-order decomposition",
     "parameters": {"variant": "contiguous_write", "M": 16, "N": 32, "K": 16,
                    "distribution": "zero"}},
]


def execute(case, inputs, launcher, backend):
    """Launch the index order this case names. The [N, M] destination is handed in poisoned
    with NaN and seeded into the launch, so an element no thread claims reads back as NaN --
    which is the only way the zero cases prove anything."""
    p = case["parameters"]
    out = torch.full((p["N"], p["M"]), float("nan"))
    op = OpExec(ENTRIES[p["variant"]], launcher=launcher, backend=backend, device="a5",
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"out": op(inputs["a"], inputs["b"], out, p["M"], p["N"], p["K"])}


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
            print(f"{case['id']:32s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  (M={p['M']} N={p['N']} K={p['K']}, {p['distribution']}, "
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
