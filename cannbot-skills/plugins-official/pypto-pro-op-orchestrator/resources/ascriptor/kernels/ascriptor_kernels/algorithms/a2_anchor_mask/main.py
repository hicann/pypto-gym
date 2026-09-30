# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the dense anchor mask through OpExec and check it byte for byte against the reference.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case window_causal_edges
    python main.py --launcher aclnn         # cce backend, on this machine's card

The kernel writes a dense uint8[Q, T+Q] mask. Its left half is the base sequence: a key is
visible when it shares the query anchor's document, sits strictly before that anchor, and --
when a window is given -- no earlier than anchor-window. Its right half is the synthetic
block: a query sees only its own anchor block, optionally truncated at the query's own row.

There is no integer comparison in the kernel. Every predicate is an FP32 subtraction clamped
to [0, 1] with `vmaxs`/`vmins` and then multiplied together, because the metadata is exact
integers in FP32 and a clamped difference of exact integers is exactly 0 or 1. The result
narrows FP32 -> FP16 -> uint8, and only 0 and 1 ever cross that path.

Case scale: the four `source_*` cases build the source's 2048 x 4096 mask. A whole-suite run
under sim or pipesim skips them, because each one costs a minute or two of simulation and
every mode combination they cover has a `model`-scale twin that runs in seconds. Naming one
with `--case` runs it anyway: `source_shape`, `source_window` and `source_causal` each take
80-120s and are exact. `source_window_causal` is the one that does not finish -- it is the
largest, on eight cores, and it passes the functional simulator's 120-second lane limit, which
`OpExec.__call__` does not forward its own `timeout=` to for the `sim` launcher. Run that case
on hardware with `--launcher aclnn`.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import kernel_for
from reference import make_inputs, reference

# The contract declares this body on both A2 and A3, and `kernel_for` binds it to one facade;
# change this to "a3" to run the identical source against the A3 target.
DEVICE = "a2"

# Bitwise. Every output byte is an independently computed Boolean, and the kernel's FP32 path
# can only produce exactly 0.0 or 1.0 before the narrowing, so there is nothing for a tolerance
# to absorb. Comparing as uint8 rather than as bool is deliberate: a byte of 2 is a defect that
# `bool(x)` would silently accept.
TOLERANCE = None

CASES = [
    {"id": "toy_2doc", "seed": 9600, "block_dim": 1, "scale": "model",
     "purpose": "Two documents, random anchors: the plain default with no window and no causal "
                "truncation",
     "parameters": {"T": 256, "lengths": [100, 156], "anchors": 16, "block_size": 8,
                    "Q": 128, "KV": 384}},
    {"id": "padding", "seed": 9601, "block_dim": 1, "scale": "model",
     "purpose": "Anchors placed on and around the two document ends, with 76 padded positions: "
                "a padded anchor has document -1 and must see no base key at all, while keeping "
                "its own synthetic block",
     "parameters": {"T": 256, "lengths": [80, 100], "anchors": 16, "block_size": 8,
                    "Q": 128, "KV": 384,
                    "anchor_positions": [0, 79, 80, 179, 180, 255, 100, 200, 250, 13, 89, 117,
                                         150, 177, 222, 244]}},
    {"id": "slot_reuse", "seed": 9602, "block_dim": 1, "scale": "model",
     "purpose": "Eight query tiles on one core, each sweeping eight base chunks and eight "
                "synthetic chunks: the double-buffered column slots wrap many times",
     "parameters": {"T": 512, "lengths": [200, 312], "anchors": 64, "block_size": 8,
                    "Q": 512, "KV": 1024}},
    {"id": "multi_core", "seed": 9603, "block_dim": 2, "scale": "model",
     "purpose": "Three documents over two cores: the query-tile split must not let one core "
                "publish into the other's rows",
     "parameters": {"T": 512, "lengths": [200, 100, 212], "anchors": 32, "block_size": 8,
                    "Q": 256, "KV": 768}},
    {"id": "edge_anchors", "seed": 9604, "block_dim": 1, "scale": "model",
     "purpose": "Anchors at both document boundaries, including position 0, whose block sees an "
                "empty prefix, and the last position of each document",
     "parameters": {"T": 64, "lengths": [32, 32], "anchors": 8, "block_size": 8,
                    "Q": 64, "KV": 128, "anchor_positions": [0, 31, 32, 63, 1, 30, 33, 62]}},
    {"id": "block_one", "seed": 9605, "block_dim": 2, "scale": "model",
     "purpose": "Block size 1: every query is its own block, so the synthetic half is exactly "
                "the identity matrix and an off-by-one in blk_idx is unmissable",
     "parameters": {"T": 128, "lengths": [64, 64], "anchors": 64, "block_size": 1,
                    "Q": 64, "KV": 192}},
    {"id": "all_padding", "seed": 9606, "block_dim": 1, "scale": "model",
     "purpose": "A single zero-length document: every position is padding, so the base half is "
                "entirely zero and only the synthetic blocks are set -- the control that proves "
                "an all-zero base half is computed rather than skipped",
     "parameters": {"T": 64, "lengths": [0], "anchors": 8, "block_size": 8,
                    "Q": 64, "KV": 128}},
    {"id": "source_shape", "seed": 9607, "block_dim": 4, "scale": "full_shape",
     "purpose": "The source's 2048x4096 mask on four cores, default modes",
     "parameters": {"T": 2048, "lengths": [384, 512, 256, 448, 320, 128], "anchors": 256,
                    "block_size": 8, "Q": 2048, "KV": 4096}},

    {"id": "window_zero", "seed": 9700, "block_dim": 1, "scale": "model",
     "purpose": "Window 0 admits no base key at all: key >= anchor and key < anchor cannot both "
                "hold, so the base half must be zero while the synthetic half is unchanged",
     "parameters": {"T": 64, "lengths": [32, 32], "anchors": 8, "block_size": 8, "Q": 64,
                    "KV": 128, "sliding_window": 0, "synthetic_causal": False,
                    "anchor_positions": [0, 31, 32, 63, 1, 30, 33, 62]}},
    {"id": "window_one", "seed": 9701, "block_dim": 1, "scale": "model",
     "purpose": "Window 1 admits exactly the key at anchor-1, when that key is in the same "
                "document: the narrowest non-empty window",
     "parameters": {"T": 64, "lengths": [32, 32], "anchors": 8, "block_size": 8, "Q": 64,
                    "KV": 128, "sliding_window": 1, "synthetic_causal": False,
                    "anchor_positions": [0, 31, 32, 63, 1, 30, 33, 62]}},
    {"id": "window_padding", "seed": 9702, "block_dim": 1, "scale": "model",
     "purpose": "A 16-key window with padded anchors: the window predicate and the document "
                "predicate must both be applied, not either one",
     "parameters": {"T": 256, "lengths": [80, 100], "anchors": 16, "block_size": 8, "Q": 128,
                    "KV": 384, "sliding_window": 16, "synthetic_causal": False,
                    "anchor_positions": [0, 79, 80, 179, 180, 255, 100, 200, 250, 13, 89, 117,
                                         150, 177, 222, 244]}},
    {"id": "window_maximum", "seed": 9703, "block_dim": 1, "scale": "model",
     "purpose": "Window 2^24, the largest the domain allows: the FP32 metadata still holds the "
                "difference exactly, and the answer must equal the unbounded case",
     "parameters": {"T": 64, "lengths": [32, 32], "anchors": 8, "block_size": 8, "Q": 64,
                    "KV": 128, "sliding_window": 16777216, "synthetic_causal": False,
                    "anchor_positions": [0, 31, 32, 63, 1, 30, 33, 62]}},
    {"id": "causal_full", "seed": 9704, "block_dim": 1, "scale": "model",
     "purpose": "Causal synthetic blocks with an unbounded base prefix: the diagonal is "
                "included, so row j of a block sees columns 0..j of that block",
     "parameters": {"T": 64, "lengths": [32, 32], "anchors": 8, "block_size": 8, "Q": 64,
                    "KV": 128, "synthetic_causal": True,
                    "anchor_positions": [0, 31, 32, 63, 1, 30, 33, 62]}},
    {"id": "causal_block_one", "seed": 9705, "block_dim": 2, "scale": "model",
     "purpose": "Causal with block size 1: the causal truncation and the same-block test agree "
                "on exactly the diagonal, so one wrong inequality shows as a whole empty row",
     "parameters": {"T": 128, "lengths": [64, 64], "anchors": 64, "block_size": 1, "Q": 64,
                    "KV": 192, "synthetic_causal": True}},
    {"id": "causal_padding", "seed": 9706, "block_dim": 1, "scale": "model",
     "purpose": "Causal with every position padded: a padded query keeps its causal synthetic "
                "triangle even though it owns no base key",
     "parameters": {"T": 64, "lengths": [0], "anchors": 8, "block_size": 8, "Q": 64,
                    "KV": 128, "synthetic_causal": True}},
    {"id": "window_causal_edges", "seed": 9707, "block_dim": 1, "scale": "model",
     "purpose": "Both modes at once on padded, boundary-placed anchors: window 32 on the base "
                "half and a causal triangle on the synthetic half are independent settings",
     "parameters": {"T": 256, "lengths": [80, 100], "anchors": 16, "block_size": 8, "Q": 128,
                    "KV": 384, "sliding_window": 32, "synthetic_causal": True,
                    "anchor_positions": [0, 79, 80, 179, 180, 255, 100, 200, 250, 13, 89, 117,
                                         150, 177, 222, 244]}},
    {"id": "window_causal_block3", "seed": 9708, "block_dim": 2, "scale": "model",
     "purpose": "Block size 3 with both modes: 192 queries in 64 blocks of three do not divide "
                "the 64-column synthetic chunk, so a block straddles the chunk boundary",
     "parameters": {"T": 128, "lengths": [63, 65], "anchors": 64, "block_size": 3, "Q": 192,
                    "KV": 320, "sliding_window": 33, "synthetic_causal": True}},
    {"id": "source_window", "seed": 9709, "block_dim": 4, "scale": "full_shape",
     "purpose": "The source's 2048x4096 mask with a 128-key window",
     "parameters": {"T": 2048, "lengths": [384, 512, 256, 448, 320, 128], "anchors": 256,
                    "block_size": 8, "Q": 2048, "KV": 4096, "sliding_window": 128,
                    "synthetic_causal": False}},
    {"id": "source_causal", "seed": 9710, "block_dim": 4, "scale": "full_shape",
     "purpose": "The source's 2048x4096 mask with causal synthetic blocks",
     "parameters": {"T": 2048, "lengths": [384, 512, 256, 448, 320, 128], "anchors": 256,
                    "block_size": 8, "Q": 2048, "KV": 4096, "synthetic_causal": True}},
    # The only case that does not finish under sim: eight lanes and both modes pass the
    # functional simulator's 120-second limit, and OpExec's `sim` branch calls run_kernel
    # without forwarding its own `timeout=`, so a caller cannot raise it. Run it on hardware.
    {"id": "source_window_causal", "seed": 9711, "block_dim": 8, "scale": "full_shape",
     "purpose": "The source's 2048x4096 mask with both modes on eight cores",
     "parameters": {"T": 2048, "lengths": [384, 512, 256, 448, 320, 128], "anchors": 256,
                    "block_size": 8, "Q": 2048, "KV": 4096, "sliding_window": 128,
                    "synthetic_causal": True}},
]


def execute(case, inputs, launcher, backend):
    """Launch the kernel. The destination is handed in filled with 0xA5 and seeded into the
    launch: this output is uint8, so there is no NaN to poison it with, and any byte that is
    neither 0 nor 1 in the result names a region no core wrote."""
    metadata = inputs["metadata"]
    queries = inputs["anchors"].numel() * inputs["block_size"]
    mask = torch.full((queries, inputs["T"] + queries), 0xA5, dtype=torch.uint8)
    window = inputs["sliding_window"]
    window = -1 if window is None or window < 0 else window
    op = OpExec(kernel_for(DEVICE), launcher=launcher, backend=backend, device=DEVICE,
                block_dim=case["block_dim"], out_dir=f"tmp/{launcher}", seed_outputs=True)
    return {"mask": op(*(metadata[name] for name in
                         ("kv_doc", "kv_iota", "q_doc", "q_anchor", "blk_idx")),
                       mask, inputs["T"], queries, inputs["block_size"], window,
                       int(inputs["synthetic_causal"]))}


def compare(name, got, want, tolerance):
    """Bitwise over uint8. `wrong` counts disagreeing bytes rather than only reporting a
    maximum, because on a mask of zeros and ones the interesting number is how many bytes
    moved and whether any of them is outside {0, 1}."""
    got, want = got.cpu(), want.cpu()
    ok = torch.equal(got, want)
    wrong = int((got != want).sum())
    stray = int(((got != 0) & (got != 1)).sum())
    worst = (got.float() - want.float()).abs().max().item()
    print(f"    {name:5s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}  "
          f"wrong_bytes={wrong}  non_boolean_bytes={stray}")
    if not ok:
        # Which bytes, not just how many: a contiguous run points at a store extent.
        outside = got != want
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
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:22s} {case['scale']:10s} {case['purpose']}")
        return 0

    selected = [case for case in CASES if args.case in ("all", case["id"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed, skipped = [], []
    for case in selected:
        # Naming a full-shape case with --case runs it; only a whole-suite simulator run skips.
        if (case["scale"] == "full_shape" and args.case == "all"
                and args.launcher in ("sim", "pipesim")):
            skipped.append(case["id"])
            continue
        p = case["parameters"]
        print(f"{case['id']}  (T={p['T']} Q={p['Q']}, window={p.get('sliding_window')}, "
              f"causal={p.get('synthetic_causal', False)}, launcher={args.launcher}, "
              f"block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    ran = len(selected) - len(skipped)
    print(f"\n{ran - len({f.split('/')[0] for f in failed})}/{ran} cases passed")
    if skipped:
        print(f"skipped under {args.launcher} (a 2048x4096 mask is a minute or two of "
              "simulation each and the model-scale cases cover every mode combination); "
              "name one with --case to run it anyway: " + ", ".join(skipped))
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
