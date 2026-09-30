# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent CPU model of the presence-masked softmax numerator.

No DSL and no simulator: this file decides what the answer is, and it decides it the
obvious way — build the whole ``[KEYS, QUERIES]`` predicate at once and exponentiate under
it. The kernel's chunked rebuild and its generation markers are one implementation of this
statement, and nothing here knows that either exists."""

import torch

# The same fixed geometry kernel.py declares. Only the chunk width and the generation count
# belong to the kernel's rebuild; the shapes here are what the stage's caller would hand it.
QUERIES, KEYS, TOPK = 64, 1024, 64
CHUNK, EPOCHS = 128, 4
CHUNKS = KEYS // CHUNK
NO_KEY = KEYS  # an index slot holding KEYS selects nothing; every row is TOPK slots wide


def _rows(selected):
    """Pad each query's chosen columns out to the fixed TOPK slots."""
    index = torch.full((QUERIES, TOPK), NO_KEY, dtype=torch.int32)
    for query, columns in enumerate(selected):
        if len(columns) > TOPK:
            raise ValueError(f"query {query} selects {len(columns)} columns; only {TOPK} slots exist")
        index[query, :len(columns)] = torch.tensor(sorted(columns), dtype=torch.int32)
    return index


def make_inputs(case):
    """Deterministic scores, a selection per query and the running maximum that goes with it."""
    pattern = case["parameters"]["pattern"]
    generator = torch.Generator().manual_seed(case["seed"])
    score_t = torch.randn((KEYS, QUERIES), generator=generator) * 3.0
    if pattern == "spread":
        # The ordinary case: every chunk holds some of every query's keys, so every rebuild
        # both writes new marks and steps over the previous generation's residue.
        selected = [torch.randperm(KEYS, generator=generator)[:TOPK].tolist() for _ in range(QUERIES)]
    elif pattern == "stale_overlap":
        # Eight columns in the even chunks, one in the odd ones. Every odd chunk must report
        # seven local rows absent while they still carry the previous chunk's marker.
        selected = [[chunk * CHUNK + (query + step) % CHUNK
                     for chunk in range(CHUNKS) for step in range(8 if chunk % 2 == 0 else 1)]
                    for query in range(QUERIES)]
    elif pattern == "wrap_after_four":
        # Chunk 0 and chunk EPOCHS carry the same generation number. Their column sets are
        # disjoint, so anything that survives the wrap shows up as a key that is not there.
        selected = [[step for step in range(32 - query % 8)]
                    + [EPOCHS * CHUNK + 64 + step for step in range(32)]
                    for query in range(QUERIES)]
    elif pattern == "one_chunk":
        selected = [[3 * CHUNK + (2 * step + query) % CHUNK for step in range(TOPK)] for query in range(QUERIES)]
    elif pattern == "empty_rows":
        selected = [torch.randperm(KEYS, generator=generator)[:TOPK // 2].tolist() if query % 2 == 0 else []
                    for query in range(QUERIES)]
    else:
        raise ValueError(f"Unknown generated presence pattern {pattern!r}")
    index = _rows(selected)
    present = presence(index)
    # A running maximum is what an attention kernel hands this stage; taking it over the
    # selected columns only is what keeps every exponent at or below zero.
    covered = torch.where(present.bool(), score_t, torch.full_like(score_t, -float("inf")))
    rowmax = covered.max(dim=0).values
    rowmax = torch.where(torch.isfinite(rowmax), rowmax, torch.zeros_like(rowmax)).reshape(1, QUERIES)
    return {"score_t": score_t.contiguous(), "index": index, "rowmax": rowmax.contiguous()}


def presence(index):
    """The predicate the kernel rebuilds, stated once over the whole tile."""
    present = torch.zeros((KEYS, QUERIES), dtype=torch.float32)
    for query in range(QUERIES):
        live = index[query][index[query] < KEYS].long()
        present[live, query] = 1.0
    return present


def reference(inputs):
    present = presence(inputs["index"])
    exponent = (inputs["score_t"].double() - inputs["rowmax"].double()).exp() * present.double()
    prob = exponent.float()
    return {"prob_t": prob, "present_t": present, "denom": exponent.sum(dim=0, keepdim=True).float()}
