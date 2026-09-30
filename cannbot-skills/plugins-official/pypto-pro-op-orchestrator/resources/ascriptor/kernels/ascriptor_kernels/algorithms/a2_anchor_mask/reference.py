# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent integer/Boolean anchor-mask reference and generated metadata."""

import torch


def document_ids(lengths, total):
    if type(total) is not int or not 1 <= total <= 2**24:
        raise ValueError("total_seq_len must be an integer in1..2^24")
    if not isinstance(lengths, list) or not lengths or any(type(length) is not int or length < 0 for length in lengths) or sum(lengths) > total:
        raise ValueError("Document lengths must be nonnegative integers summing to at most T")
    ids = torch.full((total,), -1, dtype=torch.int64)
    begin = 0
    for index, length in enumerate(lengths):
        ids[begin:begin + length] = index
        begin += length
    return ids


def metadata_for(lengths, total, anchors, block):
    docs = document_ids(lengths, total)
    qblock = torch.arange(anchors.numel() * block) // block
    qanchor = anchors[qblock]
    key_docs = torch.where(docs == -1, -2, docs)
    return {"kv_doc": key_docs.float().reshape(1, total),
            "kv_iota": torch.arange(total, dtype=torch.float32).reshape(1, total),
            "q_doc": docs[qanchor].float().reshape(1, -1),
            "q_anchor": qanchor.float().reshape(1, -1),
            "blk_idx": qblock.float().reshape(1, -1)}


def make_inputs(case):
    """Deterministic anchors and the five FP32 metadata rows the kernel reads. Key padding is
    marked -2 and query padding -1 so that the kernel's base test, which is an equality between
    a query's document and a key's document, is false for every padded query without needing a
    separate validity flag."""
    p = case["parameters"]
    total, block = p["T"], p["block_size"]
    document_ids(p["lengths"], total)
    if type(block) is not int or block <= 0 or type(p["anchors"]) is not int or not 1 <= p["anchors"] <= total:
        raise ValueError("Require positive block_size and1..T anchors")
    generator = torch.Generator().manual_seed(case["seed"])
    anchors = torch.randperm(total, generator=generator)[:p["anchors"]].contiguous()
    if "anchor_positions" in p:
        anchors = torch.tensor(p["anchor_positions"], dtype=torch.int64)
    inputs = {"lengths": list(p["lengths"]), "T": total, "block_size": block,
              "anchors": anchors, "block_dim": case.get("block_dim", 1),
              "sliding_window": p.get("sliding_window"), "synthetic_causal": p.get("synthetic_causal", False)}
    if anchors.numel() * block != p["Q"] or total + p["Q"] != p["KV"]:
        raise ValueError("Q/KV must match the anchor count and block size")
    if total % 64 or p["Q"] % 64 or not 1 <= p["Q"] <= 2**24:
        raise ValueError("T and Q must be positive multiples64 with exact FP32 integer metadata")
    if anchors.ndim != 1 or not ((anchors >= 0) & (anchors < total)).all():
        raise ValueError("Anchor positions must be within the base sequence")
    inputs["metadata"] = metadata_for(inputs["lengths"], total, anchors, block)
    return inputs


def reference(inputs):
    total, block, anchors = inputs["T"], inputs["block_size"], inputs["anchors"]
    docs = document_ids(inputs["lengths"], total)
    queries = anchors.numel() * block
    query_block = torch.arange(queries) // block
    query_anchor = anchors[query_block]
    query_doc = docs[query_anchor]
    base = (query_doc[:, None] >= 0) & (query_doc[:, None] == docs[None, :])
    base &= torch.arange(total)[None, :] < query_anchor[:, None]
    window = inputs.get("sliding_window")
    if window is not None and window >= 0:
        base &= torch.arange(total)[None, :] >= query_anchor[:, None] - window
    synthetic = query_block[:, None] == query_block[None, :]
    if inputs.get("synthetic_causal", False):
        synthetic &= torch.arange(queries)[None, :] <= torch.arange(queries)[:, None]
    return {"mask": torch.cat((base, synthetic), dim=1).to(torch.uint8)}
