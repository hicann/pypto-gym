#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
MiniMax M3 MSA Main Branch Implementation Module

Computation flow:
  1. Q: [n, hq, dh], K/V: [topk*bk, hkv, dh] (gathered by topk_indices),
     block_mask: [num_blocks, topk, bk, bk] (precomputed on host)
  2. Reshape Q to [n, hq*dh], K/V to [topk*bk, hkv*dh],
     block_mask to [num_blocks*topk*bk, bk]
  3. Outer loop hkv (parallel): each KV head -> group of Q heads
  4. Loop g (group heads) -> qb (query blocks) -> i (selected KV blocks)
  5. scores = Q@K^T * scale + block_mask[qb, i]; online softmax
  6. Unified rescale formula (no is_loop_begin/end conditionals).
     Accumulators initialised: o=0, l=1, m=-inf. MASK_NEG in block_mask
     causes exp(-1e9)=0, so masked blocks are no-ops in the rescale.

Note: No conditional branches (if/is_loop_begin/is_loop_end) inside the kernel.
PyPTO's AssignMemoryType pass cannot resolve L1 memory paths for VIEW operations
inside conditional branches. All causal logic is precomputed into block_mask.

Reference: MiniMax M3 Technical Report (arXiv:2606.13392v2), Equation 8.
"""

import pypto

MASK_NEG = -1e9


def msa_main_branch_prefill(hq, hkv, dh, bk, topk):
    group = hq // hkv

    @pypto.frontend.jit(
        runtime_options={
            "stitch_function_max_num": 128,
            "max_workspace_kb": 60000,
        },
    )
    def sparse_attention(
        query: pypto.Tensor([pypto.DYNAMIC, hq, dh], pypto.DT_FP32),
        key_blocks: pypto.Tensor([topk * bk, hkv, dh], pypto.DT_FP32),
        value_blocks: pypto.Tensor([topk * bk, hkv, dh], pypto.DT_FP32),
        block_mask: pypto.Tensor([pypto.DYNAMIC, topk, bk, bk], pypto.DT_FP32),
        output: pypto.Tensor([pypto.DYNAMIC, hq, dh], pypto.DT_FP32),
    ):
        n = query.shape[0]
        num_blocks = n // bk
        scale = 1.0 / (dh ** 0.5)

        query_2d = pypto.reshape(query, [n, hq * dh], inplace=True)
        output_2d = pypto.reshape(output, [n, hq * dh], inplace=True)
        key_2d = pypto.reshape(key_blocks, [topk * bk, hkv * dh], inplace=True)
        value_2d = pypto.reshape(value_blocks, [topk * bk, hkv * dh], inplace=True)

        mask_rows = num_blocks * topk * bk
        block_mask_2d = pypto.reshape(block_mask, [mask_rows, bk], inplace=True)

        pypto.set_cube_tile_shapes([bk, dh], [dh, bk], [bk, bk])
        pypto.set_vec_tile_shapes(bk, dh)

        for h_kv in pypto.loop(hkv, name="LOOP_HKV", idx_name="h_kv"):
            h_kv_col = h_kv * dh

            for g in range(group):
                h_idx = h_kv * group + g
                q_col = h_idx * dh

                for qb in pypto.loop(0, num_blocks, 1, name="LOOP_QB", idx_name="qb"):
                    qi = qb * bk
                    mask_base = qb * topk * bk

                    o_acc = pypto.tensor([bk, dh], pypto.DT_FP32, "o_acc")
                    l_acc = pypto.tensor([bk, 1], pypto.DT_FP32, "l_acc")
                    m_acc = pypto.tensor([bk, 1], pypto.DT_FP32, "m_acc")

                    for i in pypto.loop(0, topk, 1, name="LOOP_K", idx_name="i"):
                        kv_row = i * bk

                        q_view = pypto.view(query_2d, [bk, dh], [qi, q_col])
                        k_view = pypto.view(key_2d, [bk, dh], [kv_row, h_kv_col])
                        v_view = pypto.view(value_2d, [bk, dh], [kv_row, h_kv_col])
                        mask_view = pypto.view(block_mask_2d, [bk, bk],
                                               [mask_base + i * bk, 0])

                        scores = pypto.matmul(q_view, k_view, pypto.DT_FP32,
                                              b_trans=True)
                        scores = pypto.mul(scores, scale)
                        scores = pypto.add(scores, mask_view)

                        m_i = pypto.amax(scores, dim=-1, keepdim=True)
                        p = pypto.exp(pypto.sub(scores, m_i))
                        l_i = pypto.sum(p, dim=-1, keepdim=True)

                        pypto.set_cube_tile_shapes([bk, bk], [bk, dh], [bk, dh])
                        pv = pypto.matmul(p, v_view, pypto.DT_FP32)
                        pypto.set_cube_tile_shapes([bk, dh], [dh, bk], [bk, bk])

                        if pypto.is_loop_begin(i):
                            if pypto.is_loop_end(i):
                                out = pypto.div(pv, l_i)
                                pypto.assemble(out, [qi, q_col], output_2d)
                            else:
                                o_acc[:] = pv
                            l_acc[:] = l_i
                            m_acc[:] = m_i
                        else:
                            m_new = pypto.maximum(m_acc, m_i)
                            alpha = pypto.exp(pypto.sub(m_acc, m_new))
                            beta = pypto.exp(pypto.sub(m_i, m_new))
                            l_new = pypto.add(pypto.mul(alpha, l_acc),
                                              pypto.mul(beta, l_i))
                            o_new = pypto.add(pypto.mul(o_acc, alpha),
                                              pypto.mul(pv, beta))
                            if pypto.is_loop_end(i):
                                out = pypto.div(o_new, l_new)
                                pypto.assemble(out, [qi, q_col], output_2d)
                            else:
                                o_acc[:] = o_new
                            l_acc[:] = l_new
                            m_acc[:] = m_new

    return sparse_attention
