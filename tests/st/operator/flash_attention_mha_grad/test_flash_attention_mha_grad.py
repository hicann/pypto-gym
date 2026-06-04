#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Flash Attention Backward with Dynamic Variable Length Sequences

语义约定:
  - Q 侧: s1_size (Q seqlen), 张量包括 Q/O/dO/L/M/dQ
  - KV 侧: s2_size (KV seqlen), 张量包括 K/V/dK/dV
  - S2_TILE: KV 序列维度的分块大小 (将 s2_size 切分为多个 tile 迭代)

3 loops: batch + head + kv_tile (KV sequence tiling).
Tiles KV sequence dimension by S2_TILE to reduce intermediate attention matrix
from [s1_size, s2_size] to [s1_size, S2_TILE] per iteration.
dK and dV are accumulated across kv tiles.
"""


import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import logging
from dataclasses import dataclass

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch
import torch_npu

from flash_attention_mha_grad_impl \
    import flash_attention_varlen_backward_kernel_small_seq, flash_attention_mha_grad_kernel_long_seq


logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM

# KV 序列维度的分块大小 (全局配置常量)
S2_TILE = 256


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set TILE_FWK_DEVICE_ID before running:")
        logging.info("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        return None


@pytest.fixture(scope="module")
def device():
    device_id = get_device_id()
    if device_id is None:
        pytest.skip("TILE_FWK_DEVICE_ID not set")
    torch.npu.set_device(device_id)
    return f'npu:{device_id}'


########################################################################
# 公共工具函数
########################################################################


@dataclass
class TileConfig:
    """
    分块配置结构体，封装 kernel 所需的 tile 参数。

    将 tile 信息集中管理，避免在 run_test 中直接引用全局变量。

    Attributes:
        s2_tile:     KV 序列维度的分块大小 (kernel 中 S2_TILE,
                     用于将 KV seqlen 切分为多个 tile 迭代)
    """
    s2_tile: int = S2_TILE


def create_inputs(batch_size, s1_size, s2_size, num_heads, head_dim, device,
                  q_seqlens=None, kv_seqlens=None):
    """
    创建 varlen 布局的输入张量。

    布局说明 (varlen, 通过 cumsum 描述每 batch 的 seq):
      - Q/O/dO/L/M/dQ: 第 i 个 batch 占用 q_seqlens[i] 行
      - K/V/dK/dV:      第 i 个 batch 占用 kv_seqlens[i] 行
      - actual_q[0]=0, actual_q[i+1] = sum(q_seqlens[:i+1])
      - actual_kv 同理

    数据类型严格对应 kernel 签名:
      - q/k/v:       BF16  (kernel: pypto.DT_BF16)
      - actual_q:    INT32 (kernel: pypto.DT_INT32) — Q seqlen 的前缀累加 (cumsum)
      - actual_kv:   INT32 (kernel: pypto.DT_INT32) — KV seqlen 的前缀累加 (cumsum)

    Args:
        batch_size: 批次数量
        s1_size:    Q 序列长度 (当 q_seqlens=None 时使用, 每个 batch 等长)
        s2_size:    KV 序列长度 (当 kv_seqlens=None 时使用)
        num_heads:  注意力头数
        head_dim:   每个头的维度
        device:     计算设备
        q_seqlens:  可选, 每个 batch 的 Q seqlen 列表, len==batch_size
                    若为 None 则用 [s1_size]*batch_size
        kv_seqlens: 可选, 每个 batch 的 KV seqlen 列表, len==batch_size
                    若为 None 则用 [s2_size]*batch_size
    Returns:
        q:          shape=[sum(q_seqlens), num_heads, head_dim], dtype=bfloat16
        k, v:       shape=[sum(kv_seqlens), num_heads, head_dim], dtype=bfloat16
        actual_q:   shape=[batch_size + 1], dtype=int32 — Q seqlen 前缀累加
        actual_kv:  shape=[batch_size + 1], dtype=int32 — KV seqlen 前缀累加
        q_seqlens:  list[int], 每个 batch 的 Q seqlen
        kv_seqlens: list[int], 每个 batch 的 KV seqlen
    """
    if q_seqlens is None:
        q_seqlens = [s1_size] * batch_size
    if kv_seqlens is None:
        kv_seqlens = [s2_size] * batch_size
    assert len(q_seqlens) == batch_size
    assert len(kv_seqlens) == batch_size

    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    torch.manual_seed(42)
    # Q 张量: shape=[sum(q_seqlens), num_heads, head_dim], dtype=BF16
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.01
    # K/V 张量: shape=[sum(kv_seqlens), num_heads, head_dim], dtype=BF16
    k = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.01
    v = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.01

    # 前缀累加构造 actual_q / actual_kv (shape=[batch_size + 1])
    q_cumsum = [0]
    for sq in q_seqlens:
        q_cumsum.append(q_cumsum[-1] + sq)
    actual_q = torch.tensor(q_cumsum, dtype=torch.int32, device=device)
    kv_cumsum = [0]
    for skv in kv_seqlens:
        kv_cumsum.append(kv_cumsum[-1] + skv)
    actual_kv = torch.tensor(kv_cumsum, dtype=torch.int32, device=device)
    return q, k, v, actual_q, actual_kv, q_seqlens, kv_seqlens


def attention_backward_golden(q, k, v, o_input, do_t, scale):
    """
    Golden reference: 严格模拟 kernel 内部的 dtype 转换流程。

    Q: [s1_size, head_dim],  KV: [s2_size, head_dim]
    输入 q/k/v/o_input/do_t 均为 BF16, 与 kernel 签名一致。

    Kernel dtype 转换对照:
      1. O, dO: BF16 → cast → FP32               (pypto.cast → DT_FP32)
      2. scores = Q(BF16) @ K^T(BF16) → FP32      (matmul out_dtype=FP32)
      3. P = softmax(scores) → FP32                (FP32 全程)
      4. dP = dO(BF16) @ V^T(BF16) → FP32          (matmul out_dtype=FP32)
      5. D = sum(O_fp32 * dO_fp32) → FP32
      6. dS = P * (dP - D) → FP32
      7. ds_half = cast(dS, BF16)                  (pypto.cast → DT_BF16)
         p_half  = cast(P, BF16)                   (pypto.cast → DT_BF16)
      8. dK = ds_half^T(BF16) @ Q(BF16) → FP32 * scale → cast BF16
      9. dV = p_half^T(BF16) @ dO(BF16) → FP32     (matmul out_dtype=FP32, 改为 FP32 以支持读-改-写累加)
     10. dQ_partial = ds_half(BF16) @ K(BF16) → FP32 * scale
         dQ 跨 tile 累加 (FP32), 最终 cast BF16

    Args:
        q:       [s1_size, head_dim] BF16 — Q 切片
        k, v:    [s2_size, head_dim] BF16 — KV 切片
        o_input: [s1_size, head_dim] BF16 — 前向输出 O (预计算, 与 kernel 输入一致)
        do_t:    [s1_size, head_dim] BF16 — dO 切片
        scale:   attention scale factor (1/sqrt(head_dim))
    Returns:
        dq: [s1_size, head_dim] BF16, dk: [s2_size, head_dim] BF16, dv: [s2_size, head_dim] BF16
    """
    # 计算流：scores = Q(BF16) @ K^T(BF16) → FP32 (kernel: matmul out_dtype=FP32)
    scores = torch.matmul(q.float(), k.float().T) * scale
    # 计算流： P = softmax (kernel: exp(S*scale - M) / L, 全程 FP32)
    p = torch.softmax(scores, dim=-1)

    #  计算流：---- D = sum(O_fp32 * dO_fp32, dim=-1) ----
    # kernel: O 来自输入张量 (BF16) → cast(O, FP32) * cast(dO, FP32) → sum
    # 使用传入的 o_input (BF16) 而非重算, 与 kernel 严格一致
    d = (o_input.float() * do_t.float()).sum(dim=-1, keepdim=True)

    #  计算流： ---- dP = dO(BF16) @ V^T(BF16) → FP32 ----
    dp = torch.matmul(do_t.float(), v.float().T)

    #  计算流：---- dS = P * (dP - D) → FP32 ----
    ds = p * (dp - d)

    # ---- 中间 cast: FP32 → BF16 (kernel: pypto.cast → DT_BF16) ----
    ds_half = ds.to(torch.bfloat16)
    p_half = p.to(torch.bfloat16)

    #  计算流：---- dK = ds_half^T(BF16) @ Q(BF16) → FP32 * scale → BF16 ----
    #  计算流：kernel: matmul(ds_half, qi, out_dtype=FP32, a_trans=True) 即 ds_half^T @ Q
    #  计算流：ds_half: [s1, s2], Q: [s1, D] → ds_half^T: [s2, s1] @ Q: [s1, D] → [s2, D]
    dk_fp32 = torch.matmul(ds_half.float().T, q.float()) * scale
    dk = dk_fp32.to(torch.bfloat16)

    #  计算流：---- dV = p_half^T(BF16) @ dO(BF16) → BF16 ----
    #  计算流：kernel: matmul(p_half, doi, out_dtype=BF16, a_trans=True) 即 p_half^T @ dO
    #  计算流：p_half: [s1, s2], dO: [s1, D] → p_half^T: [s2, s1] @ dO: [s1, D] → [s2, D]
    # 注: 模拟 BF16 matmul — 先 FP32 计算再 cast BF16
    dv = torch.matmul(p_half.float().T, do_t.float()).to(torch.bfloat16)

    #  计算流：---- dQ = ds_half(BF16) @ K(BF16) → FP32 * scale → BF16 ----
    #  计算流：kernel: matmul(ds_half, ki_tile, out_dtype=FP32) → mul(scale) → 累加(FP32) → cast(BF16)
    dq_fp32 = torch.matmul(ds_half.float(), k.float()) * scale
    dq = dq_fp32.to(torch.bfloat16)

    return dq, dk, dv


def compute_l_m_o(q, k, v, scale):
    """
    预计算 softmax 中间量 L, M 和前向输出 O。

    Q: [s1_size, head_dim],  KV: [s2_size, head_dim]

    对应 kernel 输入:
      - l_input: pypto.DT_FP32 — softmax 分母 L
      - m_input: pypto.DT_FP32 — softmax 最大值 M
      - o:       pypto.DT_BF16 — 前向注意力输出 O

    计算过程:
      scores = Q @ K^T * scale        [s1_size, s2_size]  (FP32)
      M = max(scores, dim=-1)         [s1_size, 1]        (FP32, 数值稳定)
      P_unnorm = exp(scores - M)      [s1_size, s2_size]  (FP32)
      L = sum(P_unnorm, dim=-1)       [s1_size, 1]        (FP32, softmax 分母)
      O = (P_unnorm / L) @ V          [s1_size, head_dim] (FP32 → BF16)

    Args:
        q:        [s1_size, head_dim] — Q 切片
        k, v:     [s2_size, head_dim] — KV 切片
        scale:    attention scale factor
    Returns:
        l_val: [s1_size, 1], dtype=float32
        m:     [s1_size, 1], dtype=float32
        o:     [s1_size, head_dim], dtype=bfloat16
    """
    scores = torch.matmul(q.float(), k.float().T) * scale
    m = scores.max(dim=-1, keepdim=True)[0]
    p = torch.exp(scores - m)
    l_val = p.sum(dim=-1, keepdim=True)
    o = torch.matmul(p / l_val, v.float())
    # L, M 保持 [sq, 1], 不再 expand
    return l_val, m, o.to(torch.bfloat16)


def run_test(batch_size=None, num_heads=None, s1_size=None,
             s2_size=None, dim=None, tile_config=None,
             q_seqlens=None, kv_seqlens=None):
    """
    运行单个测试用例: 构造输入 → 调用 kernel → 与 golden 对比。

    参数语义 (Q: s1_size, KV: s2_size):
      - batch_size: 批次数量              (默认 1)
      - num_heads:  注意力头数            (默认 NUM_HEADS=8)
      - s1_size:    Q 序列长度            (默认 320)
      - s2_size:    KV 序列长度           (默认 = s1_size, 自注意力时相等)
      - dim:        每个头的维度 head_dim (默认 HEAD_DIM=64)
      - tile_config: TileConfig 分块配置  (默认使用全局 S2_TILE)

    输入张量布局:
      - Q/O/dO/L/M/dQ: [batch_size * s1_size, hidden_dim]  — Q 侧
      - K/V/dK/dV:      [batch_size * s2_size, hidden_dim]  — KV 侧
      - actual_q:        [batch_size + 1] — Q seqlen 前缀累加,
                          s1_i = actual_q[i+1] - actual_q[i]
      - actual_kv:       [batch_size + 1] — KV seqlen 前缀累加,
                          s2_i = actual_kv[i+1] - actual_kv[i]

    数据类型严格对应 kernel 签名:
      ┌────────────┬──────────────────────────────────────────┬─────────────┐
      │ 张量       │ kernel 签名                              │ host dtype  │
      ├────────────┼──────────────────────────────────────────┼─────────────┤
      │ q/k/v/o/do │ pypto.Tensor([DYN, HIDDEN_DIM], DT_BF16) │ bfloat16    │
      │ l/m        │ pypto.Tensor([DYN, HIDDEN_DIM], DT_FP32) │ float32     │
      │ dq/dk/dv   │ pypto.Tensor([DYN, HIDDEN_DIM], DT_BF16) │ bfloat16    │
      │ actual_seq │ pypto.Tensor([DYN],            DT_INT32) │ int32       │
      └────────────┴──────────────────────────────────────────┴─────────────┘

    Kernel 内部 dtype 转换流程:
      1. O, dO: BF16 → cast → FP32 (计算 D = sum(O*dO))
      2. matmul 输出: 大部分 out_dtype=FP32，仅 dV 的 matmul 直接输出 BF16
      3. P, dS: FP32 → cast → BF16 (作为后续 matmul 的输入，减少带宽)
      4. dQ/dK 输出: FP32 → cast → BF16 后 assemble 写回

    Args:
        batch_size:  批次数量 (可选)
        num_heads:   注意力头数 (可选)
        s1_size:     Q 序列长度 (可选)
        s2_size:     KV 序列长度 (可选, 默认 = s1_size)
        dim:         每个头的维度 head_dim (可选)
        tile_config: TileConfig 分块配置 (可选)
        q_seqlens:   每个 batch 的 Q seqlen 列表 (可选, varlen 场景)
        kv_seqlens:  每个 batch 的 KV seqlen 列表 (可选, varlen 场景)
    Returns:
        passed: 是否通过精度校验
    """
    device_id = get_device_id()
    if device_id is None:
        return None

    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'

    # ---- 参数默认值，未传则使用全局常量 ----
    if batch_size is None:
        batch_size = 1
    if num_heads is None:
        num_heads = NUM_HEADS
    if s1_size is None:
        s1_size = 320
    if s2_size is None:
        # 默认自注意力: KV seqlen = Q seqlen
        s2_size = s1_size
    if dim is None:
        dim = HEAD_DIM
    if tile_config is None:
        tile_config = TileConfig()

    # varlen: 若显式传入 q_seqlens / kv_seqlens, 则覆盖 batch_size 与 s1_size/s2_size
    if q_seqlens is not None:
        batch_size = len(q_seqlens)
        s1_size = max(q_seqlens)
    if kv_seqlens is not None:
        assert len(kv_seqlens) == batch_size, "kv_seqlens must match batch_size"
        s2_size = max(kv_seqlens)

    # 派生常量 (全部从输入参数计算，不直接引用全局变量)
    hidden_dim = num_heads * dim
    scale = 1.0 / (dim ** 0.5)

    logging.info("=" * 60)
    logging.info(f"Test Case: batch={batch_size}, heads={num_heads}, "
                 f"Q:s1_size={s1_size}, KV:s2_size={s2_size}, dim={dim}")
    if q_seqlens is not None or kv_seqlens is not None:
        logging.info(f"  varlen: q_seqlens={q_seqlens}, kv_seqlens={kv_seqlens}")
    logging.info(f"  hidden_dim={hidden_dim}, scale={scale:.6f}, "
                 f"s2_tile={tile_config.s2_tile}")
    logging.info("=" * 60)

    # ---- 构造输入张量，dtype 严格匹配 kernel 签名 ----
    # Q: [batch * s1_size, num_heads, dim], KV: [batch * s2_size, num_heads, dim]
    # kernel 签名为三维 [DYNAMIC, N, D]
    q, k, v, actual_q, actual_kv, q_seqlens, kv_seqlens = create_inputs(
        batch_size, s1_size, s2_size, num_heads, dim, device,
        q_seqlens=q_seqlens, kv_seqlens=kv_seqlens)

    # 从 actual_q / actual_kv (cumsum) 中派生 host 侧 offset/seqlen
    # 描述： actual_q / actual_kv shape=[batch_size + 1], offset_i = actual_q[i], seq_i = actual_q[i+1] - actual_q[i]
    q_cumsum = actual_q.cpu().tolist()
    kv_cumsum = actual_kv.cpu().tolist()
    total_q = q_cumsum[-1]
    total_kv = kv_cumsum[-1]

    # 注意: 此处使用与 create_inputs 不同的种子，避免 do_t 与 q 数值完全相同，
    # 否则会掩盖 dQ/dK/dV 计算路径中与 dO 相关的错误。
    torch.manual_seed(2026)
    # dO: Q 侧, shape=[batch * s1_size, num_heads, dim], dtype=BF16
    do_t = torch.randn(total_q, num_heads, dim, dtype=torch.bfloat16, device=device) * 0.01

    # L, M: Q 侧, shape=[batch * s1_size, num_heads, 1], dtype=FP32
    l_out = torch.empty(total_q, num_heads, 1, dtype=torch.float32, device=device)
    m_out = torch.empty(total_q, num_heads, 1, dtype=torch.float32, device=device)
    # O: Q 侧, shape=[batch * s1_size, num_heads, dim], dtype=BF16
    o_out = torch.empty(total_q, num_heads, dim, dtype=torch.bfloat16, device=device)

    # 预计算每个 (batch, head) 的 L, M, O
    # 三维张量按 [seq, head, dim] 索引: tensor[q_off:q_off+sq, h, :]
    # offset/seqlen 从 actual_q / actual_kv (cumsum) 派生, 支持每 batch 不等长
    for b in range(batch_size):
        q_off = q_cumsum[b]
        kv_off = kv_cumsum[b]
        sq = q_cumsum[b + 1] - q_off      # Q seqlen for this batch
        skv = kv_cumsum[b + 1] - kv_off   # KV seqlen for this batch
        for h in range(num_heads):
            l_h, m_h, o_h = compute_l_m_o(
                q[q_off: q_off + sq, h, :],
                k[kv_off: kv_off + skv, h, :],
                v[kv_off: kv_off + skv, h, :],
                scale)
            # l_h, m_h: [sq, 1], o_h: [sq, dim]
            l_out[q_off: q_off + sq, h, :] = l_h
            m_out[q_off: q_off + sq, h, :] = m_h
            o_out[q_off: q_off + sq, h, :] = o_h

    # dQ: Q 侧, shape=[batch * s1_size, hidden_dim] (二维), dtype=BF16
    dq_out = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16, device=device)
    # dK, dV: KV 侧, shape=[batch * s2_size, hidden_dim] (二维), dtype=BF16
    dk_out = torch.empty(total_kv, hidden_dim, dtype=torch.bfloat16, device=device)
    dv_out = torch.empty(total_kv, hidden_dim, dtype=torch.bfloat16, device=device)

    # ---- Golden 计算: 先完整计算所有 batch/head 的 golden dQ/dK/dV ----
    # golden dQ/dK/dV 与 kernel 输出同 shape: [total, hidden_dim], dtype=BF16
    dq_golden = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16, device=device)
    dk_golden = torch.empty(total_kv, hidden_dim, dtype=torch.bfloat16, device=device)
    dv_golden = torch.empty(total_kv, hidden_dim, dtype=torch.bfloat16, device=device)

    for b in range(batch_size):
        q_off = q_cumsum[b]
        kv_off = kv_cumsum[b]
        sq = q_cumsum[b + 1] - q_off
        skv = kv_cumsum[b + 1] - kv_off
        for h in range(num_heads):
            h_off = h * dim
            dq_g, dk_g, dv_g = attention_backward_golden(
                q[q_off: q_off + sq, h, :],
                k[kv_off: kv_off + skv, h, :],
                v[kv_off: kv_off + skv, h, :],
                o_out[q_off: q_off + sq, h, :],
                do_t[q_off: q_off + sq, h, :],
                scale)
            # golden 返回 [seq, dim] BF16, 写入对应 [total, hidden_dim] 位置
            dq_golden[q_off: q_off + sq, h_off: h_off + dim] = dq_g
            dk_golden[kv_off: kv_off + skv, h_off: h_off + dim] = dk_g
            dv_golden[kv_off: kv_off + skv, h_off: h_off + dim] = dv_g

    # ---- 调用 kernel ----
    logging.info("  Running kernel...")
    import time
    # 路由决策基于 max(q_seqlens) (varlen) 或 s1_size (uniform)
    max_s1 = max(q_seqlens)
    if max_s1 > 320:
        # perf run
        start_time = time.time()
        flash_attention_mha_grad_kernel_long_seq(
            q, k, v, o_out, do_t, l_out, m_out,
            dq_out, dk_out, dv_out,
            actual_q, actual_kv)
        elapsed = time.time() - start_time
        logging.info(f"  Kernel time: {elapsed * 1000:.2f} ms")
    else:
        # perf run
        start_time = time.time()
        flash_attention_varlen_backward_kernel_small_seq(
            q, k, v, o_out, do_t, l_out, m_out,
            dq_out, dk_out, dv_out,
            actual_q, actual_kv)
        elapsed = time.time() - start_time
        logging.info(f"  Kernel time: {elapsed * 1000:.2f} ms")
    # ---- 精度校验: kernel 输出 vs golden 输出, 使用 numpy assert_allclose ----
    rtol = 0.0078125  # 1/128, 约 BF16 精度
    atol = 0.0001

    passed = True
    for name, npu_tensor, golden_tensor in [
        ("dQ", dq_out, dq_golden),
        ("dK", dk_out, dk_golden),
        ("dV", dv_out, dv_golden),
    ]:
        npu_np = npu_tensor.float().cpu().numpy()
        golden_np = golden_tensor.float().cpu().numpy()
        max_diff = np.abs(npu_np - golden_np).max()
        try:
            assert_allclose(npu_np, golden_np, rtol=rtol, atol=atol)
            logging.info(f"  {name}: PASSED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
        except AssertionError as e:
            logging.info(f"  {name}: FAILED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
            logging.info(f"    {e}")
            passed = False

    logging.info(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


########################################################################
# 独立测试用例
#
# 每个 test_XX 函数定义一组独立的测试规格。
# run_test 支持可选参数:
#   batch_size, num_heads, s1_size(Q seqlen), s2_size(KV seqlen), dim, tile_config
# 不传的参数使用全局默认值。
# 在 main() 中选择要执行的用例，注释/取消注释即可切换。
########################################################################


@pytest.mark.soc("950")
def test_01():
    """ 用例规格信息：batch=8, heads=8, s1=320, s2=320, dim=64"""
    return run_test(batch_size=8, num_heads=8, s1_size=320, s2_size=320, dim=64)


@pytest.mark.skip(reason="large test case")
def test_02():
    """ 用例规格信息：batch=1, heads=8, s1=4096, s2=4096, dim=128"""
    return run_test(batch_size=1, num_heads=8, s1_size=4096, s2_size=4096, dim=128)


@pytest.mark.soc("950")
def test_03():
    """ 用例规格信息：batch=8, heads=16, s1=32, s2=32, dim=32"""
    return run_test(batch_size=8, num_heads=16, s1_size=32, s2_size=32, dim=32)


@pytest.mark.soc("950")
def test_04():
    """ 用例规格信息：batch=8, heads=16, s1=64, s2=64, dim=32"""
    return run_test(batch_size=8, num_heads=16, s1_size=64, s2_size=64, dim=32)


@pytest.mark.soc("950")
def test_05():
    """ 用例规格信息：batch=8, heads=8, s1=32, s2=32, dim=64"""
    return run_test(batch_size=8, num_heads=8, s1_size=32, s2_size=32, dim=64)


@pytest.mark.skip("950")
def test_06():
    """ 用例规格信息：batch=8, heads=4, s1=64, s2=64, dim=128"""
    return run_test(batch_size=8, num_heads=4, s1_size=64, s2_size=64, dim=128)


@pytest.mark.skip(reason="large test case")
def test_07_varlen_small_seq():
    """ 用例规格信息：batch=4, heads=8, q_seqlens=[64,128,192,256], kv_seqlens=[64,128,192,256], dim=64 """
    return run_test(num_heads=8, dim=64,
                    q_seqlens=[64, 128, 192, 256],
                    kv_seqlens=[64, 128, 192, 256])


@pytest.mark.soc("950")
def test_08_varlen_long_seq():
    """ 用例规格信息: batch=2, heads=8, q_seqlens=[384,512], kv_seqlens=[384,512], dim=64 """
    return run_test(num_heads=8, dim=64,
                    q_seqlens=[384, 512],
                    kv_seqlens=[384, 512])


@pytest.mark.soc("950")
def test_09_varlen_cross_attn():
    """ 用例规格信息：batch=3, heads=8, q_seqlens=[128,64,192], kv_seqlens=[96,256,128], dim=64 """
    return run_test(num_heads=8, dim=64,
                    q_seqlens=[128, 64, 192],
                    kv_seqlens=[96, 256, 128])


def main():
    """
    主入口: 在下方列表中选择要运行的测试用例。
    注释/取消注释即可控制执行哪些用例。
    """
    logging.info("Flash Attention Backward (3-loop, KV tiling)")

    # ---- 选择要运行的测试用例 (注释/取消注释即可) ----
    test_funcs = [
        test_01,                     # batch=8, heads=8, s1=320, s2=320, dim=64
        test_02,                     # batch=1, heads=8, s1=4096, s2=4096, dim=128
        test_03,                     # batch=8, heads=16, s1=32, s2=32, dim=32
        test_04,                     # batch=8, heads=16, s1=64, s2=64, dim=32
        test_05,                     # batch=8, heads=8, s1=32, s2=32, dim=64
        test_06,                     # batch=8, heads=4, s1=64, s2=64, dim=128
        test_07_varlen_small_seq,    # varlen, small_seq: q/kv=[64,128,192,256]
        test_08_varlen_long_seq,     # varlen, long_seq:  q/kv=[384,512]
        test_09_varlen_cross_attn,   # varlen cross-attn: q=[128,64,192], kv=[96,256,128]
    ]

    results = []
    for fn in test_funcs:
        try:
            passed = fn()
            results.append((fn.__name__, fn.__doc__, passed))
        except Exception as e:
            logging.info(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((fn.__name__, fn.__doc__, False))

    # ---- 汇总结果 ----
    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info("-" * 60)
    all_passed = True
    for name, desc, passed in results:
        status = "PASSED" if passed else "FAILED"
        logging.info(f"  {name}: {desc}  => {status}")
        if not passed:
            all_passed = False
    logging.info(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
