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

import collections
import logging
from dataclasses import dataclass

import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch
import torch_npu

from flash_attention_mha_grad_impl import flash_attention_mha_grad_kernel_impl, \
    FlashAttentionGradTileShapeConfig


logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
# KV 序列维度的分块大小 (全局配置常量)
S1_TILE = 1024
S2_TILE = 1024

MhaGradInputs = collections.namedtuple("MhaGradInputs", ["q", "k", "v", "actual_q", "actual_kv", "q_seqlens", "kv_seqlens"])
AttentionBackwardOutput = collections.namedtuple("AttentionBackwardOutput", ["dq", "dk", "dv"])


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
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    # K/V 张量: shape=[sum(kv_seqlens), num_heads, head_dim], dtype=BF16
    k = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1

    # 前缀累加构造 actual_q / actual_kv (shape=[batch_size + 1])
    q_cumsum = [0]
    for sq in q_seqlens:
        q_cumsum.append(q_cumsum[-1] + sq)
    actual_q = torch.tensor(q_cumsum, dtype=torch.int32, device=device)
    kv_cumsum = [0]
    for skv in kv_seqlens:
        kv_cumsum.append(kv_cumsum[-1] + skv)
    actual_kv = torch.tensor(kv_cumsum, dtype=torch.int32, device=device)
    return MhaGradInputs(q=q, k=k, v=v, actual_q=actual_q, actual_kv=actual_kv, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens)


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
    dk = dk_fp32.to(torch.float32)

    #  计算流：---- dV = p_half^T(BF16) @ dO(BF16) → BF16 ----
    #  计算流：kernel: matmul(p_half, doi, out_dtype=BF16, a_trans=True) 即 p_half^T @ dO
    #  计算流：p_half: [s1, s2], dO: [s1, D] → p_half^T: [s2, s1] @ dO: [s1, D] → [s2, D]
    # 注: 模拟 BF16 matmul — 先 FP32 计算再 cast BF16
    dv = torch.matmul(p_half.float().T, do_t.float()).to(torch.float32)

    #  计算流：---- dQ = ds_half(BF16) @ K(BF16) → FP32 * scale → BF16 ----
    #  计算流：kernel: matmul(ds_half, ki_tile, out_dtype=FP32) → mul(scale) → 累加(FP32) → cast(BF16)
    dq_fp32 = torch.matmul(ds_half.float(), k.float()) * scale
    dq = dq_fp32.to(torch.float32)

    return AttentionBackwardOutput(dq, dk, dv)


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


def _resolve_params(batch_size, num_heads, s1_size, s2_size, dim, q_seqlens, kv_seqlens):
    """Resolve and normalize test parameters with defaults."""
    if batch_size is None:
        batch_size = 1
    if num_heads is None:
        num_heads = NUM_HEADS
    if s1_size is None:
        s1_size = 320
    if s2_size is None:
        s2_size = s1_size
    if dim is None:
        dim = HEAD_DIM
    if q_seqlens is not None:
        batch_size = len(q_seqlens)
        s1_size = max(q_seqlens)
    if kv_seqlens is not None:
        assert len(kv_seqlens) == batch_size, "kv_seqlens must match batch_size"
        s2_size = max(kv_seqlens)
    return batch_size, num_heads, s1_size, s2_size, dim, q_seqlens, kv_seqlens


def _default_tile_config():
    """Create default FlashAttentionGradTileShapeConfig."""
    return FlashAttentionGradTileShapeConfig(
        s1_tile=S2_TILE,
        s2_tile=S2_TILE,
        c_tile=[[256, 512], [128, 256], [128, 512]],
        v_tile_s=[64, 256],
        v_tile_d=[64, 256],
    )


def _setup_device():
    """Set up NPU device from environment. Returns (device_str, device_id) or (None, None)."""
    device_id = get_device_id()
    if device_id is None:
        return None, None
    torch.npu.set_device(device_id)
    return f'npu:{device_id}', device_id


def _log_case(batch_size, num_heads, s1_size, s2_size, dim, hidden_dim, scale,
              tile_config, q_seqlens, kv_seqlens):
    """Log test case configuration."""
    logging.info("=" * 60)
    logging.info(f"Test Case: batch={batch_size}, heads={num_heads}, "
                 f"Q:s1_size={s1_size}, KV:s2_size={s2_size}, dim={dim}")
    if q_seqlens is not None or kv_seqlens is not None:
        logging.info(f"  varlen: q_seqlens={q_seqlens}, kv_seqlens={kv_seqlens}")
    logging.info(f"  hidden_dim={hidden_dim}, scale={scale:.6f}, "
                 f"s2_tile={tile_config.s2_tile}")
    logging.info("=" * 60)


def _precompute_l_m_o(batch_size, num_heads, q, k, v, q_cumsum, kv_cumsum,
                       l_out, m_out, o_out, scale):
    """Precompute L, M, O for all batch/head slices."""
    for b in range(batch_size):
        q_off = q_cumsum[b]
        kv_off = kv_cumsum[b]
        sq = q_cumsum[b + 1] - q_off
        skv = kv_cumsum[b + 1] - kv_off
        for h in range(num_heads):
            l_h, m_h, o_h = compute_l_m_o(
                q[q_off: q_off + sq, h, :],
                k[kv_off: kv_off + skv, h, :],
                v[kv_off: kv_off + skv, h, :],
                scale)
            l_out[q_off: q_off + sq, h, :] = l_h
            m_out[q_off: q_off + sq, h, :] = m_h
            o_out[q_off: q_off + sq, h, :] = o_h


def _compute_golden(batch_size, num_heads, dim, q, k, v, o_out, do_t,
                     q_cumsum, kv_cumsum, total_q, total_kv, device, scale):
    """Compute golden dQ/dK/dV for all batch/head slices."""
    dq_golden = torch.empty(total_q, HIDDEN_DIM, dtype=torch.float32, device=device)
    dk_golden = torch.empty(total_kv, HIDDEN_DIM, dtype=torch.float32, device=device)
    dv_golden = torch.empty(total_kv, HIDDEN_DIM, dtype=torch.float32, device=device)
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
            dq_golden[q_off: q_off + sq, h_off: h_off + dim] = dq_g
            dk_golden[kv_off: kv_off + skv, h_off: h_off + dim] = dk_g
            dv_golden[kv_off: kv_off + skv, h_off: h_off + dim] = dv_g
    return dq_golden, dk_golden, dv_golden


def _run_kernel(q, k, v, o_out, do_t, l_out, m_out, dq_out, dk_out, dv_out,
                actual_q, actual_kv, ws, ws_dq, tile_config):
    """Run the kernel and return elapsed time."""
    logging.info("  Running kernel...")
    import time
    start_time = time.time()
    flash_attention_mha_grad_kernel_impl(
        q, k, v, o_out, do_t, l_out, m_out,
        dq_out, dk_out, dv_out,
        actual_q, actual_kv,
        ws, ws_dq, tile_config)
    elapsed = time.time() - start_time
    logging.info(f"  Kernel time: {elapsed * 1000:.2f} ms")


def _verify_precision(dq_out, dk_out, dv_out, dq_golden, dk_golden, dv_golden):
    """Verify kernel outputs against golden with BF16 tolerance."""
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


def _prepare_grad_inputs_and_outputs(batch_size, num_heads, dim, s2_tile, device,
                                      q, k, v, actual_q, actual_kv, do_t,
                                      l_out, m_out, o_out):
    """Setup grad output tensors and golden reference."""
    total_q = do_t.shape[0]
    total_kv = k.shape[0]
    scale = 1.0 / (dim ** 0.5)
    dq_out = torch.zeros(total_q, HIDDEN_DIM, dtype=torch.float32, device=device)
    dk_out = torch.zeros(total_kv, HIDDEN_DIM, dtype=torch.float32, device=device)
    dv_out = torch.zeros(total_kv, HIDDEN_DIM, dtype=torch.float32, device=device)
    ws_rows = num_heads * s2_tile
    ws = torch.zeros(ws_rows, s2_tile, dtype=torch.float32, device=device)
    ws_dq = torch.zeros(ws_rows, dim, dtype=torch.float32, device=device)
    dq_golden, dk_golden, dv_golden = _compute_golden(
        batch_size, num_heads, dim, q, k, v, o_out, do_t,
        [0] + [q.shape[0]] if batch_size == 1 else list(range(batch_size + 1)),
        [0] + [k.shape[0]] if batch_size == 1 else list(range(batch_size + 1)),
        total_q, total_kv, device, scale)
    return dq_out, dk_out, dv_out, ws, ws_dq, dq_golden, dk_golden, dv_golden


def _run_test_setup_and_compute(device, batch_size, num_heads, s1_size, s2_size, dim,
                                 tile_config, q_seqlens, kv_seqlens):
    """Setup tensors and run kernel computation, returning outputs and goldens."""
    hidden_dim = num_heads * dim
    scale = 1.0 / (dim ** 0.5)
    _log_case(batch_size, num_heads, s1_size, s2_size, dim, hidden_dim, scale,
              tile_config, q_seqlens, kv_seqlens)
    torch.manual_seed(2026)
    inputs = create_inputs(batch_size, s1_size, s2_size, num_heads, dim, device,
                           q_seqlens=q_seqlens, kv_seqlens=kv_seqlens)
    q, k, v = inputs.q, inputs.k, inputs.v
    actual_q, actual_kv = inputs.actual_q, inputs.actual_kv
    q_seqlens, kv_seqlens = inputs.q_seqlens, inputs.kv_seqlens
    q_cumsum = actual_q.cpu().tolist()
    kv_cumsum = actual_kv.cpu().tolist()
    total_q = q_cumsum[-1]
    total_kv = kv_cumsum[-1]
    do_t = torch.randn(total_q, num_heads, dim, dtype=torch.bfloat16, device=device) * 0.1
    l_out = torch.empty(total_q, num_heads, 1, dtype=torch.float32, device=device)
    m_out = torch.empty(total_q, num_heads, 1, dtype=torch.float32, device=device)
    o_out = torch.empty(total_q, num_heads, dim, dtype=torch.bfloat16, device=device)
    _precompute_l_m_o(batch_size, num_heads, q, k, v, q_cumsum, kv_cumsum, l_out, m_out, o_out, scale)
    dq_out = torch.zeros(total_q, HIDDEN_DIM, dtype=torch.float32, device=device)
    dk_out = torch.zeros(total_kv, HIDDEN_DIM, dtype=torch.float32, device=device)
    dv_out = torch.zeros(total_kv, HIDDEN_DIM, dtype=torch.float32, device=device)
    s2_tile = tile_config.s2_tile
    ws_rows = num_heads * s2_tile
    ws = torch.zeros(ws_rows, s2_tile, dtype=torch.float32, device=device)
    ws_dq = torch.zeros(ws_rows, dim, dtype=torch.float32, device=device)
    dq_golden, dk_golden, dv_golden = _compute_golden(
        batch_size, num_heads, dim, q, k, v, o_out, do_t,
        q_cumsum, kv_cumsum, total_q, total_kv, device, scale)
    _run_kernel(q, k, v, o_out, do_t, l_out, m_out, dq_out, dk_out, dv_out,
                actual_q, actual_kv, ws, ws_dq, tile_config)
    return dq_out, dk_out, dv_out, dq_golden, dk_golden, dv_golden


def run_test(batch_size=None, num_heads=None, s1_size=None,
             s2_size=None, dim=None, tile_config=None,
             q_seqlens=None, kv_seqlens=None):
    """运行单个测试用例：构造输入 → 调用 kernel → 与 golden 对比。"""
    device, device_id = _setup_device()
    if device is None:
        return None
    batch_size, num_heads, s1_size, s2_size, dim, q_seqlens, kv_seqlens = \
        _resolve_params(batch_size, num_heads, s1_size, s2_size, dim, q_seqlens, kv_seqlens)
    if tile_config is None:
        tile_config = _default_tile_config()

    dq_out, dk_out, dv_out, dq_golden, dk_golden, dv_golden = \
        _run_test_setup_and_compute(device, batch_size, num_heads, s1_size, s2_size, dim,
                                     tile_config, q_seqlens, kv_seqlens)
    return _verify_precision(dq_out, dk_out, dv_out, dq_golden, dk_golden, dv_golden)


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
