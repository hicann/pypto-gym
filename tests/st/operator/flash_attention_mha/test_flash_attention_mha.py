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
Flash Attention Forward with Dynamic Variable Length Sequences

语义约定:
  - Q 侧: s1_size (Q seqlen), 张量包括 Q/O/L/M
  - KV 侧: s2_size (KV seqlen), 张量包括 K/V
  - Q_TILE/K_TILE: 序列维度的分块大小 (将 s1/s2 切分为多个 tile 迭代)

4 loops: batch + head + q_tile + kv_tile.
Tiles Q and KV sequence dimensions by Q_TILE/K_TILE to reduce intermediate
attention matrix from [s1_size, s2_size] to [Q_TILE, K_TILE] per iteration.
O, L, M are accumulated across kv tiles (online softmax algorithm).
"""


import sys
import os
_p = os.path.dirname(__file__)
while not os.path.isdir(os.path.join(_p, 'src')):
    _p = os.path.dirname(_p)
sys.path.insert(0, os.path.join(_p, 'src'))
sys.path.insert(0, os.path.join(_p, 'src', 'pypto_gym', 'ops', 'pypto_tile'))

import collections
import logging
from dataclasses import dataclass

import torch
import torch_npu

import numpy as np
import pytest

from flash_attention_mha_impl import flash_attention_varlen_forward_kernel


logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)


NUM_HEADS = 8
HEAD_DIM = 64
HIDDEN_DIM = NUM_HEADS * HEAD_DIM

Q_TILE = 320
K_TILE = 320

MhaInputs = collections.namedtuple(
    "MhaInputs", ["q", "k", "v", "cu_seqlens_q", "cu_seqlens_k",
                   "q_seqlens", "kv_seqlens"])
AttentionForwardOutput = collections.namedtuple("AttentionForwardOutput", ["o", "m", "l"])


@dataclass
class TileConfig:
    """
    分块配置结构体，封装 kernel 所需的 tile 参数。

    将 tile 信息集中管理，避免在 run_test 中直接引用全局变量。

    Attributes:
        q_tile: Q 序列维度的分块大小 (kernel 中 Q_TILE_SIZE,
                 用于将 Q seqlen 切分为多个 tile 迭代)
        k_tile: KV 序列维度的分块大小 (kernel 中 K_TILE_SIZE,
                 用于将 KV seqlen 切分为多个 tile 迭代)
    """
    q_tile: int = Q_TILE
    k_tile: int = K_TILE


def get_device_id():
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        logging.info("Please set TILE_FWK_DEVICE_ID before running:")
        logging.info("  export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ['TILE_FWK_DEVICE_ID'])
    except ValueError:
        return None


def create_inputs(batch_size, s1_size, s2_size, num_heads, head_dim, device):
    """
    创建 varlen 布局的输入张量（三维布局）。

    布局说明 (Q: s1_size, KV: s2_size):
      - Q/K/V: 三维 [total_seq, num_heads, head_dim]
      - Kernel从shape自动推导 num_heads 和 head_dim

    数据类型严格对应 kernel 签名:
      - q/k/v:       BF16  (kernel: pypto.DT_BF16, 三维)
      - cu_seqlens:  INT32 (kernel: pypto.DT_INT32)

    Args:
        batch_size: 批次数量
        s1_size:    Q 序列长度 (每个 batch 的 Q seqlen)
        s2_size:    KV 序列长度 (每个 batch 的 KV seqlen)
        num_heads:  注意力头数
        head_dim:   每个头的维度
        device:     计算设备
    Returns:
        q:          shape=[total_q, num_heads, head_dim], dtype=bfloat16 (三维)
        k, v:       shape=[total_kv, num_heads, head_dim], dtype=bfloat16 (三维)
        cu_seqlens_q:   shape=[batch_size + 1], dtype=int32
                        cu_seqlens_q[i]=前 i 个 batch 的累积 Q seqlen
        cu_seqlens_k:   shape=[batch_size + 1], dtype=int32
                        cu_seqlens_k[i]=前 i 个 batch 的累积 KV seqlen
        q_seqlens:  list of Q seqlens per batch
        kv_seqlens: list of KV seqlens per batch
    """
    q_seqlens = [s1_size] * batch_size
    kv_seqlens = [s2_size] * batch_size
    total_q = sum(q_seqlens)
    total_kv = sum(kv_seqlens)

    torch.manual_seed(42)
    # 三维布局: [total_seq, num_heads, head_dim]
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device) + 0.5
    k = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) + 0.5
    v = torch.randn(total_kv, num_heads, head_dim, dtype=torch.bfloat16, device=device) + 0.5

    cu_seqlens_q = torch.tensor([0] + list(np.cumsum(q_seqlens)), dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0] + list(np.cumsum(kv_seqlens)), dtype=torch.int32, device=device)

    return MhaInputs(
        q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        q_seqlens=q_seqlens, kv_seqlens=kv_seqlens)


@dataclass
class ComputeKvTileInputs:
    k_tile_idx: int
    k_tile_count: int
    q_tile_len: int
    k_f: torch.Tensor
    v_f: torch.Tensor
    q_tile_view: torch.Tensor
    scale: float
    k_tile: int
    s2_size: int
    oi_update: torch.Tensor
    li_update: torch.Tensor
    mi_update: torch.Tensor
    o_out: torch.Tensor
    q_tile_start: int
    q_tile_end: int
    l_out: torch.Tensor
    m_out: torch.Tensor


def _compute_kv_tile(inputs: ComputeKvTileInputs):
    """Compute one kv-tile within the online-softmax golden (updates accumulators in-place)."""
    k_tile_start = inputs.k_tile_idx * inputs.k_tile
    k_tile_end = min(k_tile_start + inputs.k_tile, inputs.s2_size)
    k_tile_len = k_tile_end - k_tile_start
    k_tile_view = inputs.k_f[k_tile_start:k_tile_end, :]
    v_tile_view = inputs.v_f[k_tile_start:k_tile_end, :].to(torch.bfloat16)

    scores = torch.matmul(inputs.q_tile_view, k_tile_view.T) * inputs.scale
    mij = scores.amax(dim=-1, keepdim=True)
    s_shifted = scores - mij
    pij = torch.exp(s_shifted)
    lij = pij.sum(dim=-1, keepdim=True)
    p_bf16 = pij.to(torch.bfloat16)
    oij = torch.matmul(p_bf16, v_tile_view)

    if inputs.k_tile_idx == 0:
        if inputs.k_tile_idx == inputs.k_tile_count - 1:
            pij_div = pij / lij
            pij_bf16 = pij_div.to(torch.bfloat16)
            out_bf16 = torch.matmul(pij_bf16, v_tile_view)
            inputs.o_out[inputs.q_tile_start:inputs.q_tile_end, :] = out_bf16[:inputs.q_tile_len, :]
            inputs.l_out[inputs.q_tile_start:inputs.q_tile_end, :] = lij[:inputs.q_tile_len, :]
            inputs.m_out[inputs.q_tile_start:inputs.q_tile_end, :] = mij[:inputs.q_tile_len, :]
        else:
            inputs.oi_update[:inputs.q_tile_len, :] = oij[:inputs.q_tile_len, :]
            inputs.li_update[:inputs.q_tile_len, :] = lij[:inputs.q_tile_len, :]
            inputs.mi_update[:inputs.q_tile_len, :] = mij[:inputs.q_tile_len, :]
    else:
        mi = inputs.mi_update[:inputs.q_tile_len, :]
        li = inputs.li_update[:inputs.q_tile_len, :]
        oi = inputs.oi_update[:inputs.q_tile_len, :]
        mi_new = torch.maximum(mi, mij[:inputs.q_tile_len, :])
        t1 = torch.exp(mi - mi_new)
        t2 = torch.exp(mij[:inputs.q_tile_len, :] - mi_new)
        li_new = t1 * li + t2 * lij[:inputs.q_tile_len, :]
        oi_tmp = t1 * oi + t2 * oij[:inputs.q_tile_len, :]
        if inputs.k_tile_idx == inputs.k_tile_count - 1:
            out_fp32 = oi_tmp / li_new
            out_bf16 = out_fp32.to(torch.bfloat16)
            inputs.o_out[inputs.q_tile_start:inputs.q_tile_end, :] = out_bf16[:inputs.q_tile_len, :]
            inputs.l_out[inputs.q_tile_start:inputs.q_tile_end, :] = li_new[:inputs.q_tile_len, :]
            inputs.m_out[inputs.q_tile_start:inputs.q_tile_end, :] = mi_new[:inputs.q_tile_len, :]
        else:
            inputs.oi_update[:inputs.q_tile_len, :] = oi_tmp
            inputs.li_update[:inputs.q_tile_len, :] = li_new
            inputs.mi_update[:inputs.q_tile_len, :] = mi_new


def attention_forward_golden(q, k, v, scale):
    """
    Golden reference: Flash Attention (online softmax)算法实现。

    Q: [s1_size, head_dim],  KV: [s2_size, head_dim]
    输入 q/k/v 均为 BF16。

    采用 online softmax 算法，分块计算避免存储完整的 [s1, s2] attention matrix。
    通过增量更新 O, M, L，实现内存高效的 attention 计算。

    Args:
        q:     [s1_size, head_dim] BF16 — Q 切片
        k, v:  [s2_size, head_dim] BF16 — KV 切片
        scale: attention scale factor (1/sqrt(head_dim))
    Returns:
        o:     [s1_size, head_dim] BF16 — 输出 O
        m:     [s1_size, 1] FP32 — softmax 最大值 M
        l:     [s1_size, 1] FP32 — softmax 分母 L
    """
    s1_size, head_dim = q.shape
    s2_size = k.shape[0]

    q_f = q.cpu().to(torch.float32)
    k_f = k.cpu().to(torch.float32)
    v_f = v.cpu().to(torch.float32)

    q_tile = Q_TILE
    k_tile = K_TILE

    q_tile_count = (s1_size + q_tile - 1) // q_tile
    k_tile_count = (s2_size + k_tile - 1) // k_tile

    o_out = torch.zeros(s1_size, head_dim, dtype=torch.bfloat16)
    l_out = torch.zeros(s1_size, 1, dtype=torch.float32)
    m_out = torch.zeros(s1_size, 1, dtype=torch.float32)

    for q_tile_idx in range(q_tile_count):
        q_tile_start = q_tile_idx * q_tile
        q_tile_end = min(q_tile_start + q_tile, s1_size)
        q_tile_len = q_tile_end - q_tile_start

        q_tile_view = q_f[q_tile_start:q_tile_end, :]

        oi_update = torch.zeros(q_tile, head_dim, dtype=torch.float32)
        li_update = torch.zeros(q_tile, 1, dtype=torch.float32)
        mi_update = torch.full((q_tile, 1), float('-inf'), dtype=torch.float32)

        for k_tile_idx in range(k_tile_count):
            _compute_kv_tile(ComputeKvTileInputs(
                k_tile_idx=k_tile_idx, k_tile_count=k_tile_count, q_tile_len=q_tile_len,
                k_f=k_f, v_f=v_f, q_tile_view=q_tile_view, scale=scale,
                k_tile=k_tile, s2_size=s2_size,
                oi_update=oi_update, li_update=li_update, mi_update=mi_update,
                o_out=o_out, q_tile_start=q_tile_start, q_tile_end=q_tile_end,
                l_out=l_out, m_out=m_out))

    return AttentionForwardOutput(o_out, m_out, l_out)


@dataclass
class ComputeGoldenOutputsInputs:
    batch_size: int
    num_heads: int
    dim: int
    hidden_dim: int
    s1_size: int
    s2_size: int
    scale: float
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_seqlens: list
    kv_seqlens: list


def _compute_golden_outputs(inputs: ComputeGoldenOutputsInputs):
    """Compute golden O/M/L by iterating over batches and heads."""
    total_q = inputs.batch_size * inputs.s1_size
    out_golden = torch.empty(total_q, inputs.hidden_dim, dtype=torch.bfloat16)
    l_golden = torch.empty(total_q, inputs.num_heads, dtype=torch.float32)
    m_golden = torch.empty(total_q, inputs.num_heads, dtype=torch.float32)
    q_off, k_off = 0, 0
    for b in range(inputs.batch_size):
        sq, sk = inputs.q_seqlens[b], inputs.kv_seqlens[b]
        for h in range(inputs.num_heads):
            h_off = h * inputs.dim
            q_h = inputs.q[q_off:q_off + sq, h, :]
            k_h = inputs.k[k_off:k_off + sk, h, :]
            v_h = inputs.v[k_off:k_off + sk, h, :]
            golden_o, golden_m, golden_l = attention_forward_golden(q_h, k_h, v_h, inputs.scale)
            out_golden[q_off:q_off + sq, h_off:h_off + inputs.dim] = golden_o
            m_golden[q_off:q_off + sq, h:h + 1] = golden_m
            l_golden[q_off:q_off + sq, h:h + 1] = golden_l
        q_off += sq
        k_off += sk
    return out_golden, l_golden, m_golden


def _check_outputs(out_npu, out_golden, l_out_npu, l_golden, m_out_npu, m_golden):
    """Compare kernel outputs against golden and log results."""
    torch.set_printoptions(precision=6)
    passed = True
    for name, npu_tensor, golden_tensor, rtol, atol in [
        ("O", out_npu, out_golden, 0.0078125, 0.0001),
        ("L", l_out_npu, l_golden, 0.005, 0.000025),
        ("M", m_out_npu, m_golden, 0.005, 0.000025),
    ]:
        npu_np = npu_tensor.cpu().float().numpy()
        golden_np = golden_tensor.float().numpy()
        max_diff = np.abs(npu_np - golden_np).max()
        try:
            from tests.ops.utils.compare import compare
            compare(npu_tensor.cpu(), golden_tensor, name, atol=atol, rtol=rtol, max_error_count=10)
            logging.info(f"  {name}: PASSED (max_diff={max_diff:.6f}, rtol={rtol}, atol={atol})")
        except AssertionError as e:
            logging.info(f"  {name}: FAILED (max_diff={max_diff:.6f})")
            logging.info(f"    {e}")
            passed = False
    return passed


def _resolve_test_params(batch_size, num_heads, s1_size, s2_size, dim, tile_config):
    """Resolve defaults and return concrete test parameters + hidden_dim, scale, device."""
    device_id = get_device_id()
    if device_id is None:
        return None
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
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
    if tile_config is None:
        tile_config = TileConfig()
    hidden_dim = num_heads * dim
    scale = 1.0 / (dim ** 0.5)
    return (device_id, device, batch_size, num_heads, s1_size, s2_size,
            dim, tile_config, hidden_dim, scale)


def run_test(batch_size=None, num_heads=None, s1_size=None,
             s2_size=None, dim=None, tile_config=None):
    """运行单个测试用例: 构造输入 → 调用 kernel → 与 golden 对比。

    Args: batch_size, num_heads, s1_size, s2_size, dim, tile_config.
    Returns: passed (bool) or None if device unavailable.
    """
    resolved = _resolve_test_params(batch_size, num_heads, s1_size, s2_size, dim, tile_config)
    if resolved is None:
        return None
    (device_id, device, batch_size, num_heads, s1_size, s2_size,
     dim, tile_config, hidden_dim, scale) = resolved

    logging.info("=" * 60)
    logging.info(f"Test Case: batch={batch_size}, heads={num_heads}, "
                 f"Q:s1_size={s1_size}, KV:s2_size={s2_size}, dim={dim}")
    logging.info(f"  hidden_dim={hidden_dim}, scale={scale:.6f}")
    logging.info("=" * 60)

    inputs = create_inputs(batch_size, s1_size, s2_size, num_heads, dim, device)
    q, k, v = inputs.q, inputs.k, inputs.v
    cu_seqlens_q, cu_seqlens_k = inputs.cu_seqlens_q, inputs.cu_seqlens_k
    q_seqlens, kv_seqlens = inputs.q_seqlens, inputs.kv_seqlens
    total_q = batch_size * s1_size

    out_npu = torch.empty(total_q, hidden_dim, dtype=torch.bfloat16, device=device)
    l_out_npu = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)
    m_out_npu = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)

    out_golden, l_golden, m_golden = _compute_golden_outputs(
        ComputeGoldenOutputsInputs(
            batch_size=batch_size, num_heads=num_heads, dim=dim,
            hidden_dim=hidden_dim, s1_size=s1_size, s2_size=s2_size, scale=scale,
            q=q, k=k, v=v, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens))

    logging.info("  Running kernel...")
    flash_attention_varlen_forward_kernel(
        q, k, v, out_npu, l_out_npu, m_out_npu, cu_seqlens_q, cu_seqlens_k)

    passed = _check_outputs(out_npu, out_golden, l_out_npu, l_golden, m_out_npu, m_golden)
    logging.info(f"  {'PASSED' if passed else 'FAILED'}")
    logging.info("")
    return passed


@pytest.mark.soc("950")
def test_01():
    return run_test(batch_size=8, num_heads=8, s1_size=320, s2_size=320, dim=64)


@pytest.mark.soc("950")
def test_02():
    return run_test(batch_size=1, num_heads=8, s1_size=4096, s2_size=4096, dim=128)


@pytest.mark.soc("950")
def test_03():
    return run_test(batch_size=8, num_heads=16, s1_size=32, s2_size=32, dim=32)


@pytest.mark.soc("950")
def test_04():
    return run_test(batch_size=8, num_heads=16, s1_size=64, s2_size=64, dim=32)


@pytest.mark.soc("950")
def test_05():
    return run_test(batch_size=8, num_heads=8, s1_size=32, s2_size=32, dim=64)


@pytest.mark.soc("950")
def test_06():
    return run_test(batch_size=8, num_heads=4, s1_size=64, s2_size=64, dim=128)


def main():
    """
    主入口: 在下方列表中选择要运行的测试用例。
    注释/取消注释即可控制执行哪些用例。
    """
    logging.info("\n" + "=" * 60)
    logging.info("Flash Attention Forward (4-loop, Q+KV tiling)")
    logging.info("=" * 60 + "\n")

    test_funcs = [
        test_01,
        test_02,
        test_03,
        test_04,
        test_05,
        test_06,
    ]

    results = []
    for i, fn in enumerate(test_funcs):
        logging.info(f"[{i+1}/{len(test_funcs)}] {fn.__name__}: {fn.__doc__}")
        try:
            passed = fn()
            results.append((fn.__name__, fn.__doc__, passed))
        except Exception as e:
            logging.info(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((fn.__name__, fn.__doc__, False))
            logging.info("")

    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info("-" * 60)
    all_passed = True
    for name, desc, passed in results:
        status = "PASSED" if passed else "FAILED"
        logging.info(f"  {name}: {desc}  => {status}")
        if not passed:
            all_passed = False
    logging.info("-" * 60)
    logging.info(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()