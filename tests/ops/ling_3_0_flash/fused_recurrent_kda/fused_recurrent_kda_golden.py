# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO fused_recurrent_kda golden reference implementation.

Golden 说明：
  - 本 golden 直接复用参考测试文件中的 ``naive_recurrent_kda``（纯 PyTorch 逐 token
    递推）逻辑，位于
    ``vllm-ascend-main/tests/e2e/.../test_fused_recurrent_kda_npu.py`` L33-70。
  - 计算在 NPU 上执行（torch + torch_npu）；torch_npu 未安装时直接报错引导安装，
    仅无 NPU 硬件（device_count()==0）时回退 CPU。
  - 状态布局：朴素 [B,H,K,V]（fp32 累积）。核内布局为 [S,H,V,K]（转置），
    impl/test 在比对时对核输出做 ``.transpose(-1,-2)`` 还原到朴素布局再与本 golden
    比较（与参考测试 L154/L188/L248/L283 一致）。本 golden 全程使用朴素布局，
    不含任何 ``.T``/``.t()`` 转置 hack。
  - 导出 ``fused_recurrent_kda_golden()`` 供 ``test_fused_recurrent_kda.py`` 调用。
  - 不依赖 pypto。

置信度：⭐⭐⭐⭐⭐（直接复用参考测试的逐 token 递推实现，算法与参考 1:1 对齐）。
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

_DEVICE = None
_HAS_NPU = False

try:
    import torch_npu  # noqa: F401
    _HAS_NPU = torch.npu.is_available() and torch.npu.device_count() > 0
except ImportError:
    raise ImportError(
        "torch_npu is not installed. Please install it first:\n"
        "  pip install torch_npu\n"
        "Or use the pypto-environment-setup skill to set up the full NPU environment."
    )


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if not _HAS_NPU:
            _DEVICE = torch.device("cpu")
        else:
            device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
            torch.npu.set_device(device_id)
            _DEVICE = torch.device(f"npu:{device_id}")
    return _DEVICE


# ─────────────────────────────────────────────
# L2norm（与参考测试 reference_l2norm L27-30 完全一致，eps=1e-6）
# ─────────────────────────────────────────────

def reference_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 归一化沿最后一维 D。

    全程在 fp32 下计算并返回 fp32，不回退到输入 dtype，避免中间 bf16/fp16
    量化引入与 impl（核内 l2norm 全程 fp32）不一致的精度损失。

    Args:
        x: [..., D] 任意前缀，最后一维 D 做归一化。
        eps: 数值稳定项，固定 1e-6（与参考测试一致）。

    Returns:
        fp32 张量，shape 与 x 相同。
    """
    x = x.to(torch.float32)  # [., D] fp32
    # sum(x*x, -1, keepdim) -> [., 1]；rsqrt -> [., 1]；广播乘 -> [., D]
    return x * torch.rsqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


# ─────────────────────────────────────────────
# Golden 参考实现（NPU torch，逐 token 递推）
# ─────────────────────────────────────────────

def fused_recurrent_kda_golden(
    q: torch.Tensor,                        # [B, T, H, K] 输入 dtype
    k: torch.Tensor,                        # [B, T, H, K] 输入 dtype
    v: torch.Tensor,                        # [B, T, H, V] 输入 dtype (V=K=D)
    g: torch.Tensor,                        # [B, T, H, K] 输入 dtype (≤0, logsigmoid)
    beta: torch.Tensor,                     # [B, T, H] 输入 dtype (0..1, sigmoid)
    scale: Optional[float] = None,             # None -> K**-0.5
    initial_state: Optional[torch.Tensor] = None,  # [S, H, K, V] fp32 朴素布局
    inplace_final_state: bool = True,       # 原地更新 initial_state
    use_qk_l2norm_in_kernel: bool = True,  # 核内 q/k L2norm
    cu_seqlens: Optional[torch.Tensor] = None,  # [N+1] int64 变长累积边界
    ssm_state_indices: Optional[torch.Tensor] = None,  # [T] int64 inplace slot (1D) or [N, max_tokens] (2D, spec decode)
    num_accepted_tokens: Optional[torch.Tensor] = None,  # [N] int32 spec decode: accepted token count per seq
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fused_recurrent_kda 的 PyTorch 逐 token 递推参考实现。

    完整复用参考测试 ``naive_recurrent_kda`` 的递推数学语义，并扩展支持：
      - varlen（cu_seqlens）：按参考测试 L148-166 的 per-sequence 循环，每个序列
        从自己的 initial_state slot 起始。
      - inplace decode（ssm_state_indices）：每个序列读 ssm_state_indices[seq_idx]
        指向的 slot；slot<=0 为 NULL slot（跳过该序列，不读不写，保持为 0），
        与核 L100-108/L139-144 一致。递推完成后把终态写回该 slot。
      - 投机解码（num_accepted_tokens + 2D ssm_state_indices）：每个序列从
        ssm_state_indices[seq_idx, num_accepted_tokens[seq_idx]-1] 读初始状态，
        逐 token 把终态写到 ssm_state_indices[seq_idx, t_offset]。
      - use_qk_l2norm_in_kernel：递推前对 q、k 做 reference_l2norm（eps=1e-6）。
      - inplace_final_state=False 时总返回 per-token 状态 ht（与 Triton 语义一致）。

    数学（逐 token i，朴素布局 [B,H,K,V]）：
        S = S * exp(g_i)                                  # 门控衰减 [B,H,K,V]
        S = S + outer(b_i*k_i, v_i - (k_i·S))            # KDA delta 更新
        o_i = q_i · S                                     # 输出 [B,H,V]

    状态全程 fp32 累积；o 返回输入 dtype，ht 返回 fp32。

    Args:
        q, k: [B, T, H, K] 输入 dtype (K=V=D=128)。
        v: [B, T, H, V] 输入 dtype。
        g: [B, T, H, K] 输入 dtype (logsigmoid, ≤0)。
        beta: [B, T, H] 输入 dtype (sigmoid, 0..1)。
        scale: query 缩放因子；None 时取 K**-0.5。
        initial_state: [S, H, K, V] fp32 朴素布局状态缓冲。
            非 inplace varlen 时 S=T，seq i 读 [cu_seqlens[i]]；
            inplace decode 时 S=max_slots，slot 由 ssm_state_indices 给出。
        inplace_final_state: True 时原地更新 initial_state（返回 initial_state）；
            False 时总返回 per-token 状态 ht（非 None）。
        use_qk_l2norm_in_kernel: True 时递推前对 q、k 做 L2norm。
        cu_seqlens: [N+1] int64 变长累积边界（cu_seqlens[0]=0）。
        ssm_state_indices: 1D [T] 或 2D [N, max_tokens] int64 inplace slot（>0 有效，0=NULL）。
            1D 用于连续批处理 decode（每序列单 token）；
            2D 用于投机解码（每序列多 token，各 token 独立 slot）。
        num_accepted_tokens: [N] int32 投机解码：每序列被接受的 token 数（≥1）。
            非 None 时启用投机解码路径，从 ssm_state_indices[seq, nat-1] 读初始状态。

    Returns:
        (o, ht):
          o: [B, T, H, V] 输入 dtype。
          ht: [T, H, K, V] fp32 朴素布局（非 inplace 时总返回），
              或 initial_state 的引用（inplace 时，与 Triton 一致）。
    """
    device = _get_device()
    # 迁移到目标 device
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    g = g.to(device)
    beta = beta.to(device)
    if initial_state is not None:
        initial_state = initial_state.to(device)
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device)
    if ssm_state_indices is not None:
        ssm_state_indices = ssm_state_indices.to(device)
    if num_accepted_tokens is not None:
        num_accepted_tokens = num_accepted_tokens.to(device)

    dtype = v.dtype  # 原始输入 dtype，o 返回此 dtype
    B, T, H, K = q.shape  # [B, T, H, K]
    V = v.shape[-1]  # V = K = D
    if scale is None:
        scale = K ** -0.5  # 默认缩放

    # 可选 L2norm（递推前，对整个 q、k 做；按 token 独立，整块做等价于分块做）
    if use_qk_l2norm_in_kernel:
        q = reference_l2norm(q)  # [B, T, H, K] 原 dtype
        k = reference_l2norm(k)  # [B, T, H, K] 原 dtype

    # 全部 cast 到 fp32 做递推累积（与 naive_recurrent_kda L52 一致）
    q = q.to(torch.float) * scale  # [B, T, H, K] fp32，先 l2norm 再 scale（与核 L119-122 一致）
    k = k.to(torch.float)          # [B, T, H, K] fp32
    v = v.to(torch.float)          # [B, T, H, V] fp32
    g = g.to(torch.float)          # [B, T, H, K] fp32
    beta = beta.to(torch.float)    # [B, T, H] fp32

    o = torch.zeros_like(v)  # [B, T, H, V] fp32

    ht = None
    if not inplace_final_state:
        ht = torch.zeros(T, H, K, V, dtype=torch.float, device=device)  # [T, H, K, V] fp32 朴素

    if cu_seqlens is not None:
        # ── varlen 路径（B=1，参考测试 L148-166 的 per-sequence 循环）──
        cu_list = cu_seqlens.cpu().tolist()  # [N+1]
        ssm_list = ssm_state_indices.cpu().tolist() if (ssm_state_indices is not None) else None
        nat_list = num_accepted_tokens.cpu().tolist() if (num_accepted_tokens is not None) else None
        N = len(cu_list) - 1  # 序列数
        for i in range(N):
            s = cu_list[i]      # 起始 token（含）
            e = cu_list[i + 1]  # 结束 token（不含）

            if inplace_final_state:
                if nat_list is not None:
                    # 投机解码：从 ssm_state_indices[i, num_accepted-1] 读初始状态
                    init_t = nat_list[i] - 1
                    slot = int(ssm_list[i][init_t]) if isinstance(ssm_list[i], list) else int(ssm_list[i])
                else:
                    # 连续批处理 decode：slot 由 ssm_state_indices[i] 给出
                    if ssm_list is None:
                        raise ValueError(
                            "ssm_state_indices must be provided when inplace_final_state=True"
                        )
                    slot = int(ssm_list[i])  # seq i 的 slot（1D 或 2D 第一列）
                # slot<=0 -> NULL slot，跳过该序列（核 L106-107）
                if slot <= 0:
                    continue  # NULL slot：不读不写，保持原值
                if initial_state is None:
                    raise ValueError(
                        "initial_state must be provided when inplace_final_state=True"
                    )
                # 读 slot：朴素 [H, K, V] -> [1, H, K, V]
                S = initial_state[slot].to(torch.float).unsqueeze(0)  # [1, H, K, V] fp32
            else:
                # 非 inplace：seq i 读 initial_state[s]（核 L110 h0[bos]）
                if initial_state is not None:
                    S = initial_state[s].to(torch.float).unsqueeze(0)  # [1, H, K, V] fp32
                else:
                    S = torch.zeros(1, H, K, V, dtype=torch.float, device=device)  # [1, H, K, V] fp32

            # 逐 token 递推（顺序依赖，不可并行；与 naive_recurrent_kda L59-67 一致）
            for t in range(s, e):
                q_i = q[:, t]   # [1, H, K] fp32
                k_i = k[:, t]   # [1, H, K] fp32
                v_i = v[:, t]   # [1, H, V] fp32
                g_i = g[:, t]   # [1, H, K] fp32
                b_i = beta[:, t]  # [1, H] fp32

                # 门控衰减：S * exp(g_i[..., None])
                #   g_i[..., None] -> [1, H, K, 1]；S -> [1, H, K, V]；广播乘 -> [1, H, K, V]
                S = S * g_i[..., None].exp()  # [1, H, K, V] fp32

                # KDA delta 更新：
                #   kS = (k_i[..., None] * S).sum(-2)  # 收缩 K -> [1, H, V]
                #     k_i[..., None] -> [1, H, K, 1]；* S -> [1, H, K, V]；sum(-2) -> [1, H, V]
                kS = (k_i[..., None] * S).sum(-2)  # [1, H, V] fp32
                delta = v_i - kS  # 新息 [1, H, V] fp32
                #   outer(b_i*k_i, delta) via einsum("bhk,bhv->bhkv") -> [1, H, K, V]
                #     b_i[..., None] -> [1, H, 1]；* k_i -> [1, H, K]
                S = S + torch.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, delta)  # [1, H, K, V] fp32

                # 输出：o_i = q_i · S，收缩 K -> [1, H, V]
                o[:, t] = torch.einsum("bhk,bhkv->bhv", q_i, S)  # [1, H, V] fp32

                # 投机解码：逐 token 写终态到 ssm_state_indices[i, t_offset]
                if inplace_final_state and nat_list is not None:
                    t_offset = t - s  # token index within sequence
                    final_slot = int(ssm_list[i][t_offset]) if isinstance(ssm_list[i], list) else int(ssm_list[i])
                    if final_slot > 0:
                        initial_state[final_slot] = S.squeeze(0).to(initial_state.dtype)
                # 非 inplace：逐 token 写状态 ht[t]（核 L146 写 ht[bos+i_t]）
                elif not inplace_final_state:
                    ht[t] = S.squeeze(0)  # [H, K, V] fp32 朴素

            # 序列递推结束，写终态（非投机解码的 inplace 路径）
            if inplace_final_state and nat_list is None:
                # 原地写回 slot（核 L142-144，cast 到 state dtype）
                initial_state[slot] = S.squeeze(0).to(initial_state.dtype)  # [H, K, V] -> slot
    else:
        # ── 非 varlen 路径（B 个独立序列，各 T token；即原 naive_recurrent_kda）──
        # S: [B, H, K, V] fp32
        S = torch.zeros(B, H, K, V, dtype=torch.float, device=device)  # [B, H, K, V] fp32
        if initial_state is not None:
            S = S + initial_state.to(torch.float)  # [B, H, K, V] fp32
        for i in range(T):
            q_i = q[:, i]   # [B, H, K] fp32
            k_i = k[:, i]   # [B, H, K] fp32
            v_i = v[:, i]   # [B, H, V] fp32
            g_i = g[:, i]   # [B, H, K] fp32
            b_i = beta[:, i]  # [B, H] fp32
            S = S * g_i[..., None].exp()  # [B, H, K, V] fp32
            kS = (k_i[..., None] * S).sum(-2)  # [B, H, V] fp32
            delta = v_i - kS  # [B, H, V] fp32
            S = S + torch.einsum("bhk,bhv->bhkv", b_i[..., None] * k_i, delta)  # [B, H, K, V] fp32
            o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)  # [B, H, V] fp32
            if not inplace_final_state:
                # 非 varlen 时 ht 按批铺平：token i 的状态写到 ht[i]（每批同形）
                ht[i] = S  # [B, H, K, V] fp32（若 B>1 形状不符；非 varlen 仅 B=1 时用）
        if inplace_final_state:
            if initial_state is None:
                raise ValueError("initial_state must be provided when inplace_final_state=True")
            initial_state[:] = S.to(initial_state.dtype)  # [B, H, K, V] 原地

    o_out = o.to(dtype)  # [B, T, H, V] 返回输入 dtype
    if inplace_final_state:
        return o_out, initial_state  # inplace：返回已更新的 initial_state（与 Triton 一致）
    return o_out, ht  # [B,T,H,V] dtype, [T,H,K,V] fp32 朴素


# ==========================================
# 输入构造（供 profiling --factory 与 _validate 共用）
# ==========================================

def _make_inputs(device):
    """构造 P0 典型输入，供验证和性能采集共用。

    覆盖 SPEC §11 的典型配置 + p0_shapes [[1,64,32,128]]。每 case 返回
    (args_list, kwargs_dict)。

    args_list 仅含 golden 签名前 5 个连续 tensor 位置参数 [q, k, v, g, beta]
    （签名第 6 位是 scalar ``scale``，非 tensor，故 initial_state 等后续 tensor
    必须走 kwargs，避免位置/关键字冲突）。其余参数全部放入 kwargs_dict：
      scale, initial_state, use_qk_l2norm_in_kernel,
      cu_seqlens, ssm_state_indices, inplace_final_state

    语义约束：
      - g 必须 ≤0（logsigmoid）；beta 必须 0..1（sigmoid）；cu_seqlens 单调非降首 0；
        inplace 模式 state_buf[0]=0（NULL slot），ssm_state_indices=[1..N]。
      - 状态类 tensor 用 fp32（与 SPEC 一致）。

    Returns:
        [(case_name, args_list, kwargs_dict), ...]
    """
    cases = []

    def _gen_qkvg(H, D, T, dtype, device):
        # q,k,v [1,T,H,D]；g logsigmoid(≤0) [1,T,H,D]；beta sigmoid(0..1) [1,T,H]
        q = torch.randn(1, T, H, D, dtype=dtype, device=device)
        k = torch.randn(1, T, H, D, dtype=dtype, device=device)
        v = torch.randn(1, T, H, D, dtype=dtype, device=device)
        g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float32, device=device)).to(dtype)
        beta = torch.rand(1, T, H, dtype=dtype, device=device).sigmoid()
        return q, k, v, g, beta

    # ── case 1: perf_p0_T64（SPEC p0_shapes [1,64,32,128]，单序列多 token）──
    H, D, T = 32, 128, 64
    cu_seqlens = torch.LongTensor([0, T]).to(device)
    q, k, v, g, beta = _gen_qkvg(H, D, T, torch.float16, device)
    initial_state = torch.randn(T, H, D, D, dtype=torch.float32, device=device)  # [T,H,K,V] 朴素 fp32
    cases.append(("perf_p0_T64", [q, k, v, g, beta], {
        "scale": None, "initial_state": initial_state, "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": True,
        "inplace_final_state": False,
    }))

    # ── case 2: func_p0_multitok（cu=[0,16]）──
    H, D = 32, 128
    cu_seqlens = torch.LongTensor([0, 16]).to(device)
    T = 16
    q, k, v, g, beta = _gen_qkvg(H, D, T, torch.float16, device)
    initial_state = torch.randn(T, H, D, D, dtype=torch.float32, device=device)
    cases.append(("func_p0_multitok", [q, k, v, g, beta], {
        "scale": None, "initial_state": initial_state, "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": True,
        "inplace_final_state": False,
    }))

    # ── case 3: func_p0_multi_seq（cu=[0,8,24]）──
    H, D = 32, 128
    cu_seqlens = torch.LongTensor([0, 8, 24]).to(device)
    T = 24
    q, k, v, g, beta = _gen_qkvg(H, D, T, torch.float16, device)
    initial_state = torch.randn(T, H, D, D, dtype=torch.float32, device=device)
    cases.append(("func_p0_multi_seq", [q, k, v, g, beta], {
        "scale": None, "initial_state": initial_state, "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": True,
        "inplace_final_state": False,
    }))

    # ── case 4: func_p0_inplace_decode（N=4，ssm_state_indices=[1,2,3,4]，slot0=NULL）──
    H, D, N = 32, 128, 4
    T = N  # 每 seq 1 token
    cu_seqlens = torch.LongTensor(list(range(N + 1))).to(device)  # [0,1,2,3,4]
    ssm_state_indices = torch.arange(1, N + 1, dtype=torch.long, device=device)  # [1,2,3,4]
    q, k, v, g, beta = _gen_qkvg(H, D, T, torch.float16, device)
    max_slots = N + 1
    state_buf = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=device)  # [5,H,K,V]
    state_buf[0] = 0  # NULL slot
    cases.append(("func_p0_inplace_decode", [q, k, v, g, beta], {
        "scale": None, "initial_state": state_buf, "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "use_qk_l2norm_in_kernel": True,
        "inplace_final_state": True,
    }))

    # ── case 5: func_p1_fp32（cu=[0,16]，fp32 隔离算法误差）──
    H, D = 32, 128
    cu_seqlens = torch.LongTensor([0, 16]).to(device)
    T = 16
    q, k, v, g, beta = _gen_qkvg(H, D, T, torch.float32, device)
    initial_state = torch.randn(T, H, D, D, dtype=torch.float32, device=device)
    cases.append(("func_p1_fp32", [q, k, v, g, beta], {
        "scale": None, "initial_state": initial_state, "cu_seqlens": cu_seqlens,
        "use_qk_l2norm_in_kernel": True,
        "inplace_final_state": False,
    }))

    return cases


# ==========================================
# 验证
# ==========================================

def _naive_recurrent_kda_reference(
    q, k, v, g, beta, scale=None, initial_state=None,
):
    """参考测试中的 naive_recurrent_kda 原样副本（仅用于 allclose 交叉校验）。

    与 test_fused_recurrent_kda_npu.py L33-70 1:1 一致，用于验证 golden 的非
    varlen / 非 inplace 路径数值正确性。总返回 per-token 最终状态。
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale
    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)
    return o.to(dtype), S


def _validate():
    """运行时验证：shape 一致性、NaN 守卫、数值有限、inplace NULL slot 完整性、
    与参考测试 naive_recurrent_kda 的 allclose（≥3 shape cases）。"""
    device = _get_device()
    print("=" * 60)
    print("fused_recurrent_kda_golden 验证报告")
    print("=" * 60)
    print(f"Device: {device}")

    torch.manual_seed(42)
    all_pass = True

    def _ok(cond, msg):
        nonlocal all_pass
        status = "PASS" if cond else "FAIL"
        if not cond:
            all_pass = False
        print(f"  [{status}] {msg}")

    # ── 1. 典型 case：shape / NaN / 有限 / 与参考 allclose ──
    print("\n[典型 case 验证 + allclose 交叉校验（≥3 shapes）]")
    test_shapes = [
        # (H, D, cu_seqlens, dtype)
        (32, 128, [0, 1, 2, 3, 4], torch.float16),     # decode 多序列
        (32, 128, [0, 16], torch.float16),             # 单序列多 token
        (32, 128, [0, 8, 24], torch.float16),          # 变长多序列
        (32, 128, [0, 1, 2, 3, 4], torch.bfloat16),    # bf16
        (32, 128, [0, 16], torch.float32),             # fp32 隔离
        (64, 128, [0, 1, 2, 3, 4], torch.float16),     # H=64
    ]
    allclose_pass_count = 0
    for H, D, cu, dtype in test_shapes:
        T = cu[-1]
        N = len(cu) - 1
        cu_t = torch.LongTensor(cu).to(device)
        q = torch.randn(1, T, H, D, dtype=dtype, device=device)
        k = torch.randn(1, T, H, D, dtype=dtype, device=device)
        v = torch.randn(1, T, H, D, dtype=dtype, device=device)
        g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float32, device=device)).to(dtype)
        beta = torch.rand(1, T, H, dtype=dtype, device=device).sigmoid()
        h0 = torch.randn(T, H, D, D, dtype=torch.float32, device=device)  # [T,H,K,V] 朴素

        # golden（朴素布局 initial_state / ht）
        o_g, ht_g = fused_recurrent_kda_golden(
            q, k, v, g, beta, scale=None, initial_state=h0,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_t, inplace_final_state=False,
        )

        # shape 检查
        _ok(o_g.shape == (1, T, H, D), f"o shape {tuple(o_g.shape)} == (1,{T},{H},{D})")
        _ok(ht_g.shape == (T, H, D, D), f"ht shape {tuple(ht_g.shape)} == ({T},{H},{D},{D})")
        # dtype 检查
        _ok(o_g.dtype == dtype, f"o dtype {o_g.dtype} == {dtype}")
        _ok(ht_g.dtype == torch.float32, f"ht dtype {ht_g.dtype} == float32")
        # NaN / 有限
        _ok(not torch.isnan(o_g).any(), f"o no NaN (cu={cu},{dtype})")
        _ok(torch.isfinite(o_g).all(), f"o finite (cu={cu},{dtype})")
        _ok(not torch.isnan(ht_g).any(), f"ht no NaN (cu={cu},{dtype})")
        _ok(torch.isfinite(ht_g).all(), f"ht finite (cu={cu},{dtype})")

        # 与参考测试的 per-seq naive 流程 allclose（参考测试 L148-166）
        ref_outputs = []
        ref_states = []
        for i in range(N):
            s, e = cu[i], cu[i + 1]
            q_i = reference_l2norm(q[:, s:e].contiguous())
            k_i = reference_l2norm(k[:, s:e].contiguous())
            init_state_i = h0[s].unsqueeze(0)  # 朴素布局，无需转置
            o_i, ht_i = _naive_recurrent_kda_reference(
                q_i, k_i, v[:, s:e], g[:, s:e], beta[:, s:e],
                initial_state=init_state_i,
            )
            ref_outputs.append(o_i)
            ref_states.append(ht_i)
        ref_o = torch.cat(ref_outputs, dim=1)

        # o allclose（golden 与参考同算法同 ops，应精确匹配）
        o_close = torch.allclose(o_g, ref_o, atol=1e-6, rtol=1e-5)
        _ok(o_close, f"o allclose(golden, ref) cu={cu} {dtype} "
                     f"max_diff={ (o_g - ref_o).abs().max().item():.3e}")
        # ht 逐序列 allclose：golden ht[e-1] vs ref_states[i]
        ht_all_close = True
        for i in range(N):
            e = cu[i + 1]
            if not torch.allclose(ht_g[e - 1], ref_states[i].squeeze(0), atol=1e-6, rtol=1e-5):
                ht_all_close = False
                break
        _ok(ht_all_close, f"ht allclose(golden, ref) cu={cu} {dtype}")
        if o_close and ht_all_close:
            allclose_pass_count += 1

    _ok(allclose_pass_count >= 3, f"allclose 通过 shape 数 {allclose_pass_count} >= 3")

    # ── 2. inplace decode 验证：NULL slot 完整性 + slot 写入 ──
    print("\n[inplace decode 验证]")
    H, D, N = 32, 128, 4
    T = N
    cu = list(range(N + 1))
    cu_t = torch.LongTensor(cu).to(device)
    ssm_idx = torch.arange(1, N + 1, dtype=torch.long, device=device)
    q = torch.randn(1, T, H, D, dtype=torch.float16, device=device)
    k = torch.randn(1, T, H, D, dtype=torch.float16, device=device)
    v = torch.randn(1, T, H, D, dtype=torch.float16, device=device)
    g = F.logsigmoid(torch.randn(1, T, H, D, dtype=torch.float32, device=device)).to(torch.float16)
    beta = torch.rand(1, T, H, dtype=torch.float16, device=device).sigmoid()
    max_slots = N + 1
    state_buf = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=device)
    state_buf[0] = 0  # NULL slot
    state_buf_orig = state_buf.clone()

    o_ip, ht_ip = fused_recurrent_kda_golden(
        q, k, v, g, beta, scale=None, initial_state=state_buf,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_t, ssm_state_indices=ssm_idx, inplace_final_state=True,
    )
    _ok(o_ip.shape == (1, T, H, D), f"inplace o shape {tuple(o_ip.shape)}")
    _ok(ht_ip is not None, "inplace ht is not None (returns updated initial_state)")
    _ok(not torch.isnan(o_ip).any(), "inplace o no NaN")
    # NULL slot 完整性：slot 0 必须保持 0
    _ok(torch.all(state_buf[0] == 0), "NULL slot[0] untouched (==0)")
    # slot 0 与原始一致
    _ok(torch.equal(state_buf[0], state_buf_orig[0]), "NULL slot[0] == original")
    # 有效 slot 被修改（与参考 per-seq naive 比较）
    ref_states_ip = []
    for i in range(N):
        slot = i + 1
        q_i = reference_l2norm(q[:, i:i + 1].contiguous())
        k_i = reference_l2norm(k[:, i:i + 1].contiguous())
        init_state_i = state_buf_orig[slot].unsqueeze(0)  # 用原始 state_buf 读
        o_i, ht_i = _naive_recurrent_kda_reference(
            q_i, k_i, v[:, i:i + 1], g[:, i:i + 1], beta[:, i:i + 1],
            initial_state=init_state_i,
        )
        ref_states_ip.append((slot, ht_i.squeeze(0)))
    slots_ok = True
    for slot, ref_s in ref_states_ip:
        if not torch.allclose(state_buf[slot], ref_s, atol=1e-6, rtol=1e-5):
            slots_ok = False
            break
    _ok(slots_ok, "inplace slots[1..N] allclose(golden, ref)")

    # ── 3. 数值稳定性：极负 gate（状态清零）──
    print("\n[数值稳定性：极负 gate]")
    H, D, T = 32, 128, 2
    cu_t = torch.LongTensor([0, 2]).to(device)
    q = torch.randn(1, T, H, D, dtype=torch.float16, device=device)
    k = torch.randn(1, T, H, D, dtype=torch.float16, device=device)
    v = torch.randn(1, T, H, D, dtype=torch.float16, device=device)
    g = torch.full((1, T, H, D), -50.0, dtype=torch.float16, device=device)  # exp(-50)≈0
    beta = torch.rand(1, T, H, dtype=torch.float16, device=device).sigmoid()
    h0 = torch.randn(T, H, D, D, dtype=torch.float32, device=device)
    o_st, ht_st = fused_recurrent_kda_golden(
        q, k, v, g, beta, scale=None, initial_state=h0,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_t, inplace_final_state=False,
    )
    _ok(torch.isfinite(o_st).all(), "极负 gate: o finite")
    _ok(torch.isfinite(ht_st).all(), "极负 gate: ht finite")
    _ok(not torch.isnan(o_st).any(), "极负 gate: o no NaN")

    # ── 4. _make_inputs 烟雾测试（每个 case 能跑通且输出有限）──
    print("\n[_make_inputs 烟雾测试]")
    cases = _make_inputs(device)
    _ok(len(cases) == 5, f"_make_inputs 返回 {len(cases)} cases (期望 5)")
    for case_name, args, kwargs in cases:
        try:
            o_c, ht_c = fused_recurrent_kda_golden(*args, **kwargs)
            finite = torch.isfinite(o_c).all() and not torch.isnan(o_c).any()
            _ok(bool(finite), f"case '{case_name}': o finite, shape={tuple(o_c.shape)}")
        except Exception as exc:  # noqa: BLE001
            _ok(False, f"case '{case_name}' crashed: {exc}")

    print("\n" + "=" * 60)
    print(f"验证{'通过' if all_pass else '存在失败项'}")
    print("=" * 60)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    _validate()
