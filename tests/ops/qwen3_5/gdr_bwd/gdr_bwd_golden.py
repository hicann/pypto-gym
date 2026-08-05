# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import NamedTuple

import torch

LN2 = math.log(2.0)
RCP_LN2 = 1.0 / LN2

# 16×16 是求逆的基础对角块尺寸（与 fla chunk_fwd.py 的 BC=16 一致）。
_BASE_BLOCK = 16


@dataclass
class BwdOptions:
    """反向 golden 的可选配置参数（``_bwd_golden_impl`` / ``_bwd_core`` 共用）。"""
    state_v_first: bool = False
    cu_seqlens: torch.LongTensor | None = None
    cp_context: object = None
    chunk_indices: torch.LongTensor | None = None
    use_gate_in_kernel: bool = False
    g_input: torch.Tensor | None = None
    a_log: torch.Tensor | None = None
    dt_bias: torch.Tensor | None = None
    chunk_size: int = 64
    use_given_a: bool = True


def _get_device() -> torch.device:
    """NPU 优先；torch_npu 不可用或无 NPU 硬件时回退 CPU（本 golden 是纯 torch，CPU 亦可跑）。"""
    try:
        import os
        import torch_npu
        if torch.npu.is_available() and torch.npu.device_count() > 0:
            return torch.device(f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}")
    except (ImportError, RuntimeError):
        pass
    return torch.device("cpu")


def _tp(x: torch.Tensor) -> torch.Tensor:
    """最后两维转置（禁止使用 .T / .t()，见 golden 规范 OL15）。"""
    return torch.transpose(x, -1, -2)


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp32 累加的 matmul（与 pypto.matmul 的 L0C FP32 累加路径对齐）。"""
    return torch.matmul(a.float(), b.float())


# =============================================================================
# 单位下三角矩阵求逆：16×16 对角块精确求逆 + block-doubling 合并 (16→32→64)
#
# ⚠️ 【D1/D7 之后，本节代码已不在反向主路径上】
#     本 golden 直接消费传入的 ``A``（前向存下的 ``(I+L)^{-1}``），反向不需要任何
#     矩阵求逆。保留本节的理由：
#       1. 它是 ORCHESTRATION_BRIEF **D5 兜底约束**的参考实现 ——「若未来任何环节
#          确需求逆（例如改回 kernel 内重算 A 的路径），必须按 16×16 对角块精确
#          求逆 + block-doubling 16→32→64 合并来做；禁止 solve_triangular /
#          torch.inverse 一把梭，禁止截断 Neumann 级数（会爆到 1e32）」。
#          未来的「重算 A」路径可直接复用这两个函数。
#       2. ``recompute_a_from_forward``（造 make_inputs 的合法 A、以及自检里
#          「改造前语义 vs 改造后语义」对拍）仍然依赖它。
#     其单测保留在 ``_validate()`` 的 [block-doubling 求逆 检查] 一节。
# =============================================================================


def _inv_unit_lower_diag_blocks(m_mat: torch.Tensor, bs: int) -> torch.Tensor:
    """对 ``M`` 的每个 ``bs×bs`` 对角块做**精确前代求逆**，非对角块保持 0。

    ``M`` 为单位下三角（对角恒为 1），块内 ``X = M_blk^{-1}`` 由前代递推得到::

        X[0, :] = e_0
        X[i, :] = e_i - Σ_{j<i} M[i, j] · X[j, :]

    这与 fla ``chunk_fwd.py:226-250`` 的 ``forward substitution on diagonal
    blocks`` 是同一算法（那边写成 ``-A`` 的形式，本处直接对 ``M = I + L`` 做）。

    Args:
        M: ``[*, bt, bt]`` fp32，单位下三角。
        bs: 对角块边长（16）。
    Returns:
        ``[*, bt, bt]`` fp32，块对角矩阵，第 p 个对角块 = ``M`` 第 p 个对角块的逆。
    """
    bt = m_mat.shape[-1]
    nb = bt // bs
    eye_b = torch.eye(bs, dtype=torch.float32, device=m_mat.device)

    blocks = []
    for p in range(nb):
        sl = slice(p * bs, (p + 1) * bs)
        m_b = m_mat[..., sl, sl].float()                              # [*, bs, bs]
        # X 为本地新建张量，对其 in-place 赋值不触碰任何入参。
        x_mat = eye_b.expand_as(m_b).clone()                          # [*, bs, bs]
        for i in range(1, bs):
            row = -_mm(m_b[..., i:i + 1, :i], x_mat[..., :i, :])      # [*, 1, bs]
            x_mat[..., i:i + 1, :] = row + eye_b[i:i + 1, :]
        blocks.append(x_mat)

    # 组装成块对角矩阵（out-of-place：先建零阵再 slice-assign，均为本地张量）。
    out = torch.zeros_like(m_mat, dtype=torch.float32)
    for p in range(nb):
        sl = slice(p * bs, (p + 1) * bs)
        out[..., sl, sl] = blocks[p]
    return out


def _invert_unit_lower_triangular(m_mat: torch.Tensor) -> torch.Tensor:
    """精确求 ``M^{-1}``，``M`` 为单位下三角 ``[*, bt, bt]``（fp32）。

    算法 = **16×16 对角块精确求逆 + 迭代 block-doubling 合并（16 → 32 → 64）**。

    合并一步：把 ``M`` 看作 2×2 分块（当前块尺寸 c，合并成 2c）::

        M = [[M11, 0  ],      M^{-1} = [[ x11,            0   ],
             [m21, M22]]                [ -x22 m21 x11,   x22 ]]

    其中 ``x11 = M11^{-1}``、``x22 = M22^{-1}`` 是上一层已算好的对角块逆。
    这与基准 ``GDR_impl.py:133-140`` 的 ``N ← N - dmask_lvl ⊙ (N @ M @ N)``
    在数学上完全等价——``-(N M N)`` 限制到新合并出来的左下子块上，恰好就是
    ``-x22 m21 x11``；只是这里从 16×16 起步（与 fla 的 BC=16 分块粒度对齐），
    而不是从 1×1 起步。log2 层递归 ⇒ **精确逆**，中间量恒有界。

    ⚠️ 硬约束：**不允许**用 ``torch.linalg.solve_triangular`` / ``torch.inverse``
    一把梭——pypto kernel 侧要照此分块顺序实现，golden 必须逐位可对齐。
    """
    bt = m_mat.shape[-1]
    if bt & (bt - 1) != 0:
        raise ValueError(f"chunk_size must be a power of two for block-doubling inverse, got {bt}.")

    bs = min(_BASE_BLOCK, bt)
    # 第 0 层：bs×bs 对角块精确求逆。
    x_mat = _inv_unit_lower_diag_blocks(m_mat, bs)

    # 逐层 doubling：bs → 2bs → 4bs → ... → bt。
    c = bs
    while c < bt:
        npair = bt // (2 * c)
        x_new = x_mat.clone()
        for p in range(npair):
            r0 = p * 2 * c                # 上半块（行/列）起点
            r1 = r0 + c                   # 下半块起点
            x11 = x_mat[..., r0:r1, r0:r1]                       # M11^{-1}
            x22 = x_mat[..., r1:r1 + c, r1:r1 + c]               # M22^{-1}
            m21 = m_mat[..., r1:r1 + c, r0:r1].float()           # 左下耦合块
            x_new[..., r1:r1 + c, r0:r1] = -_mm(_mm(x22, m21), x11)
        x_mat = x_new
        c *= 2
    return x_mat


# =============================================================================
# chunk 级前向重算（对应基准的 _gdr_chunk_forward）
# =============================================================================


@dataclass
class ForwardPrepConfig:
    """前向重算的参数包（_chunk_forward_prep 的 10 个入参）。"""
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    beta: torch.Tensor
    gc: torch.Tensor
    scale: float
    m_le: torch.Tensor
    m_lt: torch.Tensor
    a_c: torch.Tensor
    use_given_a: bool = True


def _chunk_forward_prep(config: ForwardPrepConfig):
    """重算每个 chunk 的 chunk-parallel 前向量（与 S 无关的部分），全 fp32。

    config 字段（BN = bh*num_chunks，即所有 (batch, head, chunk) 三元组展平）：
        q, k: ``[BN, bt, K]``；v: ``[BN, bt, V]``；beta: ``[BN, bt, 1]``；
        gc:   ``[BN, bt, 1]``（**自然对数**的 chunk 内前缀和）
        m_le: ``[bt, bt]`` 下三角含对角 0/1；m_lt: ``[bt, bt]`` 严格下三角 0/1。
        a_c:  ``[BN, bt, bt]`` fp32，**调用方传入的** ``(I+L)^{-1}``（pad 行已补单位行）。

    Args:
        use_given_a: **D7 的默认行为是 True** —— 直接用 ``a_c``，不重算。
            ``False`` 仅供自检里「改造前语义（内部 block-doubling 重算 A）」
            vs「改造后语义（吃传入 A）」的等价性对拍使用。
    """
    q, k, v, beta, gc, scale, m_le, m_lt, a_c, use_given_a = (
        config.q, config.k, config.v, config.beta, config.gc,
        config.scale, config.m_le, config.m_lt, config.a_c, config.use_given_a)
    gexp = torch.exp(gc)                                         # [BN,bt,1] e^{gc} ∈ (0,1]
    v_beta = v * beta                                            # [BN,bt,V]
    k_beta = k * beta                                            # [BN,bt,K]

    # l_mask，NS#3 mask-BEFORE-exp：先把严格上三角的 (gc_i - gc_j) > 0 置零再 exp。
    g_diff = gc - _tp(gc)                                        # [BN,bt,bt] (i - j)
    l_mask = torch.exp(g_diff * m_le) * m_le                     # [BN,bt,bt] 下三角含对角

    k_beta_g = k_beta * gexp                                     # [BN,bt,K]
    kkt = _mm(k_beta, _tp(k))                                    # [BN,bt,bt]

    if use_given_a:
        # 【D1/D7】直接吃前向存下的 A —— 反向主路径**无任何矩阵求逆**。
        a_inv = a_c.float()                                      # [BN,bt,bt]
    else:
        # 【仅自检用】改造前的语义：M = I - A0 = I + strict_lower(kkt ⊙ l_mask)，
        # 单位下三角，用 16×16 对角块 + block-doubling 精确求逆。
        m_lt = torch.eye(k.shape[-2], dtype=torch.float32,
                         device=k.device) + (kkt * l_mask) * m_lt   # [BN,bt,bt]
        a_inv = _invert_unit_lower_triangular(m_lt)              # [BN,bt,bt] 精确逆

    u = _mm(a_inv, v_beta)                                       # [BN,bt,V]
    w = _mm(a_inv, k_beta_g)                                     # [BN,bt,K]
    q_s = q * scale                                              # [BN,bt,K]
    qkt = _mm(q_s, _tp(k))                                       # [BN,bt,bt]
    attn = qkt * l_mask                                          # [BN,bt,bt]
    qg = q_s * gexp                                              # [BN,bt,K]

    g_last = gc[..., -1:, :]                                     # [BN,1,1]
    decay_k = torch.exp(g_last - gc)                             # [BN,bt,1] NS#4：参数恒 ≤ 0
    k_dec = k * decay_k                                          # [BN,bt,K]
    decay_s = torch.exp(g_last)                                  # [BN,1,1] e^{g_last}

    return dict(
        q_s=q_s, k=k, v=v, beta=beta, gc=gc, gexp=gexp,
        v_beta=v_beta, k_beta=k_beta, k_beta_g=k_beta_g,
        l_mask=l_mask, kkt=kkt, a_inv=a_inv, u=u, w=w,
        qkt=qkt, attn=attn, qg=qg,
        decay_k=decay_k, k_dec=k_dec, decay_s=decay_s,
    )


# =============================================================================
# gate-in-kernel 的链式反向（D4：逐元素形态，便于融进 pypto 主 kernel）
# =============================================================================


def _softplus_stable(x: torch.Tensor) -> torch.Tensor:
    """数值稳定的 softplus：``max(x, 0) + log1p(exp(-|x|))``。

    ⚠️ 禁止朴素 ``log(1+exp(x))``（大 x 溢出）；PyPTO 侧也无 ``softplus`` API，
    必须用本式（``maximum`` / ``abs`` / ``exp`` / ``log1p`` 四个逐元素 API 组合）。

    Args:
        x: 任意 shape fp32。
    Returns:
        同 shape fp32。
    """
    return torch.clamp(x, min=0.0) + torch.log1p(torch.exp(-torch.abs(x)))


def _gdn_gate_bwd_golden(g_input, a_log, dt_bias, dyg):
    """``gdn_gate_bwd`` 的 golden —— 对齐 ``custom/gated_delta_rule/gate.py:119-223``。

    前向 gate::

        x  = g_input + dt_bias                       (dt_bias 沿 B/T 广播)
        yg = -exp(a_log) · softplus(x)

    反向（``softplus' = sigmoid``；``∂yg/∂a_log = yg``；``∂yg/∂dt_bias = ∂yg/∂g_input``）::

        dg       = -exp(a_log) · (dyg ⊙ sigmoid(x))
        da_log   = Σ_{B, T} (dyg ⊙ yg)
        ddt_bias = Σ_{B, T} dg

    **【D4】本函数刻意写成逐元素 + 两次纯求和归约的形态**：除最后两个沿 (B, T) 的
    reduce 之外全部是 elementwise op，且不使用任何 host-only 技巧
    （无 python 分支依赖张量值、无 fancy indexing、无 in-place），
    可以直接逐行映射到 pypto 主 kernel 内的 vector 指令。

    Args:
        g_input: ``[B, T, hv]``，**raw** gate（未 softplus、未 cumsum）。
        a_log:   ``[hv]``。
        dt_bias: ``[hv]`` 或 ``None``。
        dyg:     ``[B, T, hv]`` fp32，上游对 **per-token gate** 的梯度
                 （即本文件反向主体产出、已做 chunk 内 reverse-cumsum 的 dg）。

    Returns:
        ``(dg, da_log, ddt_bias)``：
          * ``dg``       ``[B, T, hv]`` dtype = **``g_input.dtype``**（D14，对齐
            ``gate.py:219`` 的 ``dg.type_as(g)``；注意 ``g`` 在那里就是 ``g_input``）
          * ``da_log``   ``[hv]`` dtype = ``a_log.dtype``
          * ``ddt_bias`` ``[hv]`` dtype = ``dt_bias.dtype``；``dt_bias is None`` ⇒ ``None``（D12）
    """
    hv = g_input.shape[-1]
    gi = g_input.float()                                          # [B,T,hv]
    dyg_f = dyg.float()                                           # [B,T,hv]

    x = gi + dt_bias.float() if dt_bias is not None else gi       # [B,T,hv]
    neg_exp_a = -torch.exp(a_log.float())                          # [hv]
    sp = _softplus_stable(x)                                      # [B,T,hv]
    yg = neg_exp_a * sp                                            # [B,T,hv] 激活后的 gate
    sig = torch.sigmoid(x)                                        # [B,T,hv] = softplus'(x)

    dg_raw = neg_exp_a * (dyg_f * sig)                             # [B,T,hv] fp32
    da_log = torch.sum(dyg_f * yg, dim=(0, 1)).reshape(a_log.shape).to(a_log.dtype)   # [hv]

    # 【D10】严格对齐 fla：先把 dg cast 回 g_input.dtype（gate.py:219 的
    # ``dg.type_as(g)``），**再**沿 (B,T) 求和（gate.py:221）。bf16 g_input 下这与
    # 纯 fp32 求和结果不同 —— 目标是整网可原地替换，行为一致优先于「更准」。
    #
    # 📌 归约的**累加 dtype**（orchestrator 已裁决，Stage 3/5 不必再纠结）：
    #    正确语义 = 「先 cast 到 g_input.dtype → **fp32 累加器**累加 → cast 到目标 dtype」。
    #    依据：fla 对 **bf16** 的 dg 调 ``torch.sum``，而 torch 对 bf16 求和内部本就
    #    使用 fp32 累加器；``da``（da_log）更是明确先在 fp32 buffer 里跨 chunk 累加
    #    （``gate.py`` 的 ``da = a_log.new_empty(num_chunks, H, dtype=torch.float32)``），
    #    最后才 ``.sum(0).type_as(a_log)``。
    #    ⇒ 决定精度的是**入口的 cast**（D10），不是累加器；kernel 侧照此实现即可，
    #      **不要**为了"更准"跳过入口 cast，也**不要**把累加器降到 bf16。
    dg_cast = dg_raw.to(g_input.dtype)                            # [B,T,hv] g_input.dtype
    ddt_bias = (dg_cast.reshape(-1, hv).sum(0).to(dt_bias.dtype)
                if dt_bias is not None else None)                 # [hv] 或 None（D12）

    # 【D14】（原 D11 已作废）gate 路径下 dg **跟随 ``g_input.dtype``**，逐字对齐
    # fla ``gate.py:219`` 的 ``dg = dg.view_as(g).type_as(g)``（那里的 ``g`` 即
    # ``g_input``）。目标是把 triton bwd 原地替换掉、上游一行不改，行为一致优先。
    # 注意：``use_gate_in_kernel=False`` 的非 gate 路径，dg 仍恒 fp32
    # （fla 侧是 ``chunk_local_cumsum`` 的 fp32 输出）——两条路径 dtype 不同是**预期**。
    return dg_cast, da_log, ddt_bias


# =============================================================================
# 反向主体（等长 batch 路径；varlen 由公开入口逐序列调用本函数）
# =============================================================================


class BwdCoreOutput(NamedTuple):
    """_bwd_core 的 6 个返回值。"""
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    db: torch.Tensor
    dg: torch.Tensor
    dh0: torch.Tensor | None


def _bwd_core(q, k, v, g, beta, a_mat, scale, initial_state, do, dht,
              opts: BwdOptions):
    """等长 batch 的反向主体。

    Args:
        q/k ``[B, T, H, K]``；v/do ``[B, T, hv, V]``；g/beta ``[B, T, hv]``；
        A ``[B, T, hv, bt]``（= 前向存下的 ``(I+L)^{-1}``）；
        initial_state/dht ``[N=B, hv, K, V]``（``state_v_first`` 时 ``[N, hv, V, K]``）或 ``None``。
        opts: 见 :class:`BwdOptions`（``use_given_a`` / ``state_v_first`` / ``chunk_size``）。

    Returns:
        ``(dq, dk, dv, db, dg, dh0)``，均为 **fp32**（dtype 回 cast 由公开入口负责）：
          dq/dk ``[B, T, H, K]``；dv ``[B, T, hv, V]``；db/dg ``[B, T, hv]``；
          dh0 ``[N, hv, K, V]``（或 ``[N, hv, V, K]``），``initial_state is None`` ⇒ ``None``。
    """
    state_v_first = opts.state_v_first
    bt = opts.chunk_size
    use_given_a = opts.use_given_a
    batch, seq_len, n_heads, d_head = q.shape
    hv, d_value = v.shape[2], v.shape[3]
    rep = hv // n_heads

    device = q.device
    num_chunks = (seq_len + bt - 1) // bt                                      # chunk 数
    tp = num_chunks * bt                                                 # padding 后的 token 数
    pad = tp - seq_len
    bh = batch * hv

    # -------------------------------------------------- layout：→ [bh*num_chunks, bt, D]
    def _to_chunks(x, d_model, edge_pad=False):
        """``[B, T, hv, D]`` → ``[bh*num_chunks, bt, D]``。

        ``edge_pad=False``：尾部**零**填充（q/k/v/do/beta——pad token 不参与任何计算，
        beta=0 使其在 WY 中失活，k=0 使其不改变状态）。
        ``edge_pad=True``：尾部**边缘复制**（gc 专用——g_last 取 ``gc[bt-1]``，复制
        保证 pad 行给出的 g_last 仍是最后一个真实 token 的 gc，从而
        ``exp(g_last - gc)`` 对真实 token 正确、且指数参数仍恒 ≤ 0）。
        """
        x = x.float()
        if pad:
            tail = (x[:, -1:].expand(batch, pad, x.shape[2], d_model) if edge_pad
                    else x.new_zeros(batch, pad, x.shape[2], d_model))
            x = torch.cat([x, tail], dim=1)                      # [B,tp,hv,D]
        return x.permute(0, 2, 1, 3).reshape(-1, bt, d_model)          # [bh*num_chunks,bt,D]

    # q/k 只有 H 个头；GVA 时按组复制到 hv 个头，最后再把 dq/dk 求和归约回 H。
    q_e = q.float().repeat_interleave(rep, dim=2) if rep > 1 else q.float()   # [B,T,hv,K]
    k_e = k.float().repeat_interleave(rep, dim=2) if rep > 1 else k.float()   # [B,T,hv,K]

    q_c = _to_chunks(q_e, d_head)                                     # [BN,bt,K]
    k_c = _to_chunks(k_e, d_head)                                     # [BN,bt,K]
    v_c = _to_chunks(v, d_value)                                       # [BN,bt,V]
    do_c = _to_chunks(do, d_value)                                     # [BN,bt,V]
    beta_c = _to_chunks(beta.unsqueeze(-1), 1)                   # [BN,bt,1]

    # gate：fla 的 g 是 base-2 的 chunk 内前缀和 ⇒ 自然对数前缀和 gc = g * ln2。
    g_nat = g.float() * LN2                                      # [B,T,hv]
    gc_c = _to_chunks(g_nat.unsqueeze(-1), 1, edge_pad=True)     # [BN,bt,1]

    # 【D1/D7】传入的 A：pad 行补**单位行**（等价于该 pad token 的 (I+L)^{-1} = I 的
    # 对应行）。零填充在数学上同样安全（pad 行的 A 只影响 pad 行自身的 u/w，而
    # v_beta/k_beta_g 的 pad 行恒为 0），补单位行只是与「重算 A」的语义逐位一致。
    a_f = a_mat.float()                                              # [B,T,hv,bt]
    if pad:
        tail = a_f.new_zeros(batch, pad, hv, bt)                     # [B,pad,hv,bt]
        idx = torch.arange(pad, device=device)                   # [pad]
        tail[:, idx, :, (seq_len + idx) % bt] = 1.0                    # 单位行
        a_f = torch.cat([a_f, tail], dim=1)                      # [B,tp,hv,bt]
    a_c = a_f.permute(0, 2, 1, 3).reshape(-1, bt, bt)            # [BN,bt,bt]

    # ------------------------------------------------------------- 常量掩码矩阵
    ar = torch.arange(bt, device=device)                         # [bt]
    m_le = (ar.unsqueeze(-1) >= ar.unsqueeze(-2)).float()        # [bt,bt] 下三角含对角
    m_lt = (ar.unsqueeze(-1) > ar.unsqueeze(-2)).float()         # [bt,bt] 严格下三角
    e_last = torch.zeros(bt, 1, dtype=torch.float32, device=device)
    e_last[bt - 1, 0] = 1.0                                      # [bt,1] 末行指示向量

    # ------------------------------------------- chunk-parallel 前向量（与 S 无关）
    f_mat = _chunk_forward_prep(ForwardPrepConfig(
        q=q_c, k=k_c, v=v_c, beta=beta_c, gc=gc_c, scale=scale,
        m_le=m_le, m_lt=m_lt, a_c=a_c, use_given_a=use_given_a))

    def _bn(x):
        """``[BN, ...]`` → ``[bh, num_chunks, ...]``，便于按 chunk 串行索引。"""
        return x.reshape(bh, num_chunks, *x.shape[1:])

    fb = {name: _bn(t) for name, t in f_mat.items()}

    # ------------------------------------------------------- 初始状态 / dht 归一化
    # 内部一律按 [K, V] 布局；state_v_first 时在入口/出口做转置（out-of-place）。
    if initial_state is not None:
        h0 = initial_state.float()                               # [N,hv,K,V] 或 [N,hv,V,K]
        if state_v_first:
            h0 = _tp(h0)                                         # [N,hv,V,K] → [N,hv,K,V]
        h0 = h0.reshape(bh, d_head, d_value)                                # [bh,K,V]
    else:
        h0 = torch.zeros(bh, d_head, d_value, dtype=torch.float32, device=device)   # [bh,K,V]

    if dht is not None:
        ds = dht.float()                                         # [N,hv,K,V] 或 [N,hv,V,K]
        if state_v_first:
            ds = _tp(ds)
        ds = ds.reshape(bh, d_head, d_value).clone()                        # [bh,K,V] clone：绝不 in-place 改入参
    else:
        ds = torch.zeros(bh, d_head, d_value, dtype=torch.float32, device=device)   # [bh,K,V]

    # =========================================================================
    # PASS 1：正向串行扫描，缓存每个 chunk 的 ENTERING state s_in[c]
    # =========================================================================
    state = h0.clone()                                               # [bh,K,V]
    s_in = []                                                    # list[num_chunks] of [bh,K,V]
    for c in range(num_chunks):
        s_in.append(state)
        v_new = fb["u"][:, c] - _mm(fb["w"][:, c], state)            # [bh,bt,V] NS#1 残差形式
        # S ← S·e^{g_last} + (k·e^{g_last-gc})ᵀ @ v_new
        state = state * fb["decay_s"][:, c] + _mm(_tp(fb["k_dec"][:, c]), v_new)   # [bh,K,V]

    # =========================================================================
    # PASS 2：反向串行扫描，carry ds = dl/dS_{c+1}；逐 chunk 写出全部梯度
    # =========================================================================
    d_q_all = [None] * num_chunks                                        # 每项 [bh,bt,K]
    d_k_all = [None] * num_chunks                                        # 每项 [bh,bt,K]
    d_v_all = [None] * num_chunks                                        # 每项 [bh,bt,V]
    d_b_all = [None] * num_chunks                                        # 每项 [bh,bt,1]
    d_g_all = [None] * num_chunks                                        # 每项 [bh,bt,1]

    for c in range(num_chunks - 1, -1, -1):
        s_i = s_in[c]                                            # [bh,K,V] 本 chunk 的进入状态
        q_s = fb["q_s"][:, c]                                    # [bh,bt,K]
        k_f = fb["k"][:, c]                                      # [bh,bt,K]
        v_f = fb["v"][:, c]                                      # [bh,bt,V]
        b_f = fb["beta"][:, c]                                   # [bh,bt,1]
        gexp = fb["gexp"][:, c]                                  # [bh,bt,1]
        v_beta = fb["v_beta"][:, c]                              # [bh,bt,V]
        k_beta = fb["k_beta"][:, c]                              # [bh,bt,K]
        k_beta_g = fb["k_beta_g"][:, c]                          # [bh,bt,K]
        l_mask = fb["l_mask"][:, c]                              # [bh,bt,bt]
        kkt = fb["kkt"][:, c]                                    # [bh,bt,bt]
        a_inv = fb["a_inv"][:, c]                                # [bh,bt,bt]
        u = fb["u"][:, c]                                        # [bh,bt,V]
        w = fb["w"][:, c]                                        # [bh,bt,K]
        qkt = fb["qkt"][:, c]                                    # [bh,bt,bt]
        attn = fb["attn"][:, c]                                  # [bh,bt,bt]
        qg = fb["qg"][:, c]                                      # [bh,bt,K]
        decay_k = fb["decay_k"][:, c]                            # [bh,bt,1]
        k_dec = fb["k_dec"][:, c]                                # [bh,bt,K]
        decay_s = fb["decay_s"][:, c]                            # [bh,1,1]
        do_i = _bn(do_c)[:, c]                                   # [bh,bt,V]

        v_new = u - _mm(w, s_i)                                  # [bh,bt,V]

        # ---- o = qg @ s_i + attn @ v_new 的四个偏导 ----
        d_attn = _mm(do_i, _tp(v_new))                           # [bh,bt,bt]
        d_v_new_1 = _mm(_tp(attn), do_i)                         # [bh,bt,V] = triton 的 dv_local
        d_qg = _mm(do_i, _tp(s_i))                               # [bh,bt,K]
        ds_from_ointer = _mm(_tp(qg), do_i)                      # [bh,K,V]

        d_qkt = d_attn * l_mask                                  # [bh,bt,bt]
        d_lmask_1 = d_attn * qkt                                 # [bh,bt,bt]（上三角后续被 tril 清零）
        d_q_s_1 = _mm(d_qkt, k_f)                                # [bh,bt,K]
        d_k_1 = _mm(_tp(d_qkt), q_s)                             # [bh,bt,K]

        d_q_s_2 = d_qg * gexp                                    # [bh,bt,K]
        d_gcum_exp_1 = torch.sum(d_qg * q_s, dim=-1, keepdim=True)   # [bh,bt,1]

        # ---- state update 反向（用传入的 carry ds = dl/dS_{c+1}）----
        d_k_dec = _mm(v_new, _tp(ds))                            # [bh,bt,K]
        d_v_new_2 = _mm(k_dec, ds)                               # [bh,bt,V]
        ds_from_decay = ds * decay_s                             # [bh,K,V]
        d_decay_s = torch.sum(ds * s_i, dim=(-2, -1), keepdim=True)   # [bh,1,1]

        d_k_2 = d_k_dec * decay_k                                # [bh,bt,K]
        d_decay_k = torch.sum(d_k_dec * k_f, dim=-1, keepdim=True)    # [bh,bt,1]
        d_arg = d_decay_k * decay_k                              # [bh,bt,1] arg = g_last - gc
        d_gcum_col_1 = -d_arg                                    # [bh,bt,1] 来自 -gc 项
        d_glast_1 = torch.sum(d_arg, dim=-2, keepdim=True)       # [bh,1,1] 来自 +g_last 项

        d_v_new = d_v_new_1 + d_v_new_2                          # [bh,bt,V] = triton 的 dv2 (=du)
        d_u = d_v_new                                            # [bh,bt,V]
        d_v_prime = -d_v_new                                     # [bh,bt,V]
        d_w = _mm(d_v_prime, _tp(s_i))                           # [bh,bt,K] = triton 的 dw
        ds_from_vprime = _mm(_tp(w), d_v_prime)                  # [bh,K,V]

        ds_out = ds_from_decay + ds_from_ointer + ds_from_vprime  # [bh,K,V]

        d_a_inv = _mm(d_u, _tp(v_beta)) + _mm(d_w, _tp(k_beta_g))     # [bh,bt,bt]
        d_v_beta = _mm(_tp(a_inv), d_u)                          # [bh,bt,V]
        d_k_beta_g = _mm(_tp(a_inv), d_w)                        # [bh,bt,K]

        # ---- 矩阵求逆规则：a_inv = (I - A0)^{-1} ⇒ dA0 = A_invᵀ @ d_a_inv @ A_invᵀ ----
        # 注意：这里只用到 a_inv 的**值**，不需要真的求逆 —— a_inv 就是传入的 A。
        d_a_full = _mm(_mm(_tp(a_inv), d_a_inv), _tp(a_inv))     # [bh,bt,bt]

        # ---- A0 = -strict_lower(kkt ⊙ l_mask)：负号 + 严格下三角门控 ----
        d_kl = -d_a_full * m_lt                                  # [bh,bt,bt]
        d_kkt = d_kl * l_mask                                    # [bh,bt,bt]
        d_lmask_2 = d_kl * kkt                                   # [bh,bt,bt]

        d_k_beta_1 = _mm(d_kkt, k_f)                             # [bh,bt,K]
        d_k_3 = _mm(_tp(d_kkt), k_beta)                          # [bh,bt,K]

        d_lmask = d_lmask_1 + d_lmask_2                          # [bh,bt,bt]
        d_gdiff = (d_lmask * m_le) * l_mask * m_le               # [bh,bt,bt] 外层 tril→⊙exp→内层 tril
        rowsum = torch.sum(d_gdiff, dim=-1, keepdim=True)        # [bh,bt,1] 对 gc_i 的贡献（+）
        colsum = torch.sum(d_gdiff, dim=-2).unsqueeze(-1)        # [bh,bt,1] 对 gc_j 的贡献（-）
        d_gcum_l = rowsum - colsum                               # [bh,bt,1]

        d_k_beta_2 = d_k_beta_g * gexp                           # [bh,bt,K]
        d_gcum_exp_2 = torch.sum(d_k_beta_g * k_beta, dim=-1, keepdim=True)   # [bh,bt,1]

        # ---- e^{gc} 的总梯度（decay_s 的贡献用 e_last 折叠到第 bt-1 行）----
        d_gcum_exp = d_gcum_exp_1 + d_gcum_exp_2 + e_last * d_decay_s         # [bh,bt,1]
        d_gcum_from_exp = d_gcum_exp * gexp                      # [bh,bt,1] 穿过 exp

        # ---- gc 的总梯度 ----
        d_gcum_col = d_gcum_l + d_gcum_col_1 + d_gcum_from_exp + e_last * d_glast_1   # [bh,bt,1]

        # ---- cumsum 的反向 = REVERSE-CUMSUM = tril_leᵀ @ d_gc（NS#5，绝不用 total-cumsum）----
        d_g_col = _mm(_tp(m_le), d_gcum_col)                     # [bh,bt,1]

        d_v_f = d_v_beta * b_f                                   # [bh,bt,V] 最终 dv
        d_beta_1 = torch.sum(d_v_beta * v_f, dim=-1, keepdim=True)   # [bh,bt,1]

        d_k_beta = d_k_beta_1 + d_k_beta_2                       # [bh,bt,K]
        d_k_4 = d_k_beta * b_f                                   # [bh,bt,K]
        d_beta_2 = torch.sum(d_k_beta * k_f, dim=-1, keepdim=True)    # [bh,bt,1]

        # ---- 汇总 ----
        d_q_f = (d_q_s_1 + d_q_s_2) * scale                      # [bh,bt,K] q_s = q · scale
        d_k_f = d_k_1 + d_k_2 + d_k_3 + d_k_4                    # [bh,bt,K] dk 四路
        d_beta = d_beta_1 + d_beta_2                             # [bh,bt,1] dβ 两路

        d_q_all[c] = d_q_f
        d_k_all[c] = d_k_f
        d_v_all[c] = d_v_f
        d_b_all[c] = d_beta
        d_g_all[c] = d_g_col

        ds = ds_out                                              # [bh,K,V] out-of-place 反向 carry

    # 反向扫完 chunk 0 之后，ds 即 dl/d(initial_state)。
    dh0 = None
    if initial_state is not None:
        dh0 = ds.reshape(*initial_state.shape[:2], d_head, d_value)         # [N,hv,K,V]
        if state_v_first:
            dh0 = _tp(dh0).contiguous()                          # [N,hv,V,K]
        dh0 = dh0.float()

    # ---------------------------------------------------- layout：→ [B, T, hv, D]
    def _from_chunks(lst, d_model):
        x = torch.stack(lst, dim=1)                              # [bh,num_chunks,bt,D]
        x = x.reshape(batch, hv, tp, d_model).permute(0, 2, 1, 3)          # [B,tp,hv,D]
        return x[:, :seq_len].contiguous()                             # [B,T,hv,D]

    dq = _from_chunks(d_q_all, d_head)                                # [B,T,hv,K]
    dk = _from_chunks(d_k_all, d_head)                                # [B,T,hv,K]
    dv = _from_chunks(d_v_all, d_value)                                # [B,T,hv,V]
    db = _from_chunks(d_b_all, 1).squeeze(-1)                    # [B,T,hv]
    dg = _from_chunks(d_g_all, 1).squeeze(-1)                    # [B,T,hv]

    # GVA：q/k 只有 H 个头，把 hv 个头按组求和归约回 H（对齐 fla wy_fast.py:325-326）。
    if rep > 1:
        dq = dq.reshape(batch, seq_len, n_heads, rep, d_head).sum(3)                  # [B,T,H,K]
        dk = dk.reshape(batch, seq_len, n_heads, rep, d_head).sum(3)                  # [B,T,H,K]

    return BwdCoreOutput(dq=dq, dk=dk, dv=dv, db=db, dg=dg, dh0=dh0)


# =============================================================================
# varlen：chunk_indices 一致性断言（D6 —— 接受但忽略，仅做校验）
# =============================================================================


def _expected_chunk_indices(cu_seqlens, bt, device):
    """按 ``fla/ops/utils/index.py::prepare_chunk_indices`` 的语义重建 chunk_indices。

    第 s 行 ``(i_n, i_t)``：全局第 s 个 chunk 属于序列 ``i_n`` 的第 ``i_t`` 个 chunk。

    Args:
        cu_seqlens: ``[N+1]`` int64 前缀和。
    Returns:
        ``[sum(NT_i), 2]`` int64。
    """
    cu = cu_seqlens.tolist()
    rows = []
    for i_n in range(len(cu) - 1):
        nt = (cu[i_n + 1] - cu[i_n] + bt - 1) // bt
        for i_t in range(nt):
            rows.append((i_n, i_t))
    return torch.tensor(rows, dtype=torch.int64, device=device).reshape(-1, 2)


# =============================================================================
# 公开入口
# =============================================================================


def chunk_gated_delta_rule_bwd_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_mat: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cp_context=None,
    chunk_indices: torch.LongTensor | None = None,
    use_gate_in_kernel: bool = False,
    g_input: torch.Tensor | None = None,
    a_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    chunk_size: int = 64,
):
    """gated delta rule 反向的纯 torch golden 参考实现。

    签名与 ``custom/gated_delta_rule/chunk.py::chunk_gated_delta_rule_bwd``
    （triton 版）**逐参数完全一致**。

    ⚠️ **本 golden 吃传入的 ``A``（D1/D7），因此不是数学绝对真值；``A`` 自身的误差
    不计入对拍** —— 详见文件头 §关于入参 A。

    Args:
        q:  ``[B, T, H, K]``，query（若上层开 l2norm，此处已是归一化后的）。
        k:  ``[B, T, H, K]``，key。
        v:  ``[B, T, hv, V]``，value；支持 GVA（``hv`` 为 ``H`` 的整数倍）。
        g:  ``[B, T, hv]`` fp32，**chunk-local-cumsum 后 × RCP_LN2 的 base-2 gate**。
        beta: ``[B, T, hv]``，写入强度（已过 sigmoid）。
        A:  ``[B, T, hv, bt]``，前向保存的 ``(I+L)^{-1}``。**直接消费，不重算**（D1/D7）。
        scale: float，q 的缩放系数（任意值）。
        initial_state: ``[N, hv, K, V]``（``state_v_first`` 时为 ``[N, hv, V, K]``）
            或 ``None``；为 ``None`` 时不产生 ``dh0``。
        do: ``[B, T, hv, V]``，上游 ``dl/do``。
        dht: ``[N, hv, K, V]`` 或 ``None``，final_state 的上游梯度，作为 ds 的种子。
        state_v_first: 状态按 ``[V, K]`` 布局存放（D8：两种取值均支持）。
        cu_seqlens: ``[N+1]`` int64 或 ``None``。非 ``None`` 时走 **varlen** 路径
            （D3/D6）：要求 ``q.shape[0] == 1``，token 轴为 packed ``sum(T_i)``，
            按 ``s = cu_seqlens[b+1] - cu_seqlens[b]`` 逐序列处理。
        cp_context: **不在本版范围**，非 ``None`` 时抛 ``NotImplementedError``。
        chunk_indices: varlen 专用。**接受但忽略**（D6）；若传入则与 ``cu_seqlens``
            推导出的结果做一致性断言，不一致直接 ``ValueError``。
        use_gate_in_kernel: 核内融合 ``-exp(a_log)·softplus(g_input + dt_bias)``；
            为 True 时额外穿透该激活，产出 ``da_log`` / ``ddt_bias``（D4）。
        g_input / a_log / dt_bias: gate-in-kernel 的原始输入与参数。
        chunk_size: bt，**仅支持 64**（D9），其他值 ``ValueError``。

    Returns:
        8 元组 ``(dq, dk, dv, db, dg, dh0, da_log, ddt_bias)``——
        ⚠️ **``db`` 在 ``dg`` 之前**。

        * ``dq``: ``[B, T, H, K]``  dtype = ``q.dtype``
        * ``dk``: ``[B, T, H, K]``  dtype = ``k.dtype``
        * ``dv``: ``[B, T, hv, V]`` dtype = ``v.dtype``
        * ``db``: ``[B, T, hv]``    dtype = ``beta.dtype``
        * ``dg``: ``[B, T, hv]``    dtype = ``float32``（``use_gate_in_kernel=False``）
          / ``g_input.dtype``（``use_gate_in_kernel=True`` —— **D14**，逐字对齐 fla
          ``gate.py:219``；原 D11「恒 fp32」已作废）
        * ``dh0``: ``[N, hv, K, V]`` fp32，``initial_state is None`` 时为 ``None``
        * ``da_log`` / ``ddt_bias``: 仅 ``use_gate_in_kernel=True`` 时非 ``None``
          （``ddt_bias`` 还要求 ``dt_bias is not None``，见 D12）
    """
    opts = BwdOptions(
        state_v_first=state_v_first, cu_seqlens=cu_seqlens, cp_context=cp_context,
        chunk_indices=chunk_indices, use_gate_in_kernel=use_gate_in_kernel,
        g_input=g_input, a_log=a_log, dt_bias=dt_bias, chunk_size=chunk_size,
        use_given_a=True,
    )
    return _bwd_golden_impl(
        q, k, v, g, beta, a_mat, scale, initial_state, do, dht, opts,
    )


class BwdGoldenOutput(NamedTuple):
    """_bwd_golden_impl 的 8 个返回值。"""
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor
    db: torch.Tensor
    dg: torch.Tensor
    dh0: torch.Tensor | None
    da_log: torch.Tensor | None
    ddt_bias: torch.Tensor | None


def _bwd_golden_impl(
    q, k, v, g, beta, a_mat, scale, initial_state, do, dht,
    opts: BwdOptions,
):
    """公开入口的实现体。

    ``opts``: 见 :class:`BwdOptions`。``opts.use_given_a``：``True``（默认，D1/D7 语义）
    直接吃传入 ``A``；``False`` 复现**改造前**的语义（内部 block-doubling 重算 A），
    **仅供自检里的等价性对拍使用**，不对外暴露。
    """
    cu_seqlens = opts.cu_seqlens
    cp_context = opts.cp_context
    chunk_indices = opts.chunk_indices
    use_gate_in_kernel = opts.use_gate_in_kernel
    g_input = opts.g_input
    a_log = opts.a_log
    dt_bias = opts.dt_bias
    # ---------------------------------------------------------------- 入参校验
    if cp_context is not None:
        raise NotImplementedError(
            "golden 不支持 context-parallel（cp_context）路径：CP 需要跨 rank 的 h0/dht "
            "expand/compress 与集合通信（fla.ops.cp.*），超出本版单卡参考实现的范围。"
            "（ORCHESTRATION_BRIEF D3：cp_context 不在本版范围。）"
        )

    bt = opts.chunk_size

    batch, seq_len, n_heads, d_head = q.shape
    hv, d_value = v.shape[2], v.shape[3]
    if hv % n_heads != 0:
        raise ValueError(f"hv ({hv}) must be divisible by n_heads ({n_heads}).")
    if a_mat is None:
        raise ValueError("a_mat 不能为 None：本 golden 直接消费前向保存的 (eye+l_mat)^{-1}（D1/D7）。")
    if tuple(a_mat.shape) != (batch, seq_len, hv, bt):
        raise ValueError(f"a_mat shape mismatch: expected {(batch, seq_len, hv, bt)}, got {tuple(a_mat.shape)}.")

    q_dtype, k_dtype, v_dtype, beta_dtype = q.dtype, k.dtype, v.dtype, beta.dtype

    # ================================================================ 主体分发
    if cu_seqlens is None:
        # ---- 等长 batch 路径 ----
        core_out = _bwd_core(
            q, k, v, g, beta, a_mat, scale, initial_state, do, dht, opts)
        dq, dk, dv, db, dg, dh0 = core_out
    else:
        # ---- 【D3/D6】varlen（packed）路径 ----
        if batch != 1:
            raise ValueError(f"varlen 要求 q.shape[0] == 1（packed），got batch={batch}.")
        cu = cu_seqlens.to(torch.int64).reshape(-1)              # [N+1]
        n_seqs = cu.numel() - 1
        if int(cu[0]) != 0 or int(cu[-1]) != seq_len:
            raise ValueError(
                f"cu_seqlens 必须是从 0 开始、以 seq_len={seq_len} 结束的前缀和，got "
                f"[{int(cu[0])}, ..., {int(cu[-1])}]."
            )
        for n in range(n_seqs):
            if int(cu[n + 1]) <= int(cu[n]):
                raise ValueError(f"cu_seqlens 必须严格递增，第 {n} 段长度 <= 0。")
        for name, t in (("initial_state", initial_state), ("dht", dht)):
            if t is not None and t.shape[0] != n_seqs:
                raise ValueError(
                    f"varlen: {name}.shape[0] 应为 n_seqs={n_seqs}（= len(cu_seqlens)-1），"
                    f"got {t.shape[0]}."
                )
        # chunk_indices：接受但忽略；传入时做一致性断言（D6）。
        if chunk_indices is not None:
            exp_ci = _expected_chunk_indices(cu, bt, chunk_indices.device)   # [sum(NT_i),2]
            got_ci = chunk_indices.to(torch.int64).reshape(-1, 2)
            if got_ci.shape != exp_ci.shape or not bool(torch.equal(got_ci, exp_ci)):
                raise ValueError(
                    "chunk_indices 与 cu_seqlens 推导结果不一致（本 golden 只消费 "
                    f"cu_seqlens，D6）：expected shape {tuple(exp_ci.shape)}, "
                    f"got {tuple(got_ci.shape)}。"
                )

        dq_l, dk_l, dv_l, db_l, dg_l, dh0_l = [], [], [], [], [], []
        for n in range(n_seqs):
            t0, t1 = int(cu[n]), int(cu[n + 1])                  # s = t1 - t0
            is_n = initial_state[n:n + 1] if initial_state is not None else None   # [1,hv,·,·]
            dht_n = dht[n:n + 1] if dht is not None else None                      # [1,hv,·,·]
            o = _bwd_core(
                q[:, t0:t1], k[:, t0:t1], v[:, t0:t1], g[:, t0:t1], beta[:, t0:t1],
                a_mat[:, t0:t1], scale, is_n, do[:, t0:t1], dht_n, opts)
            dq_l.append(o.dq)
            dk_l.append(o.dk)
            dv_l.append(o.dv)
            db_l.append(o.db)
            dg_l.append(o.dg)
            if o.dh0 is not None:
                dh0_l.append(o.dh0)                               # [1,hv,·,·]
        # 沿 packed token 轴拼回；state 沿 N 轴拼回。
        dq = torch.cat(dq_l, dim=1)                              # [1,T,H,K]
        dk = torch.cat(dk_l, dim=1)                              # [1,T,H,K]
        dv = torch.cat(dv_l, dim=1)                              # [1,T,hv,V]
        db = torch.cat(db_l, dim=1)                              # [1,T,hv]
        dg = torch.cat(dg_l, dim=1)                              # [1,T,hv]
        dh0 = torch.cat(dh0_l, dim=0) if dh0_l else None         # [N,hv,·,·] 或 None

    # ----------------------------------------------- gate-in-kernel 的额外一层反向
    da_log, ddt_bias = None, None
    if use_gate_in_kernel:
        if g_input is None or a_log is None:
            raise ValueError("use_gate_in_kernel=True requires both `g_input` and `a_log`.")
        # 注意：da_log / ddt_bias 是**全部 token**（varlen 下跨所有序列）的求和。
        # 【D14】此分支返回的 dg 已是 g_input.dtype，下面**不得**再 .float()。
        dg, da_log, ddt_bias = _gdn_gate_bwd_golden(g_input, a_log, dt_bias, dg)
    else:
        # 非 gate 路径：dg 恒 fp32（对齐 fla ``chunk_local_cumsum`` 的 fp32 输出）。
        dg = dg.float()

    # -------------------------------------------------------------- dtype 回 cast
    # 对齐 fla：dq/dk/dv/db 跟随各自输入 dtype；dh0 恒 fp32；
    # dg 见上 —— 非 gate 路径 fp32，gate 路径跟随 g_input.dtype（D14）。
    return BwdGoldenOutput(
        dq=dq.to(q_dtype), dk=dk.to(k_dtype), dv=dv.to(v_dtype), db=db.to(beta_dtype),
        dg=dg, dh0=dh0, da_log=da_log, ddt_bias=ddt_bias)


# =============================================================================
# 前向的 WY representation：给 make_inputs 造出一致的 A
# =============================================================================


def recompute_a_from_forward(k, g, beta, chunk_size=64, out_dtype=None):
    """按前向逻辑算出 fla 约定的 ``A = (I + L)^{-1}``，shape ``[B, T, hv, bt]``。

    ``L[i, j] = β_i · (k_i·k_j) · exp2(g_i - g_j)``（严格下三角），与
    ``chunk_fwd.py::chunk_gated_delta_rule_fwd_kkt_solve_kernel`` 等价；求逆同样走
    16×16 对角块 + block-doubling。**A 必须来自前向，不能随机构造**。

    Args:
        k: ``[B, T, H, K]``；g: ``[B, T, hv]``（base-2 chunk-cumsum）；
        beta: ``[B, T, hv]``。
    """
    batch, seq_len, n_heads, d_head = k.shape
    hv = beta.shape[2]
    rep = hv // n_heads
    bt = chunk_size
    num_chunks = (seq_len + bt - 1) // bt
    tp, pad = num_chunks * bt, num_chunks * bt - seq_len
    device = k.device

    k_e = k.float().repeat_interleave(rep, dim=2) if rep > 1 else k.float()   # [B,T,hv,K]
    gc = g.float() * LN2                                         # [B,T,hv] 自然对数前缀和
    b = beta.float()                                             # [B,T,hv]
    if pad:
        k_e = torch.cat([k_e, k_e.new_zeros(batch, pad, hv, d_head)], dim=1)          # [B,tp,hv,K] 零填充
        gc = torch.cat([gc, gc[:, -1:, :].expand(batch, pad, hv)], dim=1)        # [B,tp,hv] 边缘复制
        b = torch.cat([b, b.new_zeros(batch, pad, hv)], dim=1)                   # [B,tp,hv] 零填充

    k_c = k_e.permute(0, 2, 1, 3).reshape(-1, bt, d_head)             # [B*hv*num_chunks,bt,K]
    gc_c = gc.permute(0, 2, 1).reshape(-1, bt, 1)                # [B*hv*num_chunks,bt,1]
    b_c = b.permute(0, 2, 1).reshape(-1, bt, 1)                  # [B*hv*num_chunks,bt,1]

    ar = torch.arange(bt, device=device)                         # [bt]
    m_le = (ar.unsqueeze(-1) >= ar.unsqueeze(-2)).float()        # [bt,bt] 下三角含对角
    m_lt = (ar.unsqueeze(-1) > ar.unsqueeze(-2)).float()         # [bt,bt] 严格下三角

    l_mask = torch.exp((gc_c - _tp(gc_c)) * m_le) * m_le         # [B*hv*num_chunks,bt,bt] mask-before-exp
    kkt = _mm(k_c * b_c, _tp(k_c))                               # [B*hv*num_chunks,bt,bt]
    m_lt = torch.eye(bt, dtype=torch.float32,
                     device=device) + (kkt * l_mask) * m_lt      # [B*hv*num_chunks,bt,bt] 单位下三角 I+L
    a_inv = _invert_unit_lower_triangular(m_lt)                  # [B*hv*num_chunks,bt,bt] = (I+L)^{-1}

    a_mat = a_inv.reshape(batch, hv, tp, bt).permute(0, 2, 1, 3)[:, :seq_len].contiguous()  # [B,T,hv,bt]
    return a_mat.to(out_dtype) if out_dtype is not None else a_mat


# =============================================================================
# 输入工厂（可复现）
# =============================================================================


def make_inputs(batch=2, seq_len=128, n_heads=2, d_head=64, d_value=64, hv=None, dtype=torch.float32,
                device=None, seed=0, with_gate=False):
    """构造一组可复现、语义合法的**等长** batch 反向入参。

    结构约束（不能随机）：
      * ``q`` / ``k`` 必须 L2-normalize（上游 ``use_qk_l2norm_in_kernel`` 的约定，
        也保证 ``k_i·k_j ∈ [-1, 1]`` 让 WY 求逆条件数良好）；
      * ``beta ∈ (0, 1)`` 走 sigmoid；
      * ``g`` 必须是 ``RCP_LN2 · cumsum_chunk(logsigmoid(·))``，即 base-2 的
        chunk-local 前缀和，且恒 ≤ 0；
      * ``A`` 必须由前向 WY representation 算出（``recompute_a_from_forward``），
        **不能随机**。

    Args:
        hv: v 的头数，缺省 = ``H``；``hv > H`` 时构造 GVA case（需 ``hv % H == 0``）。
        with_gate: 额外返回 ``use_gate_in_kernel=True`` 所需的
            ``g_input``/``a_log``/``dt_bias``（此时 ``g`` 仍按上式独立构造，
            因为反向不校验 g 与 g_input 的一致性）。

    Returns:
        dict：可直接 ``chunk_gated_delta_rule_bwd_golden(**inputs)``。
    """
    if device is None:
        device = _get_device()
    if hv is None:
        hv = n_heads
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device)

    bt = 64                                                      # 【D9】仅支持 64
    q = torch.nn.functional.normalize(rn(batch, seq_len, n_heads, d_head), p=2.0, dim=-1)     # [B,T,H,K]
    k = torch.nn.functional.normalize(rn(batch, seq_len, n_heads, d_head), p=2.0, dim=-1)     # [B,T,H,K]
    v = rn(batch, seq_len, hv, d_value)                                          # [B,T,hv,V]
    beta = torch.sigmoid(rn(batch, seq_len, hv))                           # [B,T,hv]
    do = rn(batch, seq_len, hv, d_value)                                         # [B,T,hv,V]

    # g_input: per-token 自然 log gate (≤0) → chunk-local cumsum → ×RCP_LN2 (base-2)
    g_in = torch.nn.functional.logsigmoid(rn(batch, seq_len, hv))          # [B,T,hv]
    num_chunks = (seq_len + bt - 1) // bt
    pad = num_chunks * bt - seq_len
    g_pad = torch.cat([g_in, g_in.new_zeros(batch, pad, hv)], dim=1) if pad else g_in
    g_cs = g_pad.reshape(batch, num_chunks, bt, hv).cumsum(dim=2).reshape(batch, num_chunks * bt, hv)[:, :seq_len]
    g = (g_cs * RCP_LN2)                                         # [B,T,hv] fp32

    a_mat = recompute_a_from_forward(k, g, beta, chunk_size=bt, out_dtype=dtype)   # [B,T,hv,bt]

    initial_state = rn(batch, hv, d_head, d_value)                              # [B,hv,K,V]
    dht = rn(batch, hv, d_head, d_value)                                        # [B,hv,K,V]

    def _cast(x):
        return x.to(dtype)

    out = dict(
        q=_cast(q), k=_cast(k), v=_cast(v), g=g.float(), beta=_cast(beta), a_mat=a_mat,
        scale=d_head ** -0.5, initial_state=initial_state.float(), do=cast(do),
        dht=dht.float(), chunk_size=bt,
    )
    if with_gate:
        out.update(
            use_gate_in_kernel=True,
            g_input=rn(batch, seq_len, hv),                                # [B,T,hv] raw gate
            a_log=rn(hv),                                        # [hv]
            dt_bias=rn(hv),                                      # [hv]
        )
    return out


def make_varlen_inputs(seq_lens, n_heads=2, d_head=64, d_value=64, hv=None, dtype=torch.float32,
                       device=None, seed=0, with_gate=False):
    """构造 **varlen（packed）** 入参 —— 由若干个 ``B=1`` 的等长 case 沿 token 轴拼接而成。

    这样构造的 packed 输入天然满足 varlen 语义（``g`` 的 chunk-local cumsum 与
    ``A`` 的 chunk 划分都在**每个序列内部**重新起算），因此它同时也给出了
    「逐序列单独调用等长路径」这一独立参照，用于 varlen 对拍。

    Args:
        seq_lens: ``list[int]``，各序列长度 ``T_i``。

    Returns:
        ``(packed_kwargs, per_seq_kwargs_list)``：
          * ``packed_kwargs``：含 ``cu_seqlens``/``chunk_indices``，
            ``q``/``k`` ``[1, sum(T_i), H, K]``、``initial_state``/``dht`` ``[N, hv, K, V]``；
          * ``per_seq_kwargs_list``：``N`` 组 ``B=1`` 的等长入参（无 ``cu_seqlens``）。
    """
    if device is None:
        device = _get_device()
    if hv is None:
        hv = n_heads
    bt = 64
    per_seq = [make_inputs(batch=1, seq_len=s, n_heads=n_heads, d_head=d_head, d_value=d_value, hv=hv, dtype=dtype,
                           device=device, seed=seed + 1000 * i, with_gate=with_gate)
               for i, s in enumerate(seq_lens)]

    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0)),
                      dtype=torch.int64, device=device)          # [N+1]
    packed = dict(
        q=torch.cat([p["q"] for p in per_seq], dim=1),           # [1,sum(T_i),H,K]
        k=torch.cat([p["k"] for p in per_seq], dim=1),           # [1,sum(T_i),H,K]
        v=torch.cat([p["v"] for p in per_seq], dim=1),           # [1,sum(T_i),hv,V]
        g=torch.cat([p["g"] for p in per_seq], dim=1),           # [1,sum(T_i),hv]
        beta=torch.cat([p["beta"] for p in per_seq], dim=1),     # [1,sum(T_i),hv]
        a_mat=torch.cat([p["a_mat"] for p in per_seq], dim=1),           # [1,sum(T_i),hv,bt]
        do=torch.cat([p["do"] for p in per_seq], dim=1),         # [1,sum(T_i),hv,V]
        initial_state=torch.cat([p["initial_state"] for p in per_seq], dim=0),   # [N,hv,K,V]
        dht=torch.cat([p["dht"] for p in per_seq], dim=0),       # [N,hv,K,V]
        scale=per_seq[0]["scale"], chunk_size=bt,
        cu_seqlens=cu,
        chunk_indices=_expected_chunk_indices(cu, bt, device),   # [sum(NT_i),2]
    )
    if with_gate:
        packed.update(
            use_gate_in_kernel=True,
            g_input=torch.cat([p["g_input"] for p in per_seq], dim=1),   # [1,sum(T_i),hv]
            a_log=per_seq[0]["a_log"],                           # [hv]
            dt_bias=per_seq[0]["dt_bias"],                       # [hv]
        )
        for p in per_seq:                                        # 各序列共享同一组 gate 参数
            p["a_log"] = packed["a_log"]
            p["dt_bias"] = packed["dt_bias"]
    return packed, per_seq


def _make_inputs(device):
    """profiling 工厂（``scripts/profile_golden.py --factory _make_inputs``）。

    返回 ``[(case_name, args_list, kwargs_dict), ...]``，覆盖 SPEC.md 的两组 P0 配置。
    位置参数顺序 = golden 签名前 10 个：
    ``q, k, v, g, beta, A, scale, initial_state, do, dht``。
    """
    def _split(d):
        args = [d["q"], d["k"], d["v"], d["g"], d["beta"], d["a_mat"], d["scale"],
                d["initial_state"], d["do"], d["dht"]]
        _exclude_keys = ("q", "k", "v", "g", "beta", "a_mat", "scale",
                         "initial_state", "do", "dht")
        kwargs = {kk: vv for kk, vv in d.items() if kk not in _exclude_keys}
        return args, kwargs

    cases = []

    # SPEC 典型配置「功能_定长_P0」：B=2,T=256,H=hv=4,K=V=64，fp32。
    d = make_inputs(batch=2, seq_len=256, n_heads=4, d_head=64, d_value=64, dtype=torch.float32,
                    device=device, seed=1)
    a, kw = _split(d)
    cases.append(("func_p0_equal_len_fp32", a, kw))

    # SPEC 典型配置「性能_P0」：训练主场景 —— bf16 + varlen + gate 融合，
    packed, _ = make_varlen_inputs([1024, 1024, 1024, 1024], n_heads=16, d_head=128, d_value=128, hv=32,
                                   dtype=torch.bfloat16, device=device, seed=2,
                                   with_gate=True)
    a, kw = _split(packed)
    cases.append(("perf_p0_varlen_bf16_gate", a, kw))

    return cases


# =============================================================================
# 自检
# =============================================================================


def _naive_forward_autograd(q, k, v, g_in, beta, scale, h0, chunk_size):
    """朴素逐 token 递推前向（可微，fp64），用于 autograd 对拍。

    与 ``custom/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule`` 同构：
    **先衰减、再 delta**（delta 读的是已衰减的状态），与 chunk 形式中 ``w`` 携带
    ``e^{gc}`` 的写法一致。
    """
    batch, seq_len, n_heads, d_head = q.shape
    d_value = v.shape[-1]
    state = h0
    outs = []
    for t in range(seq_len):
        state = state * torch.exp(g_in[:, t]).reshape(batch, n_heads, 1, 1)                     # 先衰减
        bv = v[:, t] - torch.einsum('bhkv, bhk->bhv', state, k[:, t])
        bv = bv * beta[:, t].unsqueeze(-1)
        state = state + torch.einsum('bhk, bhv->bhkv', k[:, t], bv)                    # delta 写入
        outs.append(torch.einsum('bhk, bhkv->bhv', q[:, t] * scale, state))
    return torch.stack(outs, dim=1), state


def _relerr(a, b):
    """相对 L2 误差 ``||a-b|| / ||b||``（fp64 计算）。"""
    a, b = a.double(), b.double()
    return float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-30))


def _validate():
    device = _get_device()
    ok = True
    logging.info("=" * 76)
    logging.info("chunk_gated_delta_rule_bwd_golden 验证报告（Stage 2 改造后）")
    logging.info(f"device = {device}")
    logging.info("=" * 76)

    out_names = ["dq", "dk", "dv", "db", "dg", "dh0", "da_log", "ddt_bias"]

    # ---------------------------------------------------------- [典型 case 验证]
    logging.info("\n[1] 典型 case 验证  batch=2, seq_len=128, n_heads=hv=2, d_head=d_value=64, bt=64")
    inp = make_inputs(batch=2, seq_len=128, n_heads=2, d_head=64, d_value=64,
                      dtype=torch.float32, device=device, seed=1234)
    outs = chunk_gated_delta_rule_bwd_golden(**inp)
    exp_shape = {
        "dq": (2, 128, 2, 64), "dk": (2, 128, 2, 64), "dv": (2, 128, 2, 64),
        "db": (2, 128, 2), "dg": (2, 128, 2), "dh0": (2, 2, 64, 64),
    }
    for n, o in zip(out_names, outs):
        if o is None:
            logging.info(f"  {n:9s} = None")
            continue
        norm = float(torch.linalg.vector_norm(o.float()))
        finite = bool(torch.isfinite(o.float()).all())
        shape_ok = tuple(o.shape) == exp_shape.get(n, tuple(o.shape))
        status = "PASS" if (finite and shape_ok) else "FAIL"
        ok &= (finite and shape_ok)
        logging.info(f"  {n:9s} shape={tuple(o.shape)!s:22s} dtype={str(o.dtype):15s} "
              f"norm={norm:12.5f}  {status}")
    logging.info(f"  返回值个数 = {len(outs)} (期望 8) ... {'PASS' if len(outs) == 8 else 'FAIL'}")
    ok &= (len(outs) == 8)

    # ------------------------------------------------------- [入参不被修改 检查]
    logging.info("\n[2] 入参不被 in-place 修改 检查")
    inp2 = make_inputs(batch=1, seq_len=64, n_heads=1, d_head=32, d_value=32, device=device, seed=7)
    _snap_keys = ("q", "k", "v", "g", "beta", "a_mat", "initial_state", "do", "dht")
    snap = {n: inp2[n].clone() for n in _snap_keys}
    chunk_gated_delta_rule_bwd_golden(**inp2)
    for n, s in snap.items():
        same = bool(torch.equal(s, inp2[n]))
        ok &= same
        logging.info(f"  {n:14s} unchanged ... {'PASS' if same else 'FAIL'}")

    # ============================================================================
    # [3] 【Gate 项】A 语义切换前后对拍
    #     改造前语义 = 内部 block-doubling 重算 A（use_given_a=False）
    #     改造后语义 = 吃传入的 A       （use_given_a=True，即公开入口）
    #     条件：等长 + 传入 **fp32 精确 A**（recompute_a_from_forward 的输出）
    #     期望：两者数学上是同一个量，allclose 必须通过。
    # ============================================================================
    logging.info("\n[3] a_mat 语义切换前后对拍 —— allclose(original[重算A], normalized[吃传入A])")
    logging.info("    条件：等长 batch + 传入 fp32 精确 a_mat")
    for (b, t, h, kk, vv, hv) in [(2, 128, 2, 64, 64, 2), (1, 100, 2, 32, 64, 2),
                                  (2, 256, 2, 64, 64, 4)]:
        i = make_inputs(batch=b, seq_len=t, n_heads=h, d_head=kk, d_value=vv, hv=hv,
                        dtype=torch.float32, device=device, seed=b * 31 + t)
        # 显式使用 fp32 精确 A（make_inputs 在 dtype=fp32 时本就如此，这里再钉一次）
        i["a_mat"] = recompute_a_from_forward(i["k"], i["g"], i["beta"], chunk_size=64)
        _pos_keys = ("q", "k", "v", "g", "beta", "a_mat", "scale", "initial_state", "do", "dht")
        _pos = {k: i[k] for k in _pos_keys}
        _opt = {k: v for k, v in i.items() if k not in _pos_keys}
        normalized = _bwd_golden_impl(**_pos, opts=BwdOptions(**_opt, use_given_a=True))
        original = _bwd_golden_impl(**_pos, opts=BwdOptions(**_opt, use_given_a=False))
        line_ok = True
        errs = []
        for n, a_, b_ in zip(out_names, normalized, original):
            if a_ is None:
                continue
            close = bool(torch.allclose(a_.float(), b_.float(), atol=1e-5, rtol=1e-5))
            errs.append(f"{n}={_relerr(a_, b_):.2e}")
            line_ok &= close
        ok &= line_ok
        logging.info(
            f"  batch={b}, seq_len={t}, n_heads={h}, hv={hv}, "
            f"d_head={kk}, d_value={vv}: allclose(atol=rtol=1e-5) "
            f"... {'PASS' if line_ok else 'FAIL'}"
        )
        logging.info(f"      rel_err: {'  '.join(errs)}")

    # ============================================================================
    # [4] 【Gate 项】varlen 对拍
    #     packed 调用（cu_seqlens 路径） vs 逐序列单独调用等长路径
    # ============================================================================
    logging.info("\n[4] varlen 对拍 —— packed(cu_seqlens) vs 逐序列单独调用等长路径")
    for seq_lens, hh, hv in [([100, 256, 244], 2, 2), ([64, 128], 2, 4), ([37, 64, 195], 1, 2)]:
        packed, per_seq = make_varlen_inputs(seq_lens, n_heads=hh, d_head=32, d_value=32, hv=hv,
                                             dtype=torch.float32, device=device,
                                             seed=hash(tuple(seq_lens)) % 1000)
        got = chunk_gated_delta_rule_bwd_golden(**packed)
        refs = [chunk_gated_delta_rule_bwd_golden(**p) for p in per_seq]
        ref = [None] * 8
        for j in range(5):                                       # dq,dk,dv,db,dg 沿 token 轴拼
            ref[j] = torch.cat([r[j] for r in refs], dim=1)
        ref[5] = torch.cat([r[5] for r in refs], dim=0)          # dh0 沿 N 轴拼
        line_ok, errs = True, []
        for j, n in enumerate(out_names[:6]):
            close = bool(torch.allclose(got[j].float(), ref[j].float(), atol=1e-5, rtol=1e-5))
            errs.append(f"{n}={_relerr(got[j], ref[j]):.2e}")
            line_ok &= close
        ok &= line_ok
        logging.info(f"  seq_lens={seq_lens}, n_heads={hh}, hv={hv}: allclose ... "
              f"{'PASS' if line_ok else 'FAIL'}")
        logging.info(f"      rel_err: {'  '.join(errs)}")

    # varlen + gate 融合：da_log / ddt_bias 是**跨全部序列**的求和
    logging.info("\n  [4b] varlen + use_gate_in_kernel：da_log/ddt_bias 跨序列全局求和")
    packed, per_seq = make_varlen_inputs([100, 156], n_heads=2, d_head=32, d_value=32, hv=2,
                                         dtype=torch.float32, device=device,
                                         seed=77, with_gate=True)
    got = chunk_gated_delta_rule_bwd_golden(**packed)
    refs = [chunk_gated_delta_rule_bwd_golden(**p) for p in per_seq]
    dg_ref = torch.cat([r[4] for r in refs], dim=1)              # [1,sum(T_i),hv]
    da_log_ref = sum(r[6] for r in refs)                          # [hv] 各序列求和再相加
    ddtb_ref = sum(r[7] for r in refs)                           # [hv]
    for n, a_, b_ in (("dg", got[4], dg_ref), ("da_log", got[6], da_log_ref),
                      ("ddt_bias", got[7], ddtb_ref)):
        close = bool(torch.allclose(a_.float(), b_.float(), atol=1e-5, rtol=1e-5))
        ok &= close
        logging.info(f"      {n:9s} rel_err={_relerr(a_, b_):.3e} ... {'PASS' if close else 'FAIL'}")

    # varlen 的 chunk_indices 一致性断言（D6）
    logging.info("\n  [4c] chunk_indices 一致性断言（D6：接受但忽略，不一致必须报错）")
    bad = dict(packed)
    bad["chunk_indices"] = packed["chunk_indices"].clone()
    bad["chunk_indices"][0, 1] += 1
    try:
        chunk_gated_delta_rule_bwd_golden(**bad)
        logging.info("      篡改 chunk_indices ... FAIL (未抛异常)")
        ok = False
    except ValueError:
        logging.info("      篡改 chunk_indices ... PASS (ValueError)")

    # ============================================================================
    # [5] 数学正确性：与 fp64 autograd 朴素逐 token 递推对拍
    #     ⚠️ 传入的是 fp32 **精确** A，故此处 golden 确实逼近真值；
    #        若传入 bf16 A，误差会被 A 自身的量化主导（见文件头 §关于入参 A）。
    # ============================================================================
    logging.info("\n[5] 数学正确性：vs autograd 朴素逐 token 递推 (CPU fp64 参考 vs golden fp32)")
    batch, seq_len, n_heads, d_head, d_value, bt = 2, 128, 2, 32, 32, 64
    dev64 = torch.device("cpu")
    gen = torch.Generator(device="cpu").manual_seed(20260720)
    f64 = dict(dtype=torch.float64)

    def rn(*s):
        return torch.randn(*s, generator=gen, **f64).to(dev64)

    q = torch.nn.functional.normalize(rn(batch, seq_len, n_heads, d_head), p=2.0, dim=-1).requires_grad_(True)
    k = torch.nn.functional.normalize(rn(batch, seq_len, n_heads, d_head), p=2.0, dim=-1).requires_grad_(True)
    v = rn(batch, seq_len, n_heads, d_value).requires_grad_(True)
    beta = torch.sigmoid(rn(batch, seq_len, n_heads)).requires_grad_(True)
    g_in = torch.nn.functional.logsigmoid(rn(batch, seq_len, n_heads)).requires_grad_(True)
    h0 = rn(batch, n_heads, d_head, d_value).requires_grad_(True)
    do = rn(batch, seq_len, n_heads, d_value)
    dht = rn(batch, n_heads, d_head, d_value)
    scale = d_head ** -0.5

    o_ref, ht_ref = _naive_forward_autograd(q, k, v, g_in, beta, scale, h0, bt)
    loss = (o_ref * do).sum() + (ht_ref * dht).sum()
    ref = torch.autograd.grad(loss, [q, k, v, beta, g_in, h0])
    ref = dict(zip(["dq", "dk", "dv", "db", "dg", "dh0"], ref))

    # golden 的 g 入参 = RCP_LN2 · chunk-local cumsum(g_in)
    num_chunks = seq_len // bt
    g_b2 = g_in.detach().reshape(batch, num_chunks, bt, n_heads).cumsum(2).reshape(batch, seq_len, n_heads) * RCP_LN2
    a_mat = recompute_a_from_forward(k.detach(), g_b2, beta.detach(), chunk_size=bt)
    got = chunk_gated_delta_rule_bwd_golden(
        q=q.detach(), k=k.detach(), v=v.detach(), g=g_b2, beta=beta.detach(), a_mat=a_mat,
        scale=scale, initial_state=h0.detach(), do=do, dht=dht, chunk_size=bt,
    )
    got = dict(zip(["dq", "dk", "dv", "db", "dg", "dh0"], got[:6]))
    for n in ["dq", "dk", "dv", "db", "dg", "dh0"]:
        err = _relerr(got[n], ref[n])
        good = err < 2e-5
        ok &= good
        logging.info(f"  {n:5s} rel_err = {err:.3e}   {'PASS' if good else 'FAIL'}")

    # varlen 也与 fp64 autograd 对拍（逐序列各自跑朴素递推）
    logging.info("\n  [5b] varlen 路径 vs fp64 autograd（逐序列朴素递推）")
    seq_lens = [64, 128]
    ttot = sum(seq_lens)
    q = torch.nn.functional.normalize(rn(1, ttot, n_heads, d_head), p=2.0, dim=-1).requires_grad_(True)
    k = torch.nn.functional.normalize(rn(1, ttot, n_heads, d_head), p=2.0, dim=-1).requires_grad_(True)
    v = rn(1, ttot, n_heads, d_value).requires_grad_(True)
    beta = torch.sigmoid(rn(1, ttot, n_heads)).requires_grad_(True)
    g_in = torch.nn.functional.logsigmoid(rn(1, ttot, n_heads)).requires_grad_(True)
    h0 = rn(len(seq_lens), n_heads, d_head, d_value).requires_grad_(True)
    do = rn(1, ttot, n_heads, d_value)
    dht = rn(len(seq_lens), n_heads, d_head, d_value)

    loss = 0.0
    off = 0
    for i_n, s in enumerate(seq_lens):
        o_n, ht_n = _naive_forward_autograd(
            q[:, off:off + s], k[:, off:off + s], v[:, off:off + s],
            g_in[:, off:off + s], beta[:, off:off + s], scale, h0[i_n:i_n + 1], bt)
        loss = loss + (o_n * do[:, off:off + s]).sum() + (ht_n * dht[i_n:i_n + 1]).sum()
        off += s
    ref = dict(zip(["dq", "dk", "dv", "db", "dg", "dh0"],
                   torch.autograd.grad(loss, [q, k, v, beta, g_in, h0])))

    # packed 的 g / A 必须按**每个序列**各自做 chunk-local cumsum / chunk 划分
    g_b2_parts, a_parts, off = [], [], 0
    for s in seq_lens:
        nt = (s + bt - 1) // bt
        pad = nt * bt - s
        gi_s = g_in.detach()[:, off:off + s]                     # [1,s,H]
        gi_p = torch.cat([gi_s, gi_s.new_zeros(1, pad, n_heads)], dim=1) if pad else gi_s
        g_s = gi_p.reshape(1, nt, bt, n_heads).cumsum(2).reshape(1, nt * bt, n_heads)[:, :s] * RCP_LN2
        g_b2_parts.append(g_s)
        a_parts.append(recompute_a_from_forward(
            k.detach()[:, off:off + s], g_s, beta.detach()[:, off:off + s], chunk_size=bt))
        off += s
    g_b2 = torch.cat(g_b2_parts, dim=1)                          # [1,ttot,H]
    a_mat = torch.cat(a_parts, dim=1)                                # [1,ttot,H,bt]
    cu = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0)), dtype=torch.int64)
    got = chunk_gated_delta_rule_bwd_golden(
        q=q.detach(), k=k.detach(), v=v.detach(), g=g_b2, beta=beta.detach(), a_mat=a_mat,
        scale=scale, initial_state=h0.detach(), do=do, dht=dht,
        cu_seqlens=cu, chunk_size=bt)
    got = dict(zip(["dq", "dk", "dv", "db", "dg", "dh0"], got[:6]))
    for n in ["dq", "dk", "dv", "db", "dg", "dh0"]:
        err = _relerr(got[n], ref[n])
        good = err < 2e-5
        ok &= good
        logging.info(f"      {n:5s} rel_err = {err:.3e}   {'PASS' if good else 'FAIL'}")

    # ----------------------------------------------------------- [dht/dh0 检查]
    logging.info("\n[6] dht 种子 / dh0 输出 检查")
    # gate 调温和（g*0.02）：否则 T=64 时 e^{g_last} ≈ 1e-20，dht 对 dh0 的贡献
    # 会被 fp32 完全吃掉，检查失去区分度。
    i3 = make_inputs(batch=1, seq_len=64, n_heads=1, d_head=32, d_value=32, device=device, seed=11)
    i3["g"] = i3["g"] * 0.02
    i3["a_mat"] = recompute_a_from_forward(i3["k"], i3["g"], i3["beta"], chunk_size=64)
    o_with = chunk_gated_delta_rule_bwd_golden(**i3)
    i3z = dict(i3, dht=torch.zeros_like(i3["dht"]))
    o_zero = chunk_gated_delta_rule_bwd_golden(**i3z)
    diff = float(torch.linalg.vector_norm(o_with[5] - o_zero[5]))
    ok &= diff > 1e-6
    logging.info(f"  dht 非零 vs 归零 的 dh0 差异 = {diff:.5e} (应显著非零) ... "
          f"{'PASS' if diff > 1e-6 else 'FAIL'}")
    i3n = dict(i3, initial_state=None)
    o_none = chunk_gated_delta_rule_bwd_golden(**i3n)
    ok &= (o_none[5] is None)
    logging.info(f"  initial_state=None ⇒ dh0 is None ... {'PASS' if o_none[5] is None else 'FAIL'}")

    # -------------------------------------------------------------- [求逆 检查]
    # 【D5 兜底约束的单测】主路径已不调用它，但保留实现与单测供未来重算路径复用。
    logging.info("\n[7] block-doubling 求逆 检查 (16→32→64)，vs solve_triangular")
    logging.info("    ⚠️ 该函数已不在反向主路径上（D1/D7），此处仅作 D5 兜底约束的单测保留")
    gen2 = torch.Generator(device="cpu").manual_seed(3)
    for bt in (16, 32, 64):
        l_mat = (torch.randn(4, bt, bt, generator=gen2) / bt).to(device)
        ar = torch.arange(bt, device=device)
        eye_bt = torch.eye(bt, device=device)
        m_mat = eye_bt + l_mat * (ar.unsqueeze(-1) > ar.unsqueeze(-2)).float()
        m_inv = _invert_unit_lower_triangular(m_mat)
        resid = float(torch.linalg.vector_norm(_mm(m_inv, m_mat) - eye_bt))
        expect = torch.linalg.solve_triangular(m_mat.double(), eye_bt.double().expand_as(m_mat).contiguous(),
                                               upper=False)
        rel = float(torch.linalg.vector_norm(m_inv.double() - expect) /
                    torch.linalg.vector_norm(expect))
        good = resid < 1e-4 and rel < 1e-5
        ok &= good
        logging.info(f"  bt={bt:3d}  ||a_inv @ m_mat - eye||={resid:.3e}  rel_vs_solve_tri={rel:.3e}  "
              f"{'PASS' if good else 'FAIL'}")

    # ----------------------------------------------------- [gate-in-kernel 检查]
    logging.info("\n[8] use_gate_in_kernel 检查（D4 / D10 / D12 / D14）")
    i4 = make_inputs(batch=1, seq_len=64, n_heads=2, d_head=32, d_value=32, device=device, seed=5, with_gate=True)
    o4 = chunk_gated_delta_rule_bwd_golden(**i4)
    good = (o4[6] is not None and tuple(o4[6].shape) == (2,)
            and o4[7] is not None and tuple(o4[7].shape) == (2,))
    ok &= good
    logging.info(f"  da_log shape={tuple(o4[6].shape)}, ddt_bias shape={tuple(o4[7].shape)} ... "
          f"{'PASS' if good else 'FAIL'}")
    # 【D14】dg 跟随 g_input.dtype（gate 路径） / fp32（非 gate 路径）
    good = (o4[4].dtype == torch.float32)                        # 该 case 的 g_input 是 fp32
    ok &= good
    logging.info(f"  D14  gate 路径 g_input=fp32  ⇒ dg dtype = {o4[4].dtype} "
          f"(期望 float32) ... {'PASS' if good else 'FAIL'}")
    o4bf = chunk_gated_delta_rule_bwd_golden(
        **dict(i4, g_input=i4["g_input"].to(torch.bfloat16)))
    good = (o4bf[4].dtype == torch.bfloat16)
    ok &= good
    logging.info(f"  D14  gate 路径 g_input=bf16  ⇒ dg dtype = {o4bf[4].dtype} "
          f"(期望 bfloat16) ... {'PASS' if good else 'FAIL'}")
    o4ng = chunk_gated_delta_rule_bwd_golden(
        **{kk: vv for kk, vv in i4.items()
           if kk not in ("use_gate_in_kernel", "g_input", "a_log", "dt_bias")})
    good = (o4ng[4].dtype == torch.float32 and o4ng[6] is None and o4ng[7] is None)
    ok &= good
    logging.info(f"  D14  非 gate 路径             ⇒ dg dtype = {o4ng[4].dtype} "
          f"(期望 float32，恒 fp32) ... {'PASS' if good else 'FAIL'}")
    o4b = chunk_gated_delta_rule_bwd_golden(**dict(i4, dt_bias=None))
    good = (o4b[7] is None and o4b[6] is not None)
    ok &= good
    logging.info(f"  D12  dt_bias=None ⇒ ddt_bias is None ... {'PASS' if good else 'FAIL'}")
    # D10：bf16 g_input 下，ddt_bias 必须等于「先 cast 回 bf16 再 sum」而非纯 fp32 sum
    i4c = dict(i4, g_input=i4["g_input"].to(torch.bfloat16))
    o4c = chunk_gated_delta_rule_bwd_golden(**i4c)
    # D14 之后返回的 dg 本身就是 bf16（= fla 的 ``dg.type_as(g_input)``），
    # 直接在它上面 sum 即与 fla ``dg.view(-1,H).sum(0)`` 逐位一致（fp32 累加器）。
    dg_c = o4c[4]                                                # [1,64,2] bf16
    ref_cast = dg_c.reshape(-1, 2).sum(0).to(i4["dt_bias"].dtype)
    good = bool(torch.allclose(o4c[7].float(), ref_cast.float(), atol=0, rtol=0))
    ok &= good
    logging.info(f"  D10  ddt_bias == sum(cast_to_g_input_dtype(dg)) ... "
          f"{'PASS' if good else 'FAIL'}")
    # 与「纯 fp32 sum」应有可观测差异（证明确实走了 cast 分支）
    gi32 = i4["g_input"].float()
    o4d = chunk_gated_delta_rule_bwd_golden(**dict(i4, g_input=gi32))
    d_cast_vs_fp32 = _relerr(o4c[7], o4d[7])
    logging.info(f"       (bf16-cast sum vs fp32 sum 的相对差 = {d_cast_vs_fp32:.3e}，"
          f"非零即证明 D10 生效)")
    # softplus 数值稳定性：大 |x| 下不得 NaN/Inf
    big = torch.tensor([-1e4, -80.0, 0.0, 80.0, 1e4], dtype=torch.float32, device=device)
    sp = _softplus_stable(big)
    good = bool(torch.isfinite(sp).all()) and float(sp[-1]) > 9e3
    ok &= good
    logging.info(f"  softplus 稳定式 @ x=±1e4: {sp.tolist()} ... {'PASS' if good else 'FAIL'}")

    # ------------------------------------------------------ [不支持路径 显式拒绝]
    logging.info("\n[9] 不支持路径 / 非法参数 显式拒绝")
    i9 = make_inputs(batch=1, seq_len=64, n_heads=1, d_head=32, d_value=32, device=device, seed=1)
    try:
        chunk_gated_delta_rule_bwd_golden(**dict(i9, cp_context=object()))
        logging.info("  cp_context   ... FAIL (未抛异常)")
        ok = False
    except NotImplementedError:
        logging.info("  cp_context   ... PASS (NotImplementedError)")
    for bad_bt in (16, 32, 128):
        try:
            chunk_gated_delta_rule_bwd_golden(**dict(i9, chunk_size=bad_bt))
            logging.info(f"  chunk_size={bad_bt:<4d} ... FAIL (未抛异常)")
            ok = False
        except ValueError:
            logging.info(f"  chunk_size={bad_bt:<4d} ... PASS (ValueError，D9 仅支持 64)")
    # varlen 要求 B == 1
    try:
        i9b = make_inputs(batch=2, seq_len=64, n_heads=1, d_head=32, d_value=32, device=device, seed=1)
        chunk_gated_delta_rule_bwd_golden(
            **dict(i9b, cu_seqlens=torch.tensor([0, 64], dtype=torch.int64, device=device)))
        logging.info("  varlen batch!=1  ... FAIL (未抛异常)")
        ok = False
    except ValueError:
        logging.info("  varlen batch!=1  ... PASS (ValueError)")

    # ----------------------------------------------------------- [泛化 case 验证]
    logging.info("\n[10] 泛化 case 验证（bt 固定 64）")
    for (b, t, h, hv, kk, vv) in [(1, 16, 1, 1, 64, 64), (1, 100, 1, 2, 32, 64),
                                  (3, 256, 4, 4, 64, 128)]:
        i = make_inputs(batch=b, seq_len=t, n_heads=h, d_head=kk, d_value=vv, hv=hv, device=device, seed=b * 7 + t)
        o = chunk_gated_delta_rule_bwd_golden(**i)
        good = all(torch.isfinite(x.float()).all() for x in o if x is not None)
        ok &= good
        logging.info(
            f"  batch={b}, seq_len={t}, n_heads={h}, hv={hv}, "
            f"d_head={kk}, d_value={vv} ... {'PASS' if good else 'FAIL'}"
        )

    # --------------------------------------------------------- [GVA / v_first 检查]
    logging.info("\n[11] GVA (hv>n_heads) / state_v_first(D8) 检查")
    i5 = make_inputs(batch=1, seq_len=64, n_heads=1, d_head=32, d_value=32, hv=2, device=device, seed=9)
    o5 = chunk_gated_delta_rule_bwd_golden(**i5)
    good = tuple(o5[0].shape) == (1, 64, 1, 32) and tuple(o5[1].shape) == (1, 64, 1, 32)
    ok &= good
    logging.info(f"  GVA hv=2, n_heads=1: dq shape={tuple(o5[0].shape)} ... {'PASS' if good else 'FAIL'}")
    i6 = dict(i5, initial_state=_tp(i5["initial_state"]).contiguous(),
              dht=_tp(i5["dht"]).contiguous(), state_v_first=True)
    o6 = chunk_gated_delta_rule_bwd_golden(**i6)
    err = _relerr(_tp(o6[5]), o5[5])
    good = err < 1e-5
    ok &= good
    logging.info(f"  state_v_first=True 与默认布局一致性 rel_err={err:.3e} ... "
          f"{'PASS' if good else 'FAIL'}")
    # varlen + state_v_first
    packed, per_seq = make_varlen_inputs([64, 128], n_heads=1, d_head=32, d_value=32, hv=2,
                                         dtype=torch.float32, device=device, seed=13)
    pv = dict(packed, initial_state=_tp(packed["initial_state"]).contiguous(),
              dht=_tp(packed["dht"]).contiguous(), state_v_first=True)
    ov = chunk_gated_delta_rule_bwd_golden(**pv)
    ob = chunk_gated_delta_rule_bwd_golden(**packed)
    err = _relerr(_tp(ov[5]), ob[5])
    good = err < 1e-5
    ok &= good
    logging.info(f"  varlen + state_v_first 一致性 rel_err={err:.3e} ... "
          f"{'PASS' if good else 'FAIL'}")

    # ---------------------------------------------------------- [bf16 A 的代价]
    logging.info("\n[12] 【信息项】传入 bf16 a_mat vs fp32 精确 a_mat 的输出差异")
    logging.info("     —— 量化 golden 「非绝对真值」的代价（D7）；此项不作为 PASS/FAIL 判据")
    ib = make_inputs(batch=1, seq_len=128, n_heads=2, d_head=64, d_value=64,
                     dtype=torch.float32, device=device, seed=42)
    a32 = recompute_a_from_forward(ib["k"], ib["g"], ib["beta"], chunk_size=64)
    a16 = a32.to(torch.bfloat16).float()
    o32 = chunk_gated_delta_rule_bwd_golden(**dict(ib, a_mat=a32))
    o16 = chunk_gated_delta_rule_bwd_golden(**dict(ib, a_mat=a16))
    logging.info("     " + "  ".join(f"{n}={_relerr(o16[j], o32[j]):.2e}"
                              for j, n in enumerate(out_names[:6])))

    logging.info("\n" + "=" * 76)
    logging.info("✅ 所有验证通过" if ok else "❌ 存在失败项")
    logging.info("=" * 76)
    return 0 if ok else 1


# ####################################################################################################
# ####################################################################################################
#
#   ALIGNED TORCH-GOLDEN BACKWARD  (moved here from gdr_bwd_impl.py)
#
#   `gdr_bwd_impl.py` holds ONLY the PyPTO kernel + its production wrapper; every
#   pure-torch reference path lives in this file.  What follows is the second, independent
#   golden used by `test_gdr_bwd.py`:
#
#     * SECTION A - shared math utilities (l2norm, chunk constants, forward recompute)
#     * SECTION B - pure-torch reference backward (autograd-verified)
#     * SECTION C - the Triton-aligned drop-in `chunk_gated_delta_rule_bwd_torch_golden_aligned`
#     * SECTION D - `detailed_tensor_compare`, the precision-test reporting helper
#
#   `_aligned_check_unsupported` / `_aligned_cast_outputs` are DELIBERATELY duplicated from
#   gdr_bwd_impl.py rather than imported: this module is an INDEPENDENT reference, and a
#   golden that imported the implementation's own dtype-cast contract could not catch a bug
#   in it.  Keeping the copies also keeps this file free of any `pypto` import.
#
#   NOTE: the block below uses `_LN2`; this module already defines `LN2` / `RCP_LN2`, so
#   `_LN2` is aliased to it rather than redefined.
#
# ####################################################################################################
# ####################################################################################################

_LN2 = LN2   # exp2(x) == exp(_LN2 * x): Triton's base-2 gate units

# ====================================================================================================
# SECTION 1 - shared math utilities (l2norm, chunk constants, forward recompute, gate recovery)
# ====================================================================================================


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6):
    x32 = x.to(torch.float32)
    rstd = torch.rsqrt((x32 * x32).sum(dim=-1) + eps)         # [...], no keepdim
    y = x32 * rstd[..., None]
    return y, rstd


def l2norm_bwd_chunk(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor):
    # y: [bt,D], rstd: [bt], dy: [bt,D]
    dot = (dy * y).sum(dim=-1)                                # [bt]
    dx = dy * rstd[:, None] - dot[:, None] * y * rstd[:, None]
    return dx


class ChunkConstants(NamedTuple):
    """make_chunk_constants 的 7 个返回值。"""
    eye: torch.Tensor
    m_le: torch.Tensor
    m_lt: torch.Tensor
    c_cum: torch.Tensor
    c_rcum: torch.Tensor
    ones_1l: torch.Tensor
    ones_1d: torch.Tensor


def make_chunk_constants(bt: int, d_model: int, device, dtype=torch.float32):
    idx = torch.arange(bt, device=device)
    eye = (idx[:, None] == idx[None, :]).to(dtype)             # [bt,bt]
    m_le = (idx[:, None] >= idx[None, :]).to(dtype)          # lower incl diag
    m_lt = m_le - eye                                          # strict lower
    c_cum = m_le                                              # prefix sum: y = c_cum @ x
    c_rcum = (idx[None, :] >= idx[:, None]).to(dtype)         # upper incl diag: suffix sum
    ones_1l = torch.ones(1, bt).to(dtype).to(device)                     # (1,L) ones
    ones_1d = torch.ones(1, d_model).to(dtype).to(device)
    return ChunkConstants(
        eye=eye, m_le=m_le, m_lt=m_lt, c_cum=c_cum, c_rcum=c_rcum,
        ones_1l=ones_1l, ones_1d=ones_1d)


def forward_ref(
    q: torch.Tensor,           # [B,T,H,K]
    k: torch.Tensor,           # [B,T,H,K]
    v: torch.Tensor,           # [B,T,H,V]
    g_raw: torch.Tensor,       # [B,T,H]
    beta: torch.Tensor,        # [B,T,H]
    initial_state: torch.Tensor,  # [B,H,K,V]
    bt: int,
    use_qk_l2norm_in_kernel: bool,
    l2_eps: float,
    eye: torch.Tensor, m_le: torch.Tensor, m_lt: torch.Tensor, c_cum: torch.Tensor,
):
    batch, seq_len, n_heads, d_head = q.shape
    d_value = v.shape[-1]


    num_chunks = seq_len // bt
    scale = 1.0 / math.sqrt(d_head)

    if use_qk_l2norm_in_kernel:
        q_norm, q_rstd = l2norm_fwd(q, eps=l2_eps)     # q_norm float32
        k_norm, k_rstd = l2norm_fwd(k, eps=l2_eps)
        q_used = q_norm
        k_used = k_norm
    else:
        q_used = q.to(torch.float32)
        k_used = k.to(torch.float32)
        q_rstd = None
        k_rstd = None

    v32 = v.to(torch.float32)
    beta32 = beta.to(torch.float32)
    g_raw32 = g_raw.to(torch.float32)

    out = torch.empty(batch, seq_len, n_heads, d_value, device=q.device, dtype=torch.float32)
    cache_a = torch.zeros((batch * n_heads * num_chunks * bt, bt), dtype=torch.float32, device=q.device)
    cache_s_before = torch.zeros((batch * n_heads * num_chunks * d_head, d_value), dtype=torch.float32, device=q.device)
    cache_w = torch.zeros((batch * n_heads * num_chunks * bt, d_value), dtype=torch.float32, device=q.device)
    cache_u = torch.zeros((batch * n_heads * num_chunks * bt, d_value), dtype=torch.float32, device=q.device)
    cache_v_new = torch.zeros((batch * n_heads * num_chunks * bt, d_value), dtype=torch.float32, device=q.device)

    for b in range(batch):
        for h in range(n_heads):
            state = initial_state[b, h].to(torch.float32)  # [K,V]
            for c in range(num_chunks):
                t0 = c * bt
                t1 = t0 + bt

                cache_idx = (b * n_heads + h) * num_chunks + c

                qc = q_used[b, t0:t1, h, :]            # [bt,K]
                kc = k_used[b, t0:t1, h, :]            # [bt,K]
                vc = v32[b, t0:t1, h, :]               # [bt,V]
                betac = beta32[b, t0:t1, h]            # [bt]
                gc_raw = g_raw32[b, t0:t1, h]          # [bt]

                # g_cum via matmul
                g_cum = c_cum @ gc_raw                 # [bt]
                eg = torch.exp(g_cum)                  # [bt]
                gl = g_cum[-1]

                diff = g_cum[:, None] - g_cum[None, :]
                decay = torch.exp(diff)                # [bt,bt]
                kkt = kc @ kc.t()                       # [bt,bt]
                l_mat = (betac[:, None] * kkt) * decay
                l_mat = l_mat * m_lt                             # strict-lower
                m_mat = eye + l_mat
                a_mat = torch.linalg.solve_triangular(m_mat, eye, upper=False)

                # u and w
                vb = vc * betac[:, None]
                kbg = kc * (betac[:, None] * eg[:, None])
                u = a_mat @ vb                               # [bt,V]
                w = a_mat @ kbg                              # [bt,K]

                # cache s_before, A, w, u, v_new
                cache_bt_start = cache_idx * bt
                cache_k_start = cache_idx * d_head
                cache_a[cache_bt_start:cache_bt_start + bt] = a_mat
                cache_s_before[cache_k_start:cache_k_start + d_head] = state
                cache_w[cache_bt_start:cache_bt_start + bt] = w
                cache_u[cache_bt_start:cache_bt_start + bt] = u

                # v_new
                v_prime = w @ state                          # [bt,V]
                v_new = u - v_prime
                cache_v_new[cache_bt_start:cache_bt_start + bt] = v_new

                # local attention output (mask by elementwise multiply)
                qk = qc @ kc.t()                         # [bt,bt]
                a_local = (qk * decay) * m_le            # keep lower incl diag
                o1 = eg[:, None] * (qc @ state)              # [bt,V]
                o2 = a_local @ v_new                     # [bt,V]
                out[b, t0:t1, h, :] = (o1 + o2) * scale

                # update state
                s = torch.exp(gl - g_cum)                # [bt]
                v_scaled = v_new * s[:, None]            # [bt,V]
                state = state * torch.exp(gl) + kc.t() @ v_scaled

    # Instead of recompute, capture final S in a separate pass during loop
    cache_final_state = torch.empty_like(initial_state, dtype=torch.float32, device=q.device)
    for b in range(batch):
        for h in range(n_heads):
            # final S is S_after_last = update of last chunk;
            # we can rebuild by taking last s_before and applying one forward update again,
            # but better: just re-run minimal recurrence with cached per chunk (v_new and g_cum recomputed).
            # For test sizes it's ok.
            state = initial_state[b, h].to(torch.float32)
            for c in range(num_chunks):
                cache_idx = (b * n_heads + h) * num_chunks + c
                t0 = c * bt
                t1 = t0 + bt
                kc = k_used[b, t0:t1, h, :]
                gc_raw = g_raw32[b, t0:t1, h]
                g_cum = c_cum @ gc_raw
                gl = g_cum[-1]
                cache_bt_start = cache_idx * bt
                v_new = cache_v_new[cache_bt_start:cache_bt_start + bt]
                s = torch.exp(gl - g_cum)
                v_scaled = v_new * s[:, None]
                state = state * torch.exp(gl) + kc.t() @ v_scaled
            cache_final_state[b, h] = state

    cache = {
        "a_mat": cache_a,
        "w": cache_w,
        "u": cache_u,
        "v_new": cache_v_new,
        "s_before": cache_s_before,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
        "q_norm": q_used.transpose(1, 2).contiguous().reshape(batch * n_heads * seq_len, d_head),
        "k_norm": k_used.transpose(1, 2).contiguous().reshape(batch * n_heads * seq_len, d_head),
        "q_rstd": q_rstd,     # [B,T,H] float32 or None
        "k_rstd": k_rstd,
        "scale": scale,
        "bt": bt,
        "final_state": cache_final_state
    }

    return out, cache_final_state, cache


def _recover_g_raw_nat(g, chunk_size):
    """g: [B, T, hv] Triton gate (within-chunk cumsum'd, RCP_LN2-scaled).
    Returns the natural per-token raw gate [B, T, hv] such that, within each chunk,
    cumsum(g_raw_nat) == ln2 * g  (== the exp2 exponent used by the kernels)."""
    batch, seq_len, hv = g.shape
    num_chunks = seq_len // chunk_size
    gc = (g.to(torch.float32) * _LN2).reshape(batch, num_chunks, chunk_size, hv)
    gd = gc.clone()
    gd[:, :, 1:, :] = gc[:, :, 1:, :] - gc[:, :, :-1, :]   # first difference within chunk
    return gd.reshape(batch, seq_len, hv)


# ====================================================================================================
# SECTION 2 - pure-torch REFERENCE backward (autograd-verified golden; used by the pytorch precision test)
# ====================================================================================================


class ChunkInputs(NamedTuple):
    """_slice_chunk_inputs 的 10 个返回值。"""
    qc: torch.Tensor
    kc: torch.Tensor
    vc: torch.Tensor
    betac: torch.Tensor
    gc_raw: torch.Tensor
    doc: torch.Tensor
    a_mat: torch.Tensor
    w: torch.Tensor
    s_before: torch.Tensor
    v_new: torch.Tensor


def _slice_chunk_inputs(
    q_used, k_used, v32, beta32, g_raw32, do,
    cache, b, h, c, t0, t1,
):
    qc = q_used[b, t0:t1, h, :]                          # [bt,K]
    kc = k_used[b, t0:t1, h, :]                          # [bt,K]
    vc = v32[b, t0:t1, h, :]                             # [bt,V]
    betac = beta32[b, t0:t1, h]                          # [bt]
    gc_raw = g_raw32[b, t0:t1, h]                        # [bt]
    doc = do[b, t0:t1, h, :].to(torch.float32)            # [bt,V]

    a_mat = cache["a_mat"][b, h, c]
    w = cache["w"][b, h, c]
    s_before = cache["s_before"][b, h, c]
    v_new = cache["v_new"][b, h, c]
    return ChunkInputs(
        qc=qc, kc=kc, vc=vc, betac=betac, gc_raw=gc_raw, doc=doc,
        a_mat=a_mat, w=w, s_before=s_before, v_new=v_new)


def _compute_g_and_decay(gc_raw, c_cum):
    g_cum = c_cum @ gc_raw                                # [bt]
    eg = torch.exp(g_cum)                                 # [bt]
    gl = g_cum[-1]                                        # scalar
    diff = g_cum[:, None] - g_cum[None, :]
    decay = torch.exp(diff)                               # [bt,bt]
    return g_cum, eg, gl, decay


def _local_attn_dv0(qc, kc, doc, decay, m_le, scale):
    qk = qc @ kc.t()                                      # [bt,bt]
    a_local = (qk * decay) * m_le                         # [bt,bt]
    dv0 = (a_local.t() @ doc) * scale                     # [bt,V]
    return qk, a_local, dv0


def _recurrence_backprop(kc, ds, gl, g_cum, dv0, qc, eg, doc, scale, w):
    ds_next = ds                                          # [K,V]
    s_tok = torch.exp(gl - g_cum)                          # [bt]
    dv_state = (kc @ ds_next) * s_tok[:, None]             # [bt,V]
    dv_total = dv_state + dv0                              # [bt,V]

    q_eff = qc * eg[:, None]                               # [bt,K]
    ds = ds_next * torch.exp(gl)
    ds = ds + (q_eff.t() @ doc) * scale
    ds = ds - (w.t() @ dv_total)
    return ds_next, s_tok, dv_total, ds, q_eff


def _compute_qkg_grads(
    device, bt, d_head,
    qc, kc, v_new, doc,
    g_cum, eg, gl, s_tok,
    ds_next, s_before,
    qk, decay, m_le, scale,
):
    dq_c = torch.empty((bt, d_head), device=device, dtype=torch.float32)
    dq_c.zero_()
    dk_c = torch.empty((bt, d_head), device=device, dtype=torch.float32)
    dk_c.zero_()
    dg_cum = torch.empty((bt,), device=device, dtype=torch.float32)
    dg_cum.zero_()

    dq1 = (doc @ s_before.t()) * eg[:, None] * scale
    dq_c += dq1
    dg_cum += (dq1 * qc).sum(dim=-1)

    v_scaled = v_new * s_tok[:, None]
    dk_state = v_scaled @ ds_next.t()
    dk_c += dk_state

    scalar = (kc * dk_state).sum(dim=-1)
    dg_cum -= scalar
    dg_cum[-1] += scalar.sum()
    dg_cum[-1] += torch.exp(gl) * (s_before * ds_next).sum()

    da_base = (doc @ v_new.t()) * m_le * scale
    dq_c += (da_base * decay) @ kc
    dk_c += (da_base * decay).t() @ qc
    a_base = (qk * decay) * m_le
    tmp = da_base * a_base
    dg_cum += tmp.sum(dim=-1) - tmp.sum(dim=-2)

    return dq_c, dk_c, dg_cum


def _wy_repr_fused_updates(
    vc, betac, kc, eg, du, dw,
    a_mat, m_lt, decay, dk_c, dg_cum,
):
    vb = vc * betac[:, None]
    kbg = kc * (betac[:, None] * eg[:, None])

    dvb = a_mat.t() @ du
    dkbg = a_mat.t() @ dw

    dv_c = dvb * betac[:, None]
    db_c = (dvb * vc).sum(dim=-1)

    dk_c += dkbg * (betac[:, None] * eg[:, None])
    db_c += (dkbg * (kc * eg[:, None])).sum(dim=-1)
    dg_cum += (dkbg * kbg).sum(dim=-1)

    da = dw @ kbg.t() + du @ vb.t()
    dl = -(a_mat.t() @ (da @ a_mat.t()))
    dl = dl * m_lt

    kkt = kc @ kc.t()
    e_mat = decay

    db_c += (dl * (kkt * e_mat)).sum(dim=-1)

    lmat = (betac[:, None] * kkt) * e_mat
    lmat = lmat * m_lt
    tmp2 = dl * lmat
    dg_cum += tmp2.sum(dim=-1) - tmp2.sum(dim=-2)

    mmat = dl * (betac[:, None] * e_mat)
    dk_c += (mmat + mmat.t()) @ kc

    return dv_c, db_c, dk_c, dg_cum


def _finalize_chunk_grads(
    c_rcum, dg_cum,
    use_qk_l2norm_in_kernel,
    q_used_chunk, k_used_chunk,
    q_rstd_chunk, k_rstd_chunk,
    dq_c, dk_c,
):
    dg_raw_c = c_rcum @ dg_cum

    if use_qk_l2norm_in_kernel:
        dq_raw_c = l2norm_bwd_chunk(q_used_chunk, q_rstd_chunk, dq_c)
        dk_raw_c = l2norm_bwd_chunk(k_used_chunk, k_rstd_chunk, dk_c)
    else:
        dq_raw_c = dq_c
        dk_raw_c = dk_c

    return dq_raw_c, dk_raw_c, dg_raw_c


def torch_golden_gated_delta_rule_backward_ref(
    q: torch.Tensor,           # [B,T,H,K]
    k: torch.Tensor,           # [B,T,H,K]
    v: torch.Tensor,           # [B,T,H,V]
    g_raw: torch.Tensor,       # [B,T,H]
    beta: torch.Tensor,        # [B,T,H]
    initial_state: torch.Tensor,  # [B,H,K,V]
    do: torch.Tensor,          # [B,T,H,V]
    dht: torch.Tensor,         # [B,H,K,V]
    cache: dict,
    bt: int,
    eye: torch.Tensor, m_le: torch.Tensor, m_lt: torch.Tensor, c_cum: torch.Tensor, c_rcum: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
    l2_eps: float,
):
    device = q.device
    batch, seq_len, n_heads, d_head = q.shape
    d_value = v.shape[-1]


    num_chunks = seq_len // bt
    scale = cache["scale"]

    # global grads (<=4D)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v, dtype=torch.float32)
    db = torch.empty_like(beta, dtype=torch.float32)
    dg_raw = torch.empty_like(g_raw, dtype=torch.float32)
    dh0 = torch.empty_like(initial_state, dtype=torch.float32)

    # init outputs deterministically (no torch.zeros)
    dq.zero_()
    dk.zero_()
    dv.zero_()
    db.zero_()
    dg_raw.zero_()
    dh0.zero_()

    # normalized inputs and rstd (<=4D)
    q_used = cache["q_norm"]  # [B,T,H,K] float32
    k_used = cache["k_norm"]

    q_rstd = cache["q_rstd"]  # [B,T,H] or None
    k_rstd = cache["k_rstd"]

    v32 = v.to(torch.float32)
    beta32 = beta.to(torch.float32)
    g_raw32 = g_raw.to(torch.float32)

    for b in range(batch):
        for h in range(n_heads):
            # [Tomo] For loop for Chunk is updated, becuase pypto gets wrong if reverse loop.
            for i in range(num_chunks):
                c = num_chunks - 1 - i
                if i == 0:  # i = 0 -> c = num_chunks-1 (last chunk)
                    ds = dht[b, h].to(torch.float32)

                t0 = c * bt
                t1 = t0 + bt

                # slice chunk inputs (all <=2D)
                qc, kc, vc, betac, gc_raw, doc, a_mat, w, s_before, v_new = _slice_chunk_inputs(
                    q_used, k_used, v32, beta32, g_raw32, do,
                    cache, b, h, c, t0, t1,
                )

                # ---- g_cum via matmul (no cumsum) ----
                g_cum, eg, gl, decay = _compute_g_and_decay(gc_raw, c_cum)

                # ===== (A) local attention grad wrt v_new: dv0 =====
                qk, a_local, dv0 = _local_attn_dv0(qc, kc, doc, decay, m_le, scale)

                # ===== (B) recurrence backprop (ds_next is current ds) =====
                ds_next, s_tok, dv_total, ds, q_eff = _recurrence_backprop(
                    kc, ds, gl, g_cum, dv0, qc, eg, doc, scale, w
                )

                # ===== (C) grads for q,k,g from outputs + state update + local attention =====
                dq_c, dk_c, dg_cum = _compute_qkg_grads(
                    device, bt, d_head,
                    qc, kc, v_new, doc,
                    g_cum, eg, gl, s_tok,
                    ds_next, s_before,
                    qk, decay, m_le, scale,
                )

                # ===== (D) v_new = u - wS  => dw, du =====
                dw = -(dv_total @ s_before.t())
                du = dv_total

                # ===== (E) prepare_wy_repr_bwd fused inside same loop =====
                dv_c, db_c, dk_c, dg_cum = _wy_repr_fused_updates(
                    vc, betac, kc, eg, du, dw,
                    a_mat, m_lt, decay, dk_c, dg_cum,
                )

                # ===== (F) dg_raw = reverse-cumsum(dg_cum) via matmul =====
                # ===== (G) l2norm backward inside chunk loop (as requested) =====
                dq_raw_c, dk_raw_c, dg_raw_c = _finalize_chunk_grads(
                    c_rcum, dg_cum,
                    use_qk_l2norm_in_kernel,
                    q_used[b, t0:t1, h, :], k_used[b, t0:t1, h, :],
                    q_rstd[b, t0:t1, h] if use_qk_l2norm_in_kernel else None,
                    k_rstd[b, t0:t1, h] if use_qk_l2norm_in_kernel else None,
                    dq_c, dk_c,
                )

                # ===== store into global grads =====
                dq[b, t0:t1, h, :] = dq_raw_c
                dk[b, t0:t1, h, :] = dk_raw_c
                dv[b, t0:t1, h, :] = dv_c
                db[b, t0:t1, h] = db_c
                dg_raw[b, t0:t1, h] = dg_raw_c

            # after all chunks, ds is dh0
            dh0[b, h] = ds

    return dq, dk, dv, db, dg_raw, dh0


def _aligned_check_unsupported(state_v_first, cu_seqlens, cp_context, use_gate_in_kernel):
    if state_v_first:
        raise NotImplementedError("aligned wrapper: state_v_first=True is not supported.")
    # D6 (DESIGN.md:272,311): `cu_seqlens` (varlen) IS supported -- it is the sole driver of
    # the packed path; `chunk_indices` is accepted for signature parity and ignored.
    if cp_context is not None:
        raise NotImplementedError("aligned wrapper: cp_context (context-parallel) is not supported.")
    if use_gate_in_kernel:
        raise NotImplementedError("aligned wrapper: use_gate_in_kernel=True is not supported.")


def _aligned_prepare(q, k, v, g, beta, scale, initial_state, chunk_size):
    """Host-side adapter for the TORCH-GOLDEN reference path only (it needs the Python
    forward cache).  The pypto path does NOT call this -- its forward recompute lives in
    the kernel (PASS-1/PASS-2).  Returns (g_raw_nat, cache, o_rec, ht_rec, consts, dims)."""
    batch, seq_len, n_heads, d_head = q.shape
    hv = v.shape[2]
    d_value = v.shape[-1]
    if hv != n_heads:
        raise NotImplementedError(f"aligned wrapper: GVA (hv={hv} != n_heads={n_heads}) is not supported.")
    device = q.device
    if abs(float(scale) - d_head ** -0.5) > 1e-6:
        raise NotImplementedError(f"torch-golden path: scale must be d_head**-0.5 (got {scale}).")

    g_raw_nat = _recover_g_raw_nat(g, chunk_size)
    consts = make_chunk_constants(
        chunk_size, d_head, device=device, dtype=torch.float32)

    o_rec, ht_rec, cache = forward_ref(
        q=q, k=k, v=v, g_raw=g_raw_nat, beta=beta, initial_state=initial_state,
        bt=chunk_size, use_qk_l2norm_in_kernel=False, l2_eps=1e-6,
        eye=consts.eye, m_le=consts.m_le, m_lt=consts.m_lt, c_cum=consts.c_cum,
    )
    dims = (batch, seq_len, n_heads, hv, d_head, d_value)
    return g_raw_nat, cache, o_rec, ht_rec, consts, dims


def _aligned_cast_outputs(dq, dk, dv, db, dg, dh0, q, k, v, beta, g, initial_state):
    """Cast to the REAL chunk_gated_delta_rule_bwd return dtypes and return the 8-tuple.

    fla contract, read off the allocation sites of the upstream kernels:
      dq, dk, dv -> q/k/v dtype   (fla chunk_o.py:729-730)
      db         -> `torch.empty_like(beta)`                       => beta.dtype
                    (fla/ops/gated_delta_rule/wy_fast.py:301)
      dg         -> `torch.empty_like(g) if g is not None else None`
                    => g.dtype, and None when g is None
                    (fla/ops/gated_delta_rule/wy_fast.py:300)
      dh0        -> `torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None`
                    => fp32 when initial_state is given, else None
                    (fla/ops/common/chunk_delta_h.py:700)

    NOTE: an earlier version of this docstring claimed `db, dg, dh0` are all fp32 (citing
    the comparison doc SS3.3).  That was wrong: only `dh0` is fp32, and only when
    `initial_state is not None`.  The `db.to(beta_raw)` at gated_delta_rule/chunk.py:389
    lives OUTSIDE the autograd Function and is not part of this function's contract.
    """
    return (
        dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
        db.to(beta.dtype),
        dg.to(g.dtype) if g is not None else None,
        dh0.to(torch.float32) if initial_state is not None else None,
        None, None,   # da_log, ddt_bias (use_gate_in_kernel=False)
    )


def chunk_gated_delta_rule_bwd_torch_golden_aligned(
    q, k, v, g, beta, a_mat, scale, initial_state, do, dht,
    state_v_first=False, cu_seqlens=None, cp_context=None, chunk_indices=None,
    use_gate_in_kernel=False, g_input=None, a_log=None, dt_bias=None, chunk_size=64,
):
    """Torch-golden REFERENCE path (host-side; NOT production), exact interface."""
    _aligned_check_unsupported(state_v_first, cu_seqlens, cp_context, use_gate_in_kernel)
    g_raw_nat, cache, _o, _ht, consts, dims = _aligned_prepare(
        q, k, v, g, beta, scale, initial_state, chunk_size)
    eye, m_le, m_lt, c_cum, c_rcum, ones_1l, ones_1d = consts
    batch, seq_len, n_heads, hv, d_head, d_value = dims
    num_chunks = seq_len // chunk_size

    cache_for_golden = {
        "a_mat": cache['a_mat'].reshape([batch, n_heads, num_chunks, chunk_size, chunk_size]),
        "w": cache['w'].reshape([batch, n_heads, num_chunks, chunk_size, d_value]),
        "u": cache['u'].reshape([batch, n_heads, num_chunks, chunk_size, d_value]),
        "v_new": cache['v_new'].reshape([batch, n_heads, num_chunks, chunk_size, d_value]),
        "s_before": cache['s_before'].reshape([batch, n_heads, num_chunks, d_head, d_value]),
        "use_qk_l2norm_in_kernel": False,
        "q_norm": cache['q_norm'].reshape(batch, n_heads, seq_len, d_head).permute(0, 2, 1, 3).contiguous(),
        "k_norm": cache['k_norm'].reshape(batch, n_heads, seq_len, d_head).permute(0, 2, 1, 3).contiguous(),
        "q_rstd": None,
        "k_rstd": None,
        "scale": cache['scale'],
        "bt": chunk_size,
        "final_state": cache['final_state'],
    }
    with torch.no_grad():
        dq, dk, dv, db, dg_raw, dh0 = torch_golden_gated_delta_rule_backward_ref(
            q=q, k=k, v=v, g_raw=g_raw_nat, beta=beta,
            initial_state=initial_state, do=do, dht=dht,
            cache=cache_for_golden, bt=chunk_size,
            eye=eye, m_le=m_le, m_lt=m_lt, c_cum=c_cum, c_rcum=c_rcum,
            use_qk_l2norm_in_kernel=False, l2_eps=1e-6,
        )
    dg = dg_raw   # dg already matches Triton's units (verified best-fit factor ~1.0, NOT ln2)
    return _aligned_cast_outputs(dq, dk, dv, db, dg, dh0, q, k, v, beta, g, initial_state)

# SECTION 5 - tensor comparison helper (used by the two precision tests)
# ==================================================================


def detailed_tensor_compare(tensor1, tensor2, tensor_name,
                            rtol=1e-3, atol=1e-3, verbose=True,
                            max_outliers_display=20):
    """
    Detailed tensor comparison, analyzing the proportion of elements that are out of tolerance, 
    and displaying specific information about those that exceed the tolerance.

    Args:
    tensor1: The first tensor.
    tensor2: The second tensor.
    rtol: Relative tolerance.
    atol: Absolute tolerance.
    verbose: Whether to print detailed information.
    max_outliers_display: Maximum number of out-of-tolerance elements to display.

    Returns:
    dict: A dictionary containing the comparison results.
    """
    # Ensure tensors are comparable
    t1, t2 = tensor1.cpu().float(), tensor2.cpu().float()

    # Calculate the difference
    diff = torch.abs(t1 - t2)
    relative_diff = diff / (torch.abs(t2) + 1e-8)

    # Tolerance Check
    tolerance_mask = diff <= atol + rtol * torch.abs(t2)
    out_of_tolerance_mask = ~tolerance_mask

    # Statistics
    total_elements = t1.numel()
    out_of_tolerance_count = out_of_tolerance_mask.sum().item()
    out_of_tolerance_ratio = out_of_tolerance_count / total_elements

    # Difference Statistics
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    std_diff = torch.std(diff).item()

    if out_of_tolerance_count > 0:
        out_of_tolerance_diff = diff[out_of_tolerance_mask]
        max_out_diff = torch.max(out_of_tolerance_diff).item()
        mean_out_diff = torch.mean(out_of_tolerance_diff).item()

        outlier_indices = torch.nonzero(out_of_tolerance_mask, as_tuple=True)
        outlier_values1 = t1[out_of_tolerance_mask]
        outlier_values2 = t2[out_of_tolerance_mask]
        outlier_diffs = diff[out_of_tolerance_mask]
        outlier_relative_diffs = relative_diff[out_of_tolerance_mask]

        sorted_indices = torch.argsort(outlier_diffs, descending=True)
        sorted_outlier_indices = tuple(ind[sorted_indices] for ind in outlier_indices)
        sorted_outlier_values1 = outlier_values1[sorted_indices]
        sorted_outlier_values2 = outlier_values2[sorted_indices]
        sorted_outlier_diffs = outlier_diffs[sorted_indices]
        sorted_outlier_relative_diffs = outlier_relative_diffs[sorted_indices]

    else:
        max_out_diff = 0.0
        mean_out_diff = 0.0
        sorted_outlier_indices = None
        sorted_outlier_values1 = None
        sorted_outlier_values2 = None
        sorted_outlier_diffs = None
        sorted_outlier_relative_diffs = None

    result = {
        'total_elements': total_elements,
        'out_of_tolerance_count': out_of_tolerance_count,
        'out_of_tolerance_ratio': out_of_tolerance_ratio,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'max_out_of_tolerance_diff': max_out_diff,
        'mean_out_of_tolerance_diff': mean_out_diff,
        'all_close': out_of_tolerance_count == 0,
        'tolerance_mask': tolerance_mask,
        'diff_tensor': diff,
        'outlier_indices': sorted_outlier_indices,
        'outlier_values1': sorted_outlier_values1,
        'outlier_values2': sorted_outlier_values2,
        'outlier_diffs': sorted_outlier_diffs,
        'outlier_relative_diffs': sorted_outlier_relative_diffs
    }

    if verbose:
        logging.info("\n" + "=" * 60)
        logging.info("📊 Tensor Detailed Comparison Report")
        logging.info(f"name: {tensor_name}")
        logging.info("=" * 60)
        logging.info(f"Total number of elements: {total_elements:,}")
        logging.info(f"Number of elements exceeding tolerance: {out_of_tolerance_count:,}")
        logging.info(f"Out of Tolerance Ratio: {out_of_tolerance_ratio:.6f} ({out_of_tolerance_ratio*100:.4f}%)")
        logging.info(f"Maximum difference: {max_diff:.6f}")
        logging.info(f"Average difference: {mean_diff:.6f}")
        logging.info(f"Difference Standard Deviation: {std_diff:.6f}")
        logging.info(f"Tolerance Settings: rtol={rtol}, atol={atol}")

        if out_of_tolerance_count > 0:
            logging.info(f"Maximum deviation exceeding tolerance: {max_out_diff:.6f}")
            logging.info(f"Average deviation exceeding tolerance: {mean_out_diff:.6f}")

            logging.info(
                f"\n🔍 Details of elements exceeding tolerance limits "
                f"(Before Displaying{min(max_outliers_display, out_of_tolerance_count)}):"
            )
            logging.info("-" * 80)
            logging.info(
                f"{'Index':<20} {'Tensor1 value':<15} {'Tensor2 value':<15} "
                f"{'Abs diff':<12} {'Rel diff':<12}"
            )
            logging.info("-" * 80)

            for i in range(min(max_outliers_display, out_of_tolerance_count)):
                idx_str = str(
                    tuple(sorted_outlier_indices[j][i].item()
                          for j in range(len(sorted_outlier_indices)))
                )
                logging.info(
                    f"{idx_str:<20} {sorted_outlier_values1[i].item():<15.6f} "
                    f"{sorted_outlier_values2[i].item():<15.6f} "
                    f"{sorted_outlier_diffs[i].item():<12.6f} "
                    f"{sorted_outlier_relative_diffs[i].item():<12.6f}"
                )

            if out_of_tolerance_count > max_outliers_display:
                logging.info(
                    f"... And also "
                    f"{out_of_tolerance_count - max_outliers_display} "
                    f"An element exceeding the tolerance is not displayed."
                )

        logging.info(f"\n✅ Tensor Matching: {result['all_close']}")
        logging.info("=" * 60)

    return result

logging.basicConfig(level=logging.INFO, format="%(message)s")
if __name__ == "__main__":
    import sys
    sys.exit(_validate())
