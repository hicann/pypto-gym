# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import torch

try:
    import torch_npu
    _HAS_NPU = torch.npu.is_available() and torch.npu.device_count() > 0
except ImportError as exc:
    raise ImportError(
        "torch_npu is not installed. Please install it first:\n"
        "  pip install torch_npu"
    ) from exc

_DEVICE = None
_HAS_NPU = False


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


@dataclass
class GdrFwdOptions:
    """gdr_fwd_golden / gdr_fwd_naive 的公共配置参数。"""
    scale: float | None = None
    initial_state: torch.Tensor | None = None
    output_final_state: bool = False
    use_qk_l2norm_in_kernel: bool = False
    cu_seqlens: torch.LongTensor | None = None
    chunk_size: int = 64
    emulate_bf16: bool = False


@dataclass
class BuildCaseConfig:
    """_build_case 的形状 + 配置参数。"""
    b: int
    t: int
    h: int
    hv: int
    k_dim: int
    v_dim: int
    device: torch.device
    seed: int = 42
    g_range: tuple = (-0.10, -0.001)
    with_state: bool = False
    n_seq: int | None = None


# ─────────────────────────────────────────────
# 公共工具
# ─────────────────────────────────────────────

def _bf16_round(x: torch.Tensor, enable: bool) -> torch.Tensor:
    """bf16 舍入模拟：先降 bf16 再回 fp32。

    等价于 NPU Cube「bf16 操作数 + fp32 累加」：操作数被量化，乘加过程仍是 fp32。
    ``enable=False`` 时原样返回（fp32 数学真值路径）。
    """
    if not enable:
        return x
    return x.to(torch.bfloat16).to(torch.float32)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对齐 B2 的 ``l2norm_fwd``（fla/modules/l2norm.py:100-102, 140-142）。

    eps 是**加性**的，且加在**平方和**上：``x / sqrt(Σx² + 1e-6)``。
    计算走 fp32，输出跟随输入 dtype。
    """
    xf = x.float()
    rstd = torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + eps)
    return (xf * rstd).to(x.dtype)


def _expand_qk_to_hv(x: torch.Tensor, hv: int) -> torch.Tensor:
    """GVA 头广播：``[B, T, H, D] -> [B, T, HV, D]``，映射 ``h = hv // (HV // H)``。

    B2 逐处写作 ``i_h // (HV // H)``（wy_fast.py:87、chunk_delta_h.py:109、
    chunk_o.py:80-81、chunk_fwd.py:88）。等价于沿头维 ``repeat_interleave``，
    **不是** ``repeat/tile``。
    """
    h = x.shape[2]
    if h == hv:
        return x
    return torch.repeat_interleave(x, hv // h, dim=2)


def _check_args(q, k, v, g, beta, opts: GdrFwdOptions):
    """对齐 B2 公开入口的校验（chunk.py:517-556）。"""
    initial_state = opts.initial_state
    cu_seqlens = opts.cu_seqlens
    chunk_size = opts.chunk_size
    if q.shape[2] != k.shape[2]:
        raise ValueError(
            f"q and k must have the same number of heads, "
            f"but got q.shape[2]={q.shape[2]} and k.shape[2]={k.shape[2]}"
        )
    h, hv = q.shape[2], v.shape[2]
    if hv % h != 0:
        raise ValueError(
            f"For GVA, num_v_heads (HV={hv}) must be evenly divisible by "
            f"num_heads (H={h}), but got HV % H = {hv % h}"
        )
    if chunk_size not in (16, 32, 64, 128):
        raise ValueError(
            f"`chunk_size` must be 16, 32, 64, or 128 for Gated Delta Rule, "
            f"got {chunk_size}."
        )
    if q.shape[-1] != v.shape[-1]:
        raise NotImplementedError(
            f"gdr_fwd 仅支持 K == V，收到 K={q.shape[-1]}, V={v.shape[-1]}。"
        )
    if g.shape[2] != hv or beta.shape[2] != hv:
        raise ValueError(
            f"g/beta 的头数必须等于 HV={hv}，收到 g={g.shape[2]}, beta={beta.shape[2]}"
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                f"when using `cu_seqlens`. Please flatten variable-length inputs "
                f"before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number "
                f"of input sequences, i.e., {len(cu_seqlens) - 1} rather than "
                f"{initial_state.shape[0]}."
            )


def _segments(cu_seqlens, b: int, t: int):
    """返回 ``[(batch_idx, start, end), ...]``，每个元素是一条独立序列。"""
    if cu_seqlens is None:
        return [(i, 0, t) for i in range(b)]
    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    return [(0, int(cu[i]), int(cu[i + 1])) for i in range(len(cu) - 1)]


def _inv_unit_lower_forward(a: torch.Tensor) -> torch.Tensor:
    """``(I + a)^{-1}`` 的逐行前向替换，**不建图**（调用方负责 autograd）。

    算法 = B2 的逐行前向替换（chunk_fwd.py:221-250 / solve_tril.py:76-84）在
    叶子块大小 1 上的展开：``X[i, :] = e_i + Σ_{j<i} (-a)[i, j] · X[j, :]``。
    代码里 ``x`` 保存的是 ``X - I``，故递推写成 ``x[i, :] = neg[i, :] + neg[i, :i] @ x[:i, :]``。
    精确逆（``a`` 幂零），不是 Neumann 截断。
    """
    n = a.shape[-1]
    neg = -a
    x = neg.clone()
    for i in range(1, n):
        # neg[..., i, :i] @ x[..., :i, :]  ->  [..., 1, n]
        upd = torch.matmul(neg[..., i:i + 1, :i], x[..., :i, :])
        x[..., i:i + 1, :] = neg[..., i:i + 1, :] + upd
    eye = torch.eye(n, dtype=a.dtype, device=a.device)
    return x + eye


class _InvUnitLower(torch.autograd.Function):
    """``(I + a)^{-1}``，带**解析 VJP**。

    为什么不让 autograd 直接穿过上面那个逐行循环：
      * 就地写 ``x[..., i, :] = ...`` 会顶掉上一轮读到的 ``x[..., :i, :]`` 的版本号，
        autograd 直接报 "modified by an inplace operation"；
      * 改成攒 list + ``cat`` 虽然能跑，但每步都要给 matmul 存下一份增长中的前缀，
        显存 O(n²·batch)：T=32k 档实测约 17 GB，T=128k 档 ~68 GB 直接 OOM。

    解析 VJP 则是 O(1) 额外显存、两次 matmul。设 ``M = I + a``、``Y = M^{-1}``，
    由 ``dY = -Y (dM) Y`` 得
        ``∂L/∂M = -Yᵀ (∂L/∂Y) Yᵀ``，而 ``∂M = ∂a``。
    本函数的定义域是**严格下三角** ``a``（见 Args），故梯度投影回该子空间 —— 取
    ``tril(-1)``。上游 ``_chunk_core`` 传进来的 ``a_mat`` 本就是掩过的，对角/上三角
    的梯度会被那层掩码清零，与这里的投影一致。

    Args:
        a: ``[..., n, n]``，严格下三角（对角及以上为 0；非零也会被 ``tril(-1)`` 掩掉）。
    Returns:
        ``[..., n, n]``，单位下三角。
    """

    @staticmethod
    def forward(ctx, a):
        y = _inv_unit_lower_forward(a.tril(-1))
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (y,) = ctx.saved_tensors
        yt = y.transpose(-1, -2)
        return torch.matmul(torch.matmul(yt, grad_out), yt).neg().tril(-1)


def _inv_unit_lower(a: torch.Tensor) -> torch.Tensor:
    """给定**严格**下三角 ``a``，返回精确逆 ``(I + a)^{-1}``（同为单位下三角）。

    数值与逐行前向替换逐位一致；梯度走 :class:`_InvUnitLower` 的解析 VJP。

    Args:
        a: ``[..., n, n]`` fp32，严格下三角（对角及以上为 0）。
    Returns:
        ``[..., n, n]`` fp32。
    """
    return _InvUnitLower.apply(a)


# ─────────────────────────────────────────────
# chunk-parallel 核心
# ─────────────────────────────────────────────

def _chunk_core(q, k, v, g, beta, scale, h0, bt, emulate_bf16):
    """单批（等长）序列的 chunk-parallel 前向。

    Args:
        q, k: ``[Nb, L, HV, K]`` fp32（已按 GVA 展开到 HV 头）
        v:    ``[Nb, L, HV, V]`` fp32
        g:    ``[Nb, L, HV]``    fp32，**log 空间**、**未 cumsum**
        beta: ``[Nb, L, HV]``    fp32
        h0:   ``[Nb, HV, K, V]`` fp32 或 None
        bt:   chunk_size
    Returns:
        ``o [Nb, L, HV, V]`` fp32, ``S [Nb, HV, K, V]`` fp32
    """
    dev = q.device
    nb, ll, hv, kk_dim = q.shape
    v_dim = v.shape[-1]
    nt = (ll + bt - 1) // bt
    pad = nt * bt - ll

    def _pad_t(x):
        """沿时间轴（dim=1）尾部零填充到 nt*bt，并保证 contiguous。"""
        if pad == 0:
            return x.contiguous()
        shape = list(x.shape)
        shape[1] = pad
        return torch.cat([x, torch.zeros(shape, dtype=x.dtype, device=dev)], dim=1).contiguous()

    # [Nb, NT*BT, HV, D] -> [Nb, NT, HV, BT, D]
    def _to_chunks(x):
        d = x.shape[-1]
        return _pad_t(x).view(nb, nt, bt, hv, d).permute(0, 1, 3, 2, 4).contiguous()

    qc = _to_chunks(q)                      # [Nb, NT, HV, BT, K]
    kc = _to_chunks(k)                      # [Nb, NT, HV, BT, K]
    vc = _to_chunks(v)                      # [Nb, NT, HV, BT, V]
    gc_raw = _pad_t(g.unsqueeze(-1)).view(nb, nt, bt, hv).permute(0, 1, 3, 2).contiguous()
    bc = _pad_t(beta.unsqueeze(-1)).view(nb, nt, bt, hv).permute(0, 1, 3, 2).contiguous()

    # ---- 尾块有效性掩码 m_t[i, t] = (i*BT + t) < L  （MATH.md §11）----
    t_abs = (torch.arange(nt, device=dev).unsqueeze(1) * bt
             + torch.arange(bt, device=dev).unsqueeze(0))          # [NT, BT]
    m_t = t_abs < ll                                                # [NT, BT] bool
    m_b = m_t.view(1, nt, 1, bt)                                    # 广播到 [Nb,NT,HV,BT]

    # ---- 步骤 2：chunk-local inclusive 前缀和（MATH.md §3）----
    # golden 用自然对数域的 exp，不用 exp2(RCP_LN2·γ)（SPEC §4.1）。
    # 越界行置 0，与 B2 的 boundary_check 读 0 语义一致。
    gcum = torch.cumsum(gc_raw, dim=-1)
    gcum = torch.where(m_b, gcum, torch.zeros((), dtype=gcum.dtype, device=dev))

    # ---- 步骤 3：A 矩阵（严格下三角）（MATH.md §4）----
    gdiff = gcum.unsqueeze(-1) - gcum.unsqueeze(-2)                 # [Nb,NT,HV,BT,BT]
    tri_strict = torch.tril(
        torch.ones(bt, bt, dtype=torch.bool, device=dev), diagonal=-1)
    m_row = m_t.view(1, nt, 1, bt, 1)
    m_col = m_t.view(1, nt, 1, 1, bt)
    m_strict = tri_strict.view(1, 1, 1, bt, bt) & m_row & m_col
    zero = torch.zeros((), dtype=gcum.dtype, device=dev)

    # k @ kᵀ：操作数 dtype 由调用方决定（kernel 侧是 bf16），累加恒 fp32
    kkt = torch.matmul(kc, torch.transpose(kc, -2, -1))
    # 先 where 再 exp：越界项的指数参数恒为 0，杜绝 0*inf -> NaN（chunk_fwd.py:180-181）
    dec_lo = torch.exp(torch.where(m_strict, gdiff, zero))
    a_mat = torch.where(m_strict, kkt * dec_lo * bc.unsqueeze(-1), zero)

    # ---- 步骤 4：精确求逆，结果落 bf16（MATH.md §5）----
    tinv = _bf16_round(_inv_unit_lower(a_mat), emulate_bf16)        # [Nb,NT,HV,BT,BT]

    # ---- 步骤 5：WY 表示 w / u（MATH.md §6）----
    exp_g = torch.exp(gcum)                                         # [Nb,NT,HV,BT]
    vb = _bf16_round(bc.unsqueeze(-1) * vc, emulate_bf16)
    kbg = _bf16_round(bc.unsqueeze(-1) * exp_g.unsqueeze(-1) * kc, emulate_bf16)
    u_all = _bf16_round(torch.matmul(tinv, vb), emulate_bf16)       # [Nb,NT,HV,BT,V]
    w_all = _bf16_round(torch.matmul(tinv, kbg), emulate_bf16)      # [Nb,NT,HV,BT,K]

    # ---- 步骤 7(a) 的 P 掩码（含对角）（MATH.md §8）----
    tri_incl = torch.tril(torch.ones(bt, bt, dtype=torch.bool, device=dev), diagonal=0)
    m_incl = tri_incl.view(1, 1, 1, bt, bt) & m_row & m_col
    dec_incl = torch.exp(torch.where(m_incl, gdiff, zero))

    # ---- 步骤 6 + 7：chunk 间递推（串行）+ 输出（MATH.md §7/§8）----
    if h0 is None:
        state = torch.zeros(nb, hv, kk_dim, v_dim, dtype=torch.float32, device=dev)
    else:
        state = h0.float().clone()

    o_chunks = []
    for i in range(nt):
        li = min((i + 1) * bt, ll) - i * bt          # 本 chunk 有效行数 L_i
        # h[i] 保存的是【进入本 chunk 之前】的状态，并以 bf16 落盘 / 参与 matmul
        h_bf = _bf16_round(state, emulate_bf16)

        v_new = _bf16_round(u_all[:, i] - torch.matmul(w_all[:, i], h_bf), emulate_bf16)

        # (a)(c) 输出：注意 scale 施加在最末端，P 在【乘 scale 之前】落 bf16
        p_mat = torch.matmul(qc[:, i], torch.transpose(kc[:, i], -2, -1))
        p_mat = torch.where(m_incl[:, i], p_mat * dec_incl[:, i], zero)
        o_inter = torch.matmul(qc[:, i], h_bf) * exp_g[:, i].unsqueeze(-1)
        o_i = scale * o_inter + scale * torch.matmul(_bf16_round(p_mat, emulate_bf16), v_new)
        o_chunks.append(o_i)

        # (c)(d)(e)(f) state 递推；γ_last 取第 L_i-1 行（chunk_delta_h.py:206）
        g_last = gcum[:, i, :, li - 1]                                  # [Nb, HV]
        dec_v = torch.where(
            m_t[i].view(1, 1, bt),
            torch.exp(g_last.unsqueeze(-1) - gcum[:, i]),
            zero,
        )                                                              # [Nb,HV,BT]
        vd = _bf16_round(v_new * dec_v.unsqueeze(-1), emulate_bf16)
        state = (torch.exp(g_last).unsqueeze(-1).unsqueeze(-1) * state
                 + torch.matmul(torch.transpose(kc[:, i], -2, -1), vd))

    # [Nb, NT, HV, BT, V] -> [Nb, L, HV, V]
    o = torch.stack(o_chunks, dim=1).permute(0, 1, 3, 2, 4).reshape(nb, nt * bt, hv, v_dim)
    return o[:, :ll], state


def _naive_core(q, k, v, g, beta, scale, h0):
    """单批（等长）序列的朴素逐时间步递推（对齐 naive.py:50-59）。

    ``h = h·exp(g_t)``; ``u_t = beta_t·(v_t - hᵀk_t)``; ``h += k_t ⊗ u_t``;
    ``o_t = scale · qᵀ_t h``（用**更新后**的 h）。全程 fp32。
    """
    dev = q.device
    nb, ll, hv, k_dim = q.shape
    v_dim = v.shape[-1]
    if h0 is None:
        state = torch.zeros(nb, hv, k_dim, v_dim, dtype=torch.float32, device=dev)
    else:
        state = h0.float().clone()

    outs = []
    for t in range(ll):
        state = state * torch.exp(g[:, t]).unsqueeze(-1).unsqueeze(-1)   # [Nb,HV,1,1]
        k_t = k[:, t]                                                     # [Nb,HV,K]
        hk = (state * k_t.unsqueeze(-1)).sum(dim=-2)                      # [Nb,HV,V] = hᵀ k_t
        u_t = beta[:, t].unsqueeze(-1) * (v[:, t] - hk)                   # [Nb,HV,V]
        state = state + k_t.unsqueeze(-1) * u_t.unsqueeze(-2)             # 外积
        outs.append(scale * (state * q[:, t].unsqueeze(-1)).sum(dim=-2))  # [Nb,HV,V]
    return torch.stack(outs, dim=1), state


# ─────────────────────────────────────────────
# 对外入口
# ─────────────────────────────────────────────

def _run(core, q, k, v, g, beta, opts: GdrFwdOptions, core_kwargs):
    """``gdr_fwd_golden`` / ``gdr_fwd_naive`` 的共同外壳：校验 / L2norm / GVA / varlen。"""
    device = _get_device()
    q, k, v = q.to(device), k.to(device), v.to(device)
    g, beta = g.to(device), beta.to(device)
    initial_state = opts.initial_state.to(device) if opts.initial_state is not None else None
    cu_seqlens = opts.cu_seqlens.to(device) if opts.cu_seqlens is not None else None

    _check_args(q, k, v, g, beta, opts)

    b, t, h, k_dim = q.shape
    hv, v_dim = v.shape[2], v.shape[-1]
    out_dtype = q.dtype

    # 步骤 1：scale 默认值（chunk.py:565-566），不得硬编码
    scale = opts.scale if opts.scale is not None else k_dim ** -0.5
    scale = float(scale)

    # 步骤 0：可选 L2 归一化（chunk.py:280-283），fp32 计算、保持 dtype
    if opts.use_qk_l2norm_in_kernel:
        q = _l2norm(q)
        k = _l2norm(k)

    # GVA 头广播 + 全程 fp32
    qf = _expand_qk_to_hv(q, hv).float()
    kf = _expand_qk_to_hv(k, hv).float()
    vf, gf, betaf = v.float(), g.float(), beta.float()

    if cu_seqlens is None:
        # 等长：所有 batch 一起跑，N == B
        o, final = core(qf, kf, vf, gf, betaf, scale, initial_state, **core_kwargs)
    else:
        # varlen：逐段独立跑，状态**不跨段**（chunk_delta_h.py:79-83）
        segs = _segments(cu_seqlens, b, t)
        o = torch.zeros(b, t, hv, v_dim, dtype=torch.float32, device=device)
        final = torch.zeros(len(segs), hv, k_dim, v_dim, dtype=torch.float32, device=device)
        for n, (bi, s0, s1) in enumerate(segs):
            h0 = None if initial_state is None else initial_state[n:n + 1]
            o_b, s_b = core(
                qf[bi:bi + 1, s0:s1], kf[bi:bi + 1, s0:s1], vf[bi:bi + 1, s0:s1],
                gf[bi:bi + 1, s0:s1], betaf[bi:bi + 1, s0:s1],
                scale, h0, **core_kwargs,
            )
            o[bi, s0:s1] = o_b[0]
            final[n] = s_b[0]

    # o 跟随 q 的 dtype（chunk.py:335）；emulate 模式下先落一次 bf16
    o = _bf16_round(o, opts.emulate_bf16).to(out_dtype)
    return o, (final if opts.output_final_state else None)


def gdr_fwd_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    emulate_bf16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Gated Delta Rule 前向 golden（chunk-parallel），对齐 B2。

    Args:
        q: ``[B, T, H, K]``。``use_qk_l2norm_in_kernel=False`` 时要求上游已 L2 归一化。
        k: ``[B, T, H, K]``。
        v: ``[B, T, HV, V]``。``HV > H`` 触发 GVA，要求 ``HV % H == 0``。
        g: ``[B, T, HV]``，**log 空间**遗忘门（已是 log 值，内部只做 chunk-local cumsum）。
        beta: ``[B, T, HV]``，已在 post-sigmoid 空间。
        scale: q 的缩放；``None`` → ``K ** -0.5``。
        initial_state: ``[N, HV, K, V]`` fp32，**K 在 V 前**（state_v_first=False）。
        output_final_state: 是否返回 ``final_state``。
        use_qk_l2norm_in_kernel: 是否在内部对 q/k 做 L2 归一化（eps=1e-6，加性）。
        cu_seqlens: ``[N+1]`` int64，varlen 累积长度；提供时要求 ``B == 1``。
        chunk_size: ∈ {16, 32, 64}，默认 64。
        emulate_bf16: 见模块 docstring。

    Returns:
        ``(o, final_state)``：``o`` 形如 ``[B, T, HV, V]`` 且 dtype 跟随 ``q``；
        ``final_state`` 形如 ``[N, HV, K, V]`` fp32（``output_final_state=False`` 时为 ``None``）。

    Example:
        >>> o, ht = gdr_fwd_golden(q, k, v, g, beta, output_final_state=True)
    """
    opts = GdrFwdOptions(scale, initial_state, output_final_state,
                         use_qk_l2norm_in_kernel, cu_seqlens, chunk_size, emulate_bf16)
    return _run(
        _chunk_core, q, k, v, g, beta, opts,
        core_kwargs={"bt": chunk_size, "emulate_bf16": emulate_bf16},
    )


def gdr_fwd_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    emulate_bf16: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """朴素逐时间步递推参考（交叉验证用），签名与 ``gdr_fwd_golden`` 一致。

    ``chunk_size`` 仅用于参数校验（递推与分块无关）；``emulate_bf16`` 仅作用于
    输出的最终舍入 —— 本实现恒为 fp32 数学真值，不模拟中间 bf16 路径。
    """
    opts = GdrFwdOptions(scale, initial_state, output_final_state,
                         use_qk_l2norm_in_kernel, cu_seqlens, chunk_size, emulate_bf16)
    return _run(
        _naive_core, q, k, v, g, beta, opts,
        core_kwargs={},
    )


# ==========================================
# 输入构造（供 profiling --factory 与 _validate 共用）
# ==========================================

def _build_case(config: BuildCaseConfig):
    """构造语义合法的一组输入（结构约束型算子）。

    约束来源：
      - ``q`` / ``k``：``use_qk_l2norm_in_kernel=False`` 要求上游已 L2 归一化 → ``F.normalize``
      - ``g``：**log 空间**遗忘门，必须 ``<= 0`` → 负区间均匀采样
      - ``beta``：post-sigmoid 空间 → ``(0, 1)``
      - ``initial_state``：``[N, HV, K, V]`` fp32，K 在 V 前
    """
    gen = torch.Generator(device="cpu").manual_seed(config.seed)

    def _rn(*shape):
        return torch.randn(*shape, generator=gen)

    q = _rn(config.b, config.t, config.h, config.k_dim)
    q = q / q.norm(dim=-1, keepdim=True)
    k = _rn(config.b, config.t, config.h, config.k_dim)
    k = k / k.norm(dim=-1, keepdim=True)
    v = _rn(config.b, config.t, config.hv, config.v_dim) * 0.5
    g = torch.empty(config.b, config.t, config.hv).uniform_(config.g_range[0], config.g_range[1],
                                                            generator=gen)
    beta = torch.rand(config.b, config.t, config.hv, generator=gen)

    args = [x.to(config.device) for x in (q, k, v, g, beta)]
    h0 = None
    if config.with_state:
        n = config.n_seq if config.n_seq is not None else config.b
        h0 = (_rn(n, config.hv, config.k_dim, config.v_dim) * 0.1).to(config.device).float()
    return args, h0


def _make_inputs(device):
    """构造 SPEC §6 的全部 P0 典型输入。

    Returns:
        ``[(case_name, args_list, kwargs_dict), ...]``
        args_list 按 ``gdr_fwd_golden`` 签名顺序给出 ``[q, k, v, g, beta]``。
    """
    cases = []

    # base:   B=2 T=1024 H=4  HV=4 K=V=128 BT=64  对齐序列，无 GQA
    args, _ = _build_case(BuildCaseConfig(2, 1024, 4, 4, 128, 128, device, seed=0))
    cases.append(("base", args, {"chunk_size": 64}))

    # tail:   B=2 T=1000 ...  尾块非对齐 (1000 = 15*64 + 40)
    args, _ = _build_case(BuildCaseConfig(2, 1000, 4, 4, 128, 128, device, seed=1))
    cases.append(("tail", args, {"chunk_size": 64}))

    args, _ = _build_case(BuildCaseConfig(2, 512, 4, 8, 128, 128, device, seed=2))
    cases.append(("gva", args, {"chunk_size": 64}))

    # varlen: B=1 T=1536 cu_seqlens=[0,300,812,1536]  段长均非 64 对齐
    args, _ = _build_case(BuildCaseConfig(1, 1536, 4, 4, 128, 128, device, seed=3))
    cu = torch.tensor([0, 300, 812, 1536], dtype=torch.long, device=device)
    cases.append(("varlen", args, {"chunk_size": 64, "cu_seqlens": cu}))

    # with_state: B=2 T=512 带 initial_state + output_final_state
    args, h0 = _build_case(BuildCaseConfig(2, 512, 4, 4, 128, 128, device, seed=4,
                           g_range=(-0.01, -0.0001), with_state=True))
    cases.append(("with_state", args,
                  {"chunk_size": 64, "initial_state": h0, "output_final_state": True}))

    return cases


# ==========================================
# 验证
# ==========================================

def _stats(a, b):
    """返回 (max_abs, l2_rel, max_rel)。

    ``l2_rel = ‖a-b‖₂ / ‖b‖₂`` 是主指标。``max_rel`` 只在**显著元素**
    （``|b| > 0.1·max|b|``）上统计 —— 近似抵消处的小元素相对误差天然爆表，
    统计它们没有信息量。
    """
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    max_abs = diff.max().item()
    denom = b.norm().item()
    l2_rel = (diff.norm().item() / denom) if denom > 0 else 0.0
    thr = b.abs().max().item() * 0.1
    mask = b.abs() > max(thr, 1e-12)
    max_rel = (diff[mask] / b.abs()[mask]).max().item() if mask.any() else 0.0
    return max_abs, l2_rel, max_rel


def _finite(*tensors):
    return all(torch.isfinite(x).all().item() for x in tensors if x is not None)


def _validate():
    """自动验证：chunk vs naive 交叉验证、尾块/varlen/GVA/state、emulate_bf16、值域。"""
    device = _get_device()
    logging.info("=" * 74)
    logging.info("gdr_fwd_golden 验证报告")
    logging.info("=" * 74)
    logging.info(f"Device: {device}   (TILE_FWK_DEVICE_ID={os.environ.get('TILE_FWK_DEVICE_ID', '0')})")
    logging.info("判据: golden(chunk, fp32) vs golden(naive, fp32) —— SPEC §7 要求 rtol=1e-5 级别")

    failures = []
    rtol_val, atol_val = 1e-5, 1e-5

    # ---------- 1. 典型 case：chunk vs naive ----------
    configs = [
        ("base       P0", 2, 1024, 4, 4, 128, 128, 64, {}),
        ("tail       P0", 2, 1000, 4, 4, 128, 128, 64, {}),
        ("gva        P0", 2, 512, 4, 8, 128, 128, 64, {}),
        ("with_state P0", 2, 512, 4, 4, 128, 128, 64, {"state": True}),
        ("l2norm     P1", 2, 512, 4, 4, 128, 128, 64, {"l2": True}),
        ("bt16       P1", 2, 512, 4, 4, 128, 128, 16, {}),
        ("bt32       P1", 2, 512, 4, 4, 128, 128, 32, {}),
        ("scale=0.25 P1", 2, 256, 4, 4, 128, 128, 64, {"scale": 0.25}),
    ]

    logging.info("\n[典型 case 验证 — chunk(fp32) vs naive(fp32)]")
    for ci, (name, b, t, h, hv, kd, vd, bt, extra) in enumerate(configs):
        want_state = extra.get("state", False)
        args, h0 = _build_case(BuildCaseConfig(b, t, h, hv, kd, vd, device, seed=100 + ci,
                               g_range=(-0.01, -0.0001) if want_state else (-0.10, -0.001),
                               with_state=want_state))
        kw = {"chunk_size": bt}
        if want_state:
            kw.update(initial_state=h0, output_final_state=True)
        if extra.get("l2"):
            kw["use_qk_l2norm_in_kernel"] = True
        if "scale" in extra:
            kw["scale"] = extra["scale"]

        oc, sc = gdr_fwd_golden(*args, **kw)
        on, sn = gdr_fwd_naive(*args, **kw)
        ma, l2r, mr = _stats(oc, on)
        ok = _finite(oc, on) and torch.allclose(oc.float(), on.float(), rtol=rtol_val, atol=atol_val)
        line = (f"  {name}  B={b} T={t} H={h} HV={hv} D={kd} BT={bt}  "
                f"o: max_abs={ma:.3e} l2_rel={l2r:.3e} max_rel={mr:.3e}")
        if want_state:
            sma, sl2, smr = _stats(sc, sn)
            ok = ok and torch.allclose(sc.float(), sn.float(), rtol=rtol_val, atol=atol_val)
            line += f"\n{' ' * 4}state: max_abs={sma:.3e} l2_rel={sl2:.3e} max_rel={smr:.3e}"
        logging.info(f"{line}  ... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)

    # ---------- 2. varlen ----------
    logging.info("\n[varlen 验证 — 段长 [300, 512, 724] 均非 64 对齐]")
    cu_list = [0, 300, 812, 1536]
    for with_state in (False, True):
        args, h0 = _build_case(BuildCaseConfig(1, 1536, 4, 4, 128, 128, device, seed=7,
                               g_range=(-0.01, -0.0001) if with_state else (-0.10, -0.001),
                               with_state=with_state, n_seq=3))
        cu = torch.tensor(cu_list, dtype=torch.long, device=device)
        kw = {"chunk_size": 64, "cu_seqlens": cu, "output_final_state": True}
        if with_state:
            kw["initial_state"] = h0
        oc, sc = gdr_fwd_golden(*args, **kw)
        on, sn = gdr_fwd_naive(*args, **kw)
        ma, l2r, mr = _stats(oc, on)
        sma, sl2, smr = _stats(sc, sn)
        ok = (_finite(oc, on, sc, sn)
              and torch.allclose(oc.float(), on.float(), rtol=rtol_val, atol=atol_val)
              and torch.allclose(sc.float(), sn.float(), rtol=rtol_val, atol=atol_val))
        tag = "with initial_state" if with_state else "no initial_state "
        logging.info(f"  {tag}  o: max_abs={ma:.3e} l2_rel={l2r:.3e} max_rel={mr:.3e}")
        logging.info(f"  {' ' * len(tag)}  state: max_abs={sma:.3e} l2_rel={sl2:.3e} max_rel={smr:.3e}"
              f"  ... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"varlen(state={with_state})")

        # varlen 段独立性：逐段单独调用应与打包调用逐段一致
        for n in range(len(cu_list) - 1):
            s0, s1 = cu_list[n], cu_list[n + 1]
            sub = [x[:, s0:s1] for x in args]
            kw_sub = {"chunk_size": 64, "output_final_state": True}
            if with_state:
                kw_sub["initial_state"] = h0[n:n + 1]
            o_sub, s_sub = gdr_fwd_golden(*sub, **kw_sub)
            ma2, _, _ = _stats(o_sub, oc[:, s0:s1])
            ok2 = torch.allclose(o_sub.float(), oc[:, s0:s1].float(), rtol=1e-6, atol=1e-6)
            logging.info(f"    seg{n} [{s0}:{s1}] len={s1 - s0} 独立性 max_abs={ma2:.3e}"
                  f"  ... {'PASS' if ok2 else 'FAIL'}")
            if not ok2:
                failures.append(f"varlen-seg{n}(state={with_state})")

    # ---------- 3. 尾块专项 ----------
    logging.info("\n[尾块专项 — T % BT != 0 时 zero-pad 与掩码等价性]")
    for t_len, bt in ((1000, 64), (999, 64), (65, 64), (17, 16), (100, 32)):
        args, _ = _build_case(BuildCaseConfig(1, t_len, 2, 2, 64, 64, device, seed=11))
        oc, sc = gdr_fwd_golden(*args, chunk_size=bt, output_final_state=True)
        on, sn = gdr_fwd_naive(*args, chunk_size=bt, output_final_state=True)
        ma, l2r, mr = _stats(oc, on)
        sma, _, _ = _stats(sc, sn)
        ok = (_finite(oc, on)
              and torch.allclose(oc.float(), on.float(), rtol=rtol_val, atol=atol_val)
              and torch.allclose(sc.float(), sn.float(), rtol=rtol_val, atol=atol_val))
        logging.info(f"  T={t_len:5d} BT={bt}  L_last={t_len - (t_len - 1) // bt * bt:3d}  "
              f"o max_abs={ma:.3e} max_rel={mr:.3e}  state max_abs={sma:.3e}"
              f"  ... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"tail(T={t_len}, BT={bt})")

    # 掩码 ⟺ host zero-pad 等价性（SPEC §4.1 / MATH.md §11 的代码级证明）
    for t_len, bt in ((1000, 64), (999, 64), (100, 32)):
        args, _ = _build_case(BuildCaseConfig(1, t_len, 2, 2, 64, 64, device, seed=11))
        o_mask, s_mask = gdr_fwd_golden(*args, chunk_size=bt, output_final_state=True)
        npad = (bt - t_len % bt) % bt
        padded = []
        for x in args:
            shape = list(x.shape)
            shape[1] = npad
            padded.append(torch.cat([x, torch.zeros(shape, dtype=x.dtype, device=device)], dim=1))
        o_pad, s_pad = gdr_fwd_golden(*padded, chunk_size=bt, output_final_state=True)
        ma, _, _ = _stats(o_pad[:, :t_len], o_mask)
        sma, _, _ = _stats(s_pad, s_mask)
        ok = (torch.allclose(o_pad[:, :t_len].float(), o_mask.float(), rtol=1e-6, atol=1e-6)
              and torch.allclose(s_pad.float(), s_mask.float(), rtol=1e-6, atol=1e-6))
        logging.info(f"  T={t_len:5d} BT={bt} 掩码 vs host zero-pad(+{npad}): "
              f"o max_abs={ma:.3e} state max_abs={sma:.3e} ... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"zeropad-equiv(T={t_len}, BT={bt})")

    # ---------- 4. GVA 专项 ----------
    logging.info("\n[GVA 专项 — q/k 头按 hv // (HV//H) 广播]")
    for h, hv in ((4, 4), (4, 8), (2, 8), (1, 4)):
        args, _ = _build_case(BuildCaseConfig(1, 256, h, hv, 64, 64, device, seed=13))
        oc, _ = gdr_fwd_golden(*args, chunk_size=64)
        on, _ = gdr_fwd_naive(*args, chunk_size=64)
        ma, l2r, mr = _stats(oc, on)
        ok = _finite(oc, on) and torch.allclose(oc.float(), on.float(), rtol=rtol_val, atol=atol_val)
        # 显式核对头映射：hv 组内共享同一 q/k 头
        q, k, v, g, beta = args
        g_grp = hv // h
        q_exp = _expand_qk_to_hv(q, hv)
        map_ok = all(
            torch.equal(q_exp[:, :, i], q[:, :, i // g_grp]) for i in range(hv)
        )
        logging.info(f"  H={h} HV={hv} (G={g_grp})  max_abs={ma:.3e} max_rel={mr:.3e}  "
              f"head_map={'OK' if map_ok else 'BAD'}  ... {'PASS' if ok and map_ok else 'FAIL'}")
        if not (ok and map_ok):
            failures.append(f"gva(H={h}, HV={hv})")

    # ---------- 5. emulate_bf16 ----------
    logging.info("\n[emulate_bf16 — 内部 bf16 路径 vs fp32 真值]")
    for name, b, t, h, hv, kd, bt in (
        ("base", 2, 1024, 4, 4, 128, 64),
        ("tail", 2, 1000, 4, 4, 128, 64),
        ("gva ", 2, 512, 4, 8, 128, 64),
    ):
        args, _ = _build_case(BuildCaseConfig(b, t, h, hv, kd, kd, device, seed=17))
        o32, _ = gdr_fwd_golden(*args, chunk_size=bt, emulate_bf16=False)
        o16, _ = gdr_fwd_golden(*args, chunk_size=bt, emulate_bf16=True)
        ma, l2r, mr = _stats(o16, o32)
        ok = _finite(o16) and l2r < 2e-2
        logging.info(f"  {name} B={b} T={t} HV={hv}  max_abs={ma:.3e} l2_rel={l2r:.3e} "
              f"max_rel={mr:.3e}  ... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"emulate_bf16({name})")

    # bf16 输入端到端可跑通
    args, _ = _build_case(BuildCaseConfig(2, 512, 4, 4, 128, 128, device, seed=19))
    args_bf = [x.to(torch.bfloat16) if x.dim() == 4 else x for x in args]
    o_bf, s_bf = gdr_fwd_golden(*args_bf, chunk_size=64, emulate_bf16=True,
                                output_final_state=True)
    ok = (o_bf.dtype == torch.bfloat16 and s_bf.dtype == torch.float32
          and _finite(o_bf, s_bf))
    logging.info(f"  bf16 IO: o.dtype={o_bf.dtype} state.dtype={s_bf.dtype} "
          f"finite={_finite(o_bf, s_bf)}  ... {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("bf16-io")

    # ---------- 6. 结构 / 契约检查 ----------
    logging.info("\n[结构与契约检查]")
    args, h0 = _build_case(BuildCaseConfig(2, 512, 4, 4, 128, 128, device, seed=23, with_state=True))
    o, s = gdr_fwd_golden(*args, initial_state=h0, output_final_state=True)
    checks = [
        ("o.shape == [B, T, HV, V]", tuple(o.shape) == (2, 512, 4, 128)),
        ("state.shape == [N, HV, K, V]", tuple(s.shape) == (2, 4, 128, 128)),
        ("state dtype fp32", s.dtype == torch.float32),
        ("o dtype follows q", o.dtype == args[0].dtype),
    ]
    o2, s2 = gdr_fwd_golden(*args, initial_state=h0, output_final_state=False)
    checks.append(("output_final_state=False -> None", s2 is None))
    # final_state 可直接回灌 initial_state（K 在 V 前，A↔B1 做不到的事）
    o3, s3 = gdr_fwd_golden(*args, initial_state=s, output_final_state=True)
    checks.append(("final_state 可回灌 initial_state", _finite(o3, s3)))
    # scale=None 等价于 K**-0.5
    oa, _ = gdr_fwd_golden(*args, scale=None)
    ob, _ = gdr_fwd_golden(*args, scale=128 ** -0.5)
    checks.append(("scale=None == K**-0.5", torch.equal(oa, ob)))
    # 两段拼接 = 分别跑（状态不跨序列边界）
    cu = torch.tensor([0, 200, 512], dtype=torch.long, device=device)
    args1 = [x[:1] for x in args]
    ov, _ = gdr_fwd_golden(*args1, cu_seqlens=cu)
    oa1, sa1 = gdr_fwd_golden(*[x[:, :200] for x in args1], output_final_state=True)
    ob1, _ = gdr_fwd_golden(*[x[:, 200:] for x in args1])
    checks.append(("varlen 不跨段传状态",
                   torch.allclose(ov[:, :200].float(), oa1.float(), atol=1e-6)
                   and torch.allclose(ov[:, 200:].float(), ob1.float(), atol=1e-6)))
    for label, cond in checks:
        logging.info(f"  {label} ... {'PASS' if cond else 'FAIL'}")
        if not cond:
            failures.append(label)

    # ---------- 7. 异常与边界 ----------
    logging.info("\n[异常与边界]")
    q, k, v, g, beta = args

    def _expect(exc_type, fn, label):
        try:
            fn()
        except exc:
            logging.info(f"  {label} -> {exc.__name__} ... PASS")
            return True
        except Exception as e:            # noqa: BLE001
            logging.info(f"  {label} -> 期望 {exc.__name__}，实际 {type(e).__name__} ... FAIL")
            return False
        logging.info(f"  {label} -> 未抛异常 ... FAIL")
        return False

    ok_all = True
    # chunk_size=128 已正式放开（SPEC §D1 表格第 4 行），故负例改用仍非法的 48。
    ok_all &= _expect(ValueError, lambda: gdr_fwd_golden(q, k, v, g, beta, chunk_size=48),
                      "chunk_size=48")
    ok_all &= _expect(ValueError, lambda: gdr_fwd_golden(
        q, k[:, :, :2], v, g, beta), "q/k 头数不等")
    ok_all &= _expect(ValueError, lambda: gdr_fwd_golden(
        q, k, v[:, :, :3], g[:, :, :3], beta[:, :, :3]), "HV % H != 0")
    ok_all &= _expect(ValueError, lambda: gdr_fwd_golden(
        q, k, v, g, beta, cu_seqlens=torch.tensor([0, 512], device=device)), "cu_seqlens 下 B!=1")
    # h0 有 2 条，cu_seqlens 声明 3 条 -> 不匹配
    ok_all &= _expect(ValueError, lambda: gdr_fwd_golden(
        *[x[:1] for x in args], cu_seqlens=torch.tensor([0, 100, 200, 512], device=device),
        initial_state=h0), "initial_state 数量与序列数不符")
    if not ok_all:
        failures.append("异常检查")

    # ---------- 8. 数值稳定性 ----------
    logging.info("\n[数值稳定性]")
    stab = []
    # g 极负（衰减极快）
    args, _ = _build_case(BuildCaseConfig(1, 256, 2, 2, 64, 64, device, seed=29, g_range=(-20.0, -5.0)))
    o, s = gdr_fwd_golden(*args, output_final_state=True)
    stab.append(("g in [-20, -5] 极快衰减", _finite(o, s)))
    # g 接近 0（几乎无衰减）
    args, _ = _build_case(BuildCaseConfig(1, 256, 2, 2, 64, 64, device, seed=31, g_range=(-1e-6, -1e-9)))
    o, s = gdr_fwd_golden(*args, output_final_state=True)
    stab.append(("g ≈ 0 几乎无衰减", _finite(o, s)))
    # beta 全 0（delta rule 不写入 -> o 只剩 inter 项）
    args, _ = _build_case(BuildCaseConfig(1, 128, 2, 2, 64, 64, device, seed=37))
    args[4] = torch.zeros_like(args[4])
    o, s = gdr_fwd_golden(*args, output_final_state=True)
    on, sn = gdr_fwd_naive(*args, output_final_state=True)
    stab.append(("beta=0 -> o==0 且 state==0",
                 _finite(o, s) and o.abs().max().item() < 1e-6
                 and s.abs().max().item() < 1e-6
                 and torch.allclose(o.float(), on.float(), atol=1e-6)))
    # T < BT（只有一个残缺 chunk）
    args, _ = _build_case(BuildCaseConfig(1, 5, 2, 2, 64, 64, device, seed=41))
    o, s = gdr_fwd_golden(*args, chunk_size=64, output_final_state=True)
    on, sn = gdr_fwd_naive(*args, chunk_size=64, output_final_state=True)
    stab.append(("T=5 < BT=64",
                 _finite(o, s) and torch.allclose(o.float(), on.float(), rtol=rtol_val, atol=atol_val)))

    args, _ = _build_case(BuildCaseConfig(1, 1, 2, 2, 64, 64, device, seed=43))
    o, s = gdr_fwd_golden(*args, chunk_size=16, output_final_state=True)
    on, sn = gdr_fwd_naive(*args, chunk_size=16, output_final_state=True)
    stab.append(("T=1",
                 _finite(o, s) and torch.allclose(o.float(), on.float(), rtol=rtol_val, atol=atol_val)))
    for label, cond in stab:
        logging.info(f"  {label} ... {'PASS' if cond else 'FAIL'}")
        if not cond:
            failures.append(label)

    # ---------- 9. _make_inputs 冒烟 ----------
    logging.info("\n[_make_inputs P0 冒烟]")
    for case_name, case_args, case_kw in _make_inputs(device):
        o, s = gdr_fwd_golden(*case_args, **case_kw)
        ok = _finite(o, s)
        shp = tuple(o.shape)
        logging.info(f"  {case_name:<11s} o={shp} state={None if s is None else tuple(s.shape)}"
              f"  finite={ok} ... {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"make_inputs:{case_name}")

    logging.info("\n" + "=" * 74)
    if failures:
        logging.info(f"❌ 验证失败 {len(failures)} 项: {failures}")
        logging.info("=" * 74)
        raise SystemExit(1)
    logging.info("✅ 所有验证通过")
    logging.info("=" * 74)


logging.basicConfig(level=logging.INFO, format="%(message)s")
if __name__ == "__main__":
    _validate()
