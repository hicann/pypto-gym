#!/usr/bin/env python3
# coding: utf-8
# =============================================================================
# flash_attention_score_impl.py — Integrated Production Kernel
# Flash Attention Score with PSE (Positional Score Encoding) and Dropout (GQA).
# =============================================================================

import pypto
import torch
import torch_npu  # noqa: F401  required for NPU device init

BLOCK_Q = 64
BLOCK_KV = 64

# Tile shapes from DESIGN.md §3.2.5 — first-pass minimum-viable (Stage 5 default)
VEC_TILE = (16, 64)
CUBE_QK = ([64, 64], [128, 128], [64, 64])   # Q@K^T: M≤64, K≤256, N≤64
CUBE_PV = ([64, 64], [64, 64], [128, 128])   # P@V:   M≤64, K≤64, N≤256


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def flash_attention_score_kernel_npu(  # pylint: disable=huawei-too-many-arguments
    query:       pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    key:         pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    value:       pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    atten_mask:  pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    pse:         pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    drop_mask:   pypto.Tensor([pypto.DYNAMIC, pypto.DYNAMIC], pypto.DT_BF16),
    output:      pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    softmax_max: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),
    softmax_sum: pypto.Tensor([pypto.DYNAMIC, 1], pypto.DT_FP32),
    pse_type:    int,
    keep_prob:   float,
    scale_value: float,
    N_kv:        int,
    B:           int,
    N:           int,
    Sq:          int,
    Skv:         int,
):
    """Integrated Flash Attention Score kernel.

    5 nested pypto.loop calls: batch → kv_head → group → Q-block → KV-block.
    4D→2D flatten at entry (inplace=True); all block extraction via 2D→2D
    pypto.view(valid_shape=...). Output via 2D→2D pypto.assemble into
    flattened 2D output (host-side unflatten restores 4D).
    """
    # --- Symbolic dimensions (compile-time constants from host) ---
    D           = query.shape[1]                                     # STATIC — safe for view/tensor shapes
    group       = N // N_kv
    num_blocks_q  = (Sq + BLOCK_Q - 1) // BLOCK_Q
    num_blocks_kv = (Skv + BLOCK_KV - 1) // BLOCK_KV

    # =====================================================================
    # Wrapper flattens 4D→2D on host side. Kernel works entirely in 2D.
    # All block extraction via 2D→2D pypto.view ; output via
    # pypto.assemble directly into 2D output parameters.
    # =====================================================================
    query_2d   = query
    key_2d     = key
    value_2d   = value
    pse_2d     = pse
    output_2d  = output
    mm_2d      = softmax_max
    ms_2d      = softmax_sum

    # =====================================================================
    # 5 nested loops
    #   - Outer loops: no unroll_list, no submit_before_loop (independent)
    #   - Innermost KV loop: unroll_list=[1] (OL56), submit_before_loop=True
    # =====================================================================
    for b_idx in pypto.loop(B, name="batch_idx"):                      # loop over batch
        for kv_h in pypto.loop(N_kv, name="kv_head_idx"):              # loop over KV heads
            for grp in pypto.loop(group, name="group_idx"):            # loop over GQA groups
                n_idx = kv_h * group + grp                              # GQA head mapping

                for qb in pypto.loop(num_blocks_q, name="q_block_idx"): # loop over Q blocks
                    q_start = qb * BLOCK_Q                              # SymbolicScalar
                    q_tile_len = (Sq - q_start).min(BLOCK_Q)            # tail-safe clamp

                    # ----- 2D view: Q block [BLOCK_Q, D] with valid_shape -----
                    # Linear row index in flattened query_2d:
                    #   row = b_idx*N*Sq + n_idx*Sq + q_start
                    q_row = b_idx * N * Sq + n_idx * Sq + q_start
                    q_tile_view = pypto.view(query_2d, [BLOCK_Q, D],
                                             [q_row, 0],
                                             valid_shape=[q_tile_len, D])
                    # Set vec tile before cast
                    pypto.set_vec_tile_shapes(16, 64)

                    # ----- Scratch state for online-softmax (FP32, uninitialized) -----
                    mi_update = pypto.tensor([BLOCK_Q, 1], pypto.DT_FP32, "mi")
                    li_update = pypto.tensor([BLOCK_Q, 1], pypto.DT_FP32, "li")
                    oi_update = pypto.tensor([BLOCK_Q, D], pypto.DT_FP32, "oi")

                    # ----- Inner KV-block loop -----
                    for kvb in pypto.loop(num_blocks_kv, name="kv_block_idx",
                                          unroll_list=[1], submit_before_loop=True):
                        kv_start   = kvb * BLOCK_KV                     # SymbolicScalar
                        k_tile_len = (Skv - kv_start).min(BLOCK_KV)     # tail-safe clamp

                        # Set vec tile at start of KV-block loop body
                        pypto.set_vec_tile_shapes(16, 64)

                        # =======================================================
                        # === M1: Q@K^T matmul + PSE + Scale ===
                        # =======================================================

                        # 2D view: K block [BLOCK_KV, D]
                        k_row = b_idx * N_kv * Skv + kv_h * Skv + kv_start
                        k_tile_view = pypto.view(key_2d, [BLOCK_KV, D],
                                                [k_row, 0],
                                                valid_shape=[k_tile_len, D])

                        # 2D view: PSE block [BLOCK_Q, BLOCK_KV]
                        pse_row = b_idx * N * Sq + n_idx * Sq + q_start
                        pse_tile_view = pypto.view(pse_2d, [BLOCK_Q, BLOCK_KV],
                                                   [pse_row, kv_start],
                                                   valid_shape=[q_tile_len, k_tile_len])

                        # Stage V0: cast bf16 → fp32
                        q_fp32   = pypto.cast(q_tile_view, pypto.DT_FP32)
                        k_fp32   = pypto.cast(k_tile_view, pypto.DT_FP32)
                        pse_fp32 = pypto.cast(pse_tile_view, pypto.DT_FP32)

                        # Stage C1: Q@K^T matmul (cube — b_trans on-the-fly)
                        pypto.set_cube_tile_shapes([64, 64], [D, D], [64, 64])
                        scores = pypto.matmul(q_fp32, k_fp32, pypto.DT_FP32,
                                              b_trans=True)

                        # Stage V1: PSE addition + scale multiply
                        pypto.set_vec_tile_shapes(16, 64)
                        if pse_type == 1:
                            scores_scaled = pypto.mul(
                                pypto.add(scores, pse_fp32), scale_value
                            )
                        else:  # pse_type == 2
                            scores_scaled = pypto.add(
                                pypto.mul(scores, scale_value), pse_fp32
                            )

                        # =======================================================
                        # === M2: Softmax + Dropout + PV matmul ===
                        # =======================================================

                        # 2D view: mask block [BLOCK_Q, BLOCK_KV]
                        mask_tile = pypto.view(atten_mask, [BLOCK_Q, BLOCK_KV],
                                              [q_start, kv_start],
                                              valid_shape=[q_tile_len, k_tile_len])
                        mask_fp32 = pypto.cast(mask_tile, pypto.DT_FP32)

                        # 2D view: drop block [BLOCK_Q, BLOCK_KV]
                        drop_tile = pypto.view(drop_mask, [BLOCK_Q, BLOCK_KV],
                                              [q_start, kv_start],
                                              valid_shape=[q_tile_len, k_tile_len])
                        drop_fp32 = pypto.cast(drop_tile, pypto.DT_FP32)

                        # valid_mask: (mask + (-1.0)) * (-1.0) → 0→1, non-0→0
                        # Direct float literals — NOT pypto.Element
                        valid_mask = pypto.mul(pypto.add(mask_fp32, -1.0),
                                               -1.0)

                        # Safe softmax (log-sum-exp trick)
                        m_ij = pypto.amax(scores_scaled, dim=-1,
                                          keepdim=True)                  # [BLOCK_Q,1] fp32
                        s_shifted = pypto.sub(scores_scaled, m_ij)
                        p_ij = pypto.exp(s_shifted)                      # [BLOCK_Q,BLOCK_KV] fp32
                        p_ij = pypto.mul(p_ij, valid_mask)               # apply attn mask
                        p_ij = pypto.mul(p_ij, drop_fp32)                # apply dropout mask
                        l_ij = pypto.sum(p_ij, dim=-1,
                                         keepdim=True)                   # [BLOCK_Q,1] fp32

                        # 2D view: V block [BLOCK_KV, D]
                        v_row = b_idx * N_kv * Skv + kv_h * Skv + kv_start
                        v_tile_view = pypto.view(value_2d, [BLOCK_KV, D],
                                                 [v_row, 0],
                                                 valid_shape=[k_tile_len, D])
                        v_fp32 = pypto.cast(v_tile_view, pypto.DT_FP32)

                        # Stage C2: P@V matmul (cube)
                        pypto.set_cube_tile_shapes([64, 64], [64, 64], [D, D])
                        o_ij = pypto.matmul(p_ij, v_fp32, pypto.DT_FP32)    # [BLOCK_Q,D] fp32

                        # =======================================================
                        # === M3: Online-softmax accumulation + output ===
                        # =======================================================

                        pypto.set_vec_tile_shapes(16, 64)

                        if pypto.is_loop_begin(kvb):
                            # ---- First KV block ----
                            if pypto.is_loop_end(kvb):
                                # Edge case: single KV block
                                # Normalize P → cast to bf16 → matmul in bf16 → assemble
                                p_ij_norm = pypto.div(p_ij, l_ij)        # [BLOCK_Q,BLOCK_KV] fp32
                                p_ij_bf16 = pypto.cast(p_ij_norm, pypto.DT_BF16)

                                # Both matmul inputs must have same dtype
                                pypto.set_cube_tile_shapes([64, 64], [64, 64], [D, D])
                                o_final = pypto.matmul(p_ij_bf16, v_tile_view,
                                                       pypto.DT_BF16)    # [BLOCK_Q,D] bf16
                                pypto.set_vec_tile_shapes(16, 64)

                                 # 2D assemble: write o_final [BLOCK_Q, D] → output_2d

                                out_row = b_idx * N * Sq + n_idx * Sq + q_start
                                pypto.assemble(o_final, [out_row, 0], output_2d)
                                pypto.assemble(m_ij, [out_row, 0], mm_2d)

                                if keep_prob < 1.0:
                                    l_scaled = pypto.mul(l_ij, 1.0 / keep_prob)
                                else:
                                    l_scaled = l_ij
                                pypto.assemble(l_scaled, [out_row, 0], ms_2d)
                            else:
                                # First of multiple KV blocks: store state directly
                                oi_update[:] = o_ij
                                li_update[:] = l_ij
                                mi_update[:] = m_ij
                        else:
                            # ---- Subsequent KV blocks: rescale & accumulate ----
                            # Read state via pypto.view (2D→2D, valid_shape for tail Q)
                            mi = pypto.view(mi_update, [BLOCK_Q, 1], [0, 0],
                                            valid_shape=[q_tile_len, 1])
                            li = pypto.view(li_update, [BLOCK_Q, 1], [0, 0],
                                            valid_shape=[q_tile_len, 1])
                            oi = pypto.view(oi_update, [BLOCK_Q, D], [0, 0],
                                            valid_shape=[q_tile_len, D])

                            # Online-softmax rescale factors
                            mi_new = pypto.maximum(mi, m_ij)             # [BLOCK_Q,1] fp32
                            alpha  = pypto.exp(pypto.sub(mi, mi_new))    # [BLOCK_Q,1] fp32
                            beta   = pypto.exp(pypto.sub(m_ij, mi_new))  # [BLOCK_Q,1] fp32

                            # Weighted sum accumulation
                            li_new = pypto.add(pypto.mul(alpha, li),
                                              pypto.mul(beta, l_ij))     # [BLOCK_Q,1] fp32
                            oi_new = pypto.add(pypto.mul(alpha, oi),
                                              pypto.mul(beta, o_ij))     # [BLOCK_Q,D] fp32

                            if pypto.is_loop_end(kvb):
                                # ---- Last KV block: final normalize → cast → assemble ----
                                o_final = pypto.div(oi_new, li_new)      # [BLOCK_Q,D] fp32
                                o_bf16 = pypto.cast(o_final, pypto.DT_BF16)

                                out_row = b_idx * N * Sq + n_idx * Sq + q_start
                                pypto.assemble(o_bf16, [out_row, 0], output_2d)
                                pypto.assemble(mi_new, [out_row, 0], mm_2d)

                                if keep_prob < 1.0:
                                    l_scaled = pypto.mul(li_new, 1.0 / keep_prob)
                                else:
                                    l_scaled = li_new
                                pypto.assemble(l_scaled, [out_row, 0], ms_2d)
                            else:
                                # ---- Intermediate KV block: write state forward ----
                                oi_update[:] = oi_new
                                li_update[:] = li_new
                                mi_update[:] = mi_new


def flash_attention_score_wrapper(
    query,       # [B, N, Sq, D]       bf16
    key,         # [B, N_kv, Skv, D]   bf16
    value,       # [B, N_kv, Skv, D]   bf16
    atten_mask,  # [Sq, Skv]           bf16     (0=valid, non-zero=invalid)
    pse,         # [B, N, Sq, Skv]     bf16
    drop_mask,   # [Sq, Skv]           bf16
    pse_type,    # int                          (1 or 2)
    keep_prob,   # float
    scale_value, # float                        (typically 1/sqrt(D))
):
    """Wrapper for flash_attention_score — flatten on host, kernel works in 2D.

    The kernel signature is 2D because pypto.assemble only writes back to
    kernel-parameter tensors, not to pypto-internal buffers.  Flattening
    on the host side (torch.reshape — a no-copy view) and assembling
    directly into the 2D output parameter is the only mechanism that
    reliably transfers result data out of the JIT graph.
    """
    import torch  # pylint: disable=redefined-outer-name
    import torch_npu  # noqa: F401  # pylint: disable=redefined-outer-name

    device = query.device
    B, N, Sq, D = query.shape
    N_kv = key.shape[1]
    Skv = key.shape[2]

    # Flatten 4D → 2D (no-copy view)
    q_2d = query.reshape(B * N * Sq, D)
    k_2d = key.reshape(B * N_kv * Skv, D)
    v_2d = value.reshape(B * N_kv * Skv, D)
    p_2d = pse.reshape(B * N * Sq, Skv)

    output_2d = torch.zeros(B * N * Sq, D, dtype=torch.bfloat16, device=device)
    mm_2d  = torch.zeros(B * N * Sq, 1, dtype=torch.float32, device=device)
    ms_2d  = torch.zeros(B * N * Sq, 1, dtype=torch.float32, device=device)

    flash_attention_score_kernel_npu(
        q_2d, k_2d, v_2d, atten_mask, p_2d, drop_mask,
        output_2d, mm_2d, ms_2d,
        pse_type, keep_prob, scale_value, N_kv,
        B, N, Sq, Skv,
    )

    # Unflatten 2D → 4D (no-copy view)
    output       = output_2d.reshape(B, N, Sq, D)
    softmax_max  = mm_2d.reshape(B, N, Sq, 1)
    softmax_sum  = ms_2d.reshape(B, N, Sq, 1)

    return output, softmax_max, softmax_sum
