# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# =============================================================================
# chunk_kda_golden.py  —  Stage 2 golden reference (mathematician-owned)
#
# Operator: chunk_kda (Kimi Delta Attention — chunked gated delta-rule linear
#           attention, forward only).
# Reference: theta-flash-linear-attention/fla/ops/kda/naive.py::naive_chunk_kda
#            (cross-checked vs naive_recurrent_kda). Signature + varlen (TND)
#            semantics aligned to kda.py::chunk_kda_fwd (lines 1129-1205) and its
#            wrapper chunk_kda (kda.py:1208).
# Near-exact prod idiom: models/qwen3_next/gated_delta_rule_impl.py.
#
# Confidence: ****  (known paper op; semantics 1:1 with naive_chunk_kda for the
#             fixed-length (BTHK) path, and 1:1 with per-sequence naive_recurrent
#             for the varlen TND path; both verified by torch.allclose).
#
# Signature (aligned to chunk_kda_fwd, fwd-order params; chunk_size via kwargs):
#   chunk_kda_golden(q, k, v, g, beta, scale=None, initial_state=None,
#                    output_final_state=False, cu_seqlens=None,
#                    prebuilt_meta=None, **kwargs) -> (o, final_state)
#   - cu_seqlens=None      : fixed-length BTHK. q/k/g [B,T,H,K], v [B,T,H,V],
#                            beta [B,T,H]; B independent sequences. Numerics are
#                            byte-identical to the previous golden.
#   - cu_seqlens=[N+1] int : varlen TND. B MUST be 1; T = total tokens; N seqs
#                            split by cu_seqlens. Chunks (BT=64) are per-segment,
#                            never cross a sequence; state resets per sequence.
#                            initial_state/final_state are [N,H,V,K] (upstream
#                            chunk_kda ABI = S^T; internal math stays [K,V]).
#   - prebuilt_meta=(chunk_indices_chunk64, chunk_offsets_chunk64) is accepted
#     for fwd-signature parity; golden derives segmentation from cu_seqlens, so
#     the meta is informational only (kernel tiling hint, not needed here).
#
# PyPTO-friendly normalization (per pypto-golden-generate §13 + golden_template
# Layer A-F constraints; lint OL15 = no pypto import):
#   - internal fp32; bf16/fp16 in -> o cast back, S kept fp32
#   - matmul inputs .float() first (FP32 accum, matches Cube L0C)
#   - NO .T / .t()          -> torch.transpose(t, -2, -1)
#   - NO torch.cumsum       -> prefix-sum via lower-tri @-matmul (Layer D const)
#   - NO masked_fill/tril/triu factory -> masks built from arange compare
#   - the 64x64 forward-substitution python loop is VECTORIZED to a single
#     solve_triangular; the per-column A/A2 build loops are VECTORIZED to matmul
#   - explicit reshape, shape comment on every intermediate line
#   - module boundaries marked # --- Module Mk ---
# =============================================================================

# o down-cast bf16, S fp32 internal route
_DEVICE_DEFAULT = "npu:0"  # TILE_FWK_DEVICE_ID=0 default; no sim, no card switch


def _get_device():  # honor TILE_FWK_DEVICE_ID, fall back to cpu only when no NPU
    import os, torch
    dev_id = os.environ.get("TILE_FWK_DEVICE_ID", "0")
    try:
        import torch_npu  # noqa: F401
        if torch.npu.device_count() == 0:
            return torch.device("cpu")
        return torch.device(f"npu:{dev_id}")
    except ImportError:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


# =============================================================================
# Layer D — Host-side constants (replace cumsum / tril / triu / masked_fill)
# =============================================================================
def _make_chunk_consts(BT, device, fdtype):
    import torch
    idx = torch.arange(BT, device=device)                                # [BT]
    row = idx.reshape(BT, 1)                                             # [BT,1]
    col = idx.reshape(1, BT)                                             # [1,BT]
    tril_incl = (row >= col).to(fdtype)                                  # [BT,BT] lower incl diag
    tril_strict = (row > col).to(fdtype)                                 # [BT,BT] strict lower
    eye = (row == col).to(fdtype)                                        # [BT,BT] identity
    return tril_incl, tril_strict, eye


# =============================================================================
# Module M1: rearrange + scale + gated cumsum  → q_s, k, v, g_cum, beta
# =============================================================================
def _m1_prep(q, k, v, g, beta, scale, BT, NT, tril_incl):
    import torch
    B, T, H, K = q.shape                                                 # [B,T,H,K]
    V = v.shape[-1]                                                      # V
    # b (n c) h d -> b h n c d : split T into NT chunks of BT, move H ahead
    q = q.reshape(B, NT, BT, H, K).transpose(1, 3).transpose(2, 3)       # [B,H,NT,BT,K]
    k = k.reshape(B, NT, BT, H, K).transpose(1, 3).transpose(2, 3)       # [B,H,NT,BT,K]
    v = v.reshape(B, NT, BT, H, V).transpose(1, 3).transpose(2, 3)       # [B,H,NT,BT,V]
    g = g.reshape(B, NT, BT, H, K).transpose(1, 3).transpose(2, 3)       # [B,H,NT,BT,K]
    beta = beta.reshape(B, NT, BT, H).transpose(1, 3).transpose(2, 3)    # [B,H,NT,BT]
    q_s = q * scale                                                      # [B,H,NT,BT,K]
    # g.cumsum(-2) over BT axis: prefix-sum via tril_incl @ g  (no torch.cumsum)
    g_cum = torch.matmul(tril_incl, g)                                   # [B,H,NT,BT,K]
    return q_s, k, v, g_cum, beta                                        # contiguous chunk tensors


# =============================================================================
# Module M2: intra-chunk A-matrix + (I-A) forward-subst inverse  → A_inv
# =============================================================================
def _m2_inverse(k, g_cum, beta, tril_strict, eye):
    import torch
    g_c = g_cum.unsqueeze(-2)                                            # [B,H,NT,BT,1,K] row index
    g_i = g_cum.unsqueeze(-3)                                            # [B,H,NT,1,BT,K] col index
    decay = torch.exp(g_c - g_i)                                         # [B,H,NT,BT,BT,K] e^{gc-gi} stable
    kk = k.unsqueeze(-2) * k.unsqueeze(-3)                               # [B,H,NT,BT,BT,K] k_c*k_i
    A_full = torch.sum(kk * decay, dim=-1)                               # [B,H,NT,BT,BT] sum_d, strict-lower
    A = A_full * beta.unsqueeze(-1)                                      # [B,H,NT,BT,BT] *beta_row
    A = A * tril_strict                                                  # [B,H,NT,BT,BT] zero upper+diag
    M = eye + A                                                          # [B,H,NT,BT,BT] (I + A) lower-tri
    A_inv = torch.linalg.solve_triangular(M, eye.expand_as(M), upper=False)  # [B,H,NT,BT,BT]
    A_inv = A_inv * beta.unsqueeze(-2)                                   # [B,H,NT,BT,BT] *beta_col
    return A_inv                                                         # (I-A)^-1 * beta


# =============================================================================
# Module M3: w / u value & key-decay projections  → w, u
# =============================================================================
def _m3_wu(A_inv, k, v, g_cum):
    import torch
    gk = torch.exp(g_cum) * k                                           # [B,H,NT,BT,K]
    w = torch.matmul(A_inv, gk)                                         # [B,H,NT,BT,K] A@(e^g k)
    u = torch.matmul(A_inv, v)                                          # [B,H,NT,BT,V] A@v
    return w, u                                                         # chunk delta-corrected k,v


# =============================================================================
# Module M4: cross-chunk state recurrence  → o, S
# =============================================================================
def _m4_recur(q_s, k, v, g_cum, u, w, B, H, K, V, NT, BT, S, tri_low, odtype):
    import torch
    o = torch.zeros(B, H, NT, BT, V, dtype=torch.float, device=q_s.device)  # [B,H,NT,BT,V]
    for i in range(NT):                                                 # host chunk loop (state recur)
        q_i = q_s[:, :, i]                                             # [B,H,BT,K]
        k_i = k[:, :, i]                                               # [B,H,BT,K]
        u_i = u[:, :, i]                                               # [B,H,BT,V]
        w_i = w[:, :, i]                                               # [B,H,BT,K]
        g_i = g_cum[:, :, i]                                           # [B,H,BT,K]
        qg = q_i * torch.exp(g_i)                                      # [B,H,BT,K] q*e^g
        d2 = torch.exp(g_i.unsqueeze(-2) - g_i.unsqueeze(-3))          # [B,H,BT,BT,K] e^{gc-gj} stable
        A2 = torch.sum(q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * d2, -1) # [B,H,BT,BT] sum_d q_c k_j
        A2 = A2 * tri_low                                             # [B,H,BT,BT] keep lower+diag (g_c-g_j<=0)
        v_i = u_i - torch.matmul(w_i, S)                              # [B,H,BT,V] u - w@S
        o[:, :, i] = torch.matmul(qg, S) + torch.matmul(A2, v_i)      # [B,H,BT,V] inter + intra
        g_last = g_i[:, :, -1:, :]                                    # [B,H,1,K] chunk-last cumdecay
        S = S * torch.exp(g_last).transpose(-2, -1)                   # [B,H,K,V] S*e^{g_last}
        kg = (torch.exp(g_last - g_i) * k_i).transpose(-2, -1)        # [B,H,K,BT] (e^{gl-g} k)^T
        S = S + torch.matmul(kg, v_i)                                 # [B,H,K,V] + k^T@v
    return o, S


# =============================================================================
# Dense forward core (cu_seqlens=None path).  Identical math to the previous
# golden — fixed-length numerics are unchanged. Returns (o[B,T,H,V] fp32,
# S[B,H,K,V] fp32).
# =============================================================================
def _dense_forward(qf, kf, vf, gf, bf, scale, S0, BT, device, odtype):
    import torch
    B, T, H, K = qf.shape                                              # [B,T,H,K]
    V = vf.shape[-1]                                                   # V
    NT = T // BT                                                       # T/BT
    tril_incl, tril_strict, eye = _make_chunk_consts(BT, device, torch.float)  # [BT,BT]
    # --- Module M1 ---
    q_s, kk, vv, g_cum, bb = _m1_prep(qf, kf, vf, gf, bf, scale, BT, NT, tril_incl)  # chunked
    # --- Module M2 ---
    A_inv = _m2_inverse(kk, g_cum, bb, tril_strict, eye)               # [B,H,NT,BT,BT]
    # --- Module M3 ---
    w, u = _m3_wu(A_inv, kk, vv, g_cum)                                # [.,BT,K]/[.,BT,V]
    # --- Module M4 ---
    S = torch.zeros(B, H, K, V, dtype=torch.float, device=device)      # [B,H,K,V]
    if S0 is not None:
        S = S + S0                                                     # [B,H,K,V] seed initial_state
    o5, S = _m4_recur(q_s, kk, vv, g_cum, u, w, B, H, K, V, NT, BT, S, tril_incl, odtype)  # [B,H,NT,BT,V]
    o = o5.transpose(2, 3).transpose(1, 3).reshape(B, T, H, V)         # [B,T,H,V] b h n c d->b(nc)h d
    return o, S                                                        # fp32 o, fp32 S


# =============================================================================
# q/k L2-norm over the feature dim K (kda `use_qk_l2norm_in_kernel`)
# =============================================================================
def _l2norm_lastdim(x, eps=1e-6):
    import torch
    # Per-token L2 normalization over the last (feature) dim: x / sqrt(Σx²+eps).
    # Mirrors FLA l2norm_fwd (theta-flash-linear-attention/fla/modules/l2norm.py,
    # default eps=1e-6) applied by kda.py::chunk_kda (L1225-1227) BEFORE the fwd.
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)       # [...,K]


# =============================================================================
# Public golden entry — verifier and tests import this
# =============================================================================
def chunk_kda_golden(q, k, v, g, beta, scale=None, initial_state=None,
                     output_final_state=False, use_qk_l2norm_in_kernel=False,
                     cu_seqlens=None, prebuilt_meta=None, **kwargs):
    import torch
    odtype = q.dtype                                                   # bf16/fp16/fp32 out route
    device = q.device if getattr(q, "is_npu", False) or q.is_cuda else _get_device()
    q = q.to(device); k = k.to(device); v = v.to(device)              # device migration
    g = g.to(device); beta = beta.to(device)                          # device migration
    B, T, H, K = q.shape                                              # [B,T,H,K]
    V = v.shape[-1]                                                   # V
    BT = int(kwargs.get("chunk_size", 64))                            # FLA_CHUNK_SIZE=64
    if scale is None:
        scale = K ** -0.5                                            # K**-0.5
    qf = q.float(); kf = k.float(); vf = v.float()                    # internal fp32
    gf = g.float(); bf = beta.float()                                 # internal fp32
    # q/k L2-norm BEFORE scale/forward (kda l2norm THEN scale). Per-token over K;
    # independent of B/T/H slicing, so the per-(B,H) test harness stays valid. In
    # the varlen path qf/kf are normalized here (valid tokens), then per-seg zero-
    # padded downstream — pad rows stay 0, matching the kernel's per-chunk norm.
    if use_qk_l2norm_in_kernel:
        qf = _l2norm_lastdim(qf)                                      # [B,T,H,K] normalize q
        kf = _l2norm_lastdim(kf)                                      # [B,T,H,K] normalize k

    # ---- fixed-length BTHK path (cu_seqlens=None): per-B independent, unchanged ----
    if cu_seqlens is None:
        assert T % BT == 0, "T must be a multiple of chunk_size when cu_seqlens=None"
        # upstream ABI: initial_state is [B,H,V,K] (S^T) -> transpose to internal [K,V]
        S0 = initial_state.to(device).float().transpose(-2, -1) if initial_state is not None else None  # [B,H,V,K]->[B,H,K,V]
        o, S = _dense_forward(qf, kf, vf, gf, bf, scale, S0, BT, device, odtype)       # [B,T,H,V]/[B,H,K,V]
        # upstream ABI: emit final_state as [B,H,V,K] (internal [K,V] -> [V,K])
        return o.to(odtype), (S.transpose(-2, -1).contiguous() if output_final_state else None)  # o cast, S fp32 [B,H,V,K]

    # ---- varlen TND path: B must be 1, segments split by cu_seqlens, no cross-seq ----
    assert B == 1, "varlen (cu_seqlens) requires batch size 1 (TND packed layout)"
    seq = cu_seqlens.to("cpu").long().tolist()                        # [N+1] boundaries
    N = len(seq) - 1                                                  # number of sequences
    o = torch.zeros(1, T, H, V, dtype=torch.float, device=device)    # [1,T,H,V] packed out
    finals = []                                                      # per-seq [1,H,K,V]
    for n in range(N):                                              # host segment loop (no cross-seq)
        bos, eos = seq[n], seq[n + 1]                              # token span [bos,eos)
        L = eos - bos                                              # seg length (may be % BT != 0)
        NT = (L + BT - 1) // BT                                    # ceil chunks; last chunk partial
        Lp = NT * BT                                               # pad length to chunk multiple
        qs = torch.zeros(1, Lp, H, K, dtype=torch.float, device=device)  # [1,Lp,H,K] zero-pad
        ks = torch.zeros(1, Lp, H, K, dtype=torch.float, device=device)  # [1,Lp,H,K]
        vs = torch.zeros(1, Lp, H, V, dtype=torch.float, device=device)  # [1,Lp,H,V]
        gs = torch.zeros(1, Lp, H, K, dtype=torch.float, device=device)  # [1,Lp,H,K] pad g=0
        bs = torch.zeros(1, Lp, H, dtype=torch.float, device=device)     # [1,Lp,H]  pad beta=0
        qs[:, :L] = qf[:, bos:eos]; ks[:, :L] = kf[:, bos:eos]      # fill valid rows
        vs[:, :L] = vf[:, bos:eos]; gs[:, :L] = gf[:, bos:eos]      # pad rows stay 0
        bs[:, :L] = bf[:, bos:eos]                                  # pad beta=0 -> inert chunk
        # upstream ABI: initial_state[n] is [1,H,V,K] (S^T) -> transpose to internal [K,V]
        S0 = initial_state[n:n + 1].to(device).float().transpose(-2, -1) if initial_state is not None else None  # [1,H,V,K]->[1,H,K,V]
        o_seg, S_seg = _dense_forward(qs, ks, vs, gs, bs, scale, S0, BT, device, odtype)  # [1,Lp,H,V]
        o[:, bos:eos] = o_seg[:, :L]                                # write valid rows back
        finals.append(S_seg)                                       # [1,H,K,V] per-seq final
    # upstream ABI: emit final_state as [N,H,V,K] (internal [K,V] -> [V,K])
    final_state = torch.cat(finals, dim=0).transpose(-2, -1).contiguous() if output_final_state else None  # [N,H,V,K]
    return o.to(odtype), final_state                               # o cast, S fp32 [N,H,V,K]


# SPEC mandatory contract alias
chunk_kda_wrapper = chunk_kda_golden


# =============================================================================
# Validation — fixed-len vs naive_chunk_kda allclose (>=3 shapes) + chunk-vs-
# recurrent on stable inputs; varlen vs per-sequence naive_recurrent loop.
# =============================================================================
def _load_naive():
    import importlib.util as _u, os
    p = os.path.join(os.path.dirname(__file__),
                     "../../theta-flash-linear-attention/fla/ops/kda/naive.py")
    s = _u.spec_from_file_location("kda_naive", os.path.abspath(p))
    m = _u.module_from_spec(s); s.loader.exec_module(m)
    return m.naive_chunk_kda, m.naive_recurrent_kda


def _stable(B, T, H, K, V, device, dtype):
    import torch
    g0 = torch.randn(B, T, H, K, device=device)
    return (torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1,
            torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1,
            torch.randn(B, T, H, V, device=device, dtype=dtype) * 0.1,
            torch.nn.functional.logsigmoid(g0).to(dtype),
            torch.sigmoid(torch.randn(B, T, H, device=device)).to(dtype))


def _validate():
    import torch
    dev = _get_device()
    naive_chunk, naive_rec = _load_naive()
    cases = [(1, 128, 2, 64, 64), (2, 256, 4, 128, 128), (1, 512, 16, 128, 128)]
    ok = True
    print("[fixed-len fp32] golden(cu_seqlens=None) vs naive_chunk_kda (o,S atol/rtol 1e-3)")
    for B, T, H, K, V in cases:
        q, k, v, g, beta = _stable(B, T, H, K, V, dev, torch.float32)
        o, S = chunk_kda_golden(q, k, v, g, beta, output_final_state=True)
        ro, rS = naive_chunk(q, k, v, g, beta, output_final_state=True)
        # golden public S is now [V,K] (upstream ABI); naive_chunk uses internal [K,V] -> transpose back to compare
        p = torch.allclose(o, ro, rtol=1e-3, atol=1e-3) and torch.allclose(S.transpose(-2, -1), rS, rtol=1e-3, atol=1e-3)
        ok &= p
        print(f"  T{T}K{K}: o={(o-ro).abs().max():.1e} S={(S-rS).abs().max():.1e} {'PASS' if p else 'FAIL'}")
    print("[fixed-len bf16] golden vs naive_chunk_kda (o atol/rtol 1e-2) + chunk-vs-recurrent")
    for B, T, H, K, V in cases:
        q, k, v, g, beta = _stable(B, T, H, K, V, dev, torch.bfloat16)
        o, S = chunk_kda_golden(q, k, v, g, beta, output_final_state=True)
        ro, rS = naive_chunk(q, k, v, g, beta, output_final_state=True)
        ar, _ = naive_rec(q, k, v, g, beta, output_final_state=True)
        p1 = torch.allclose(o.float(), ro.float(), rtol=1e-2, atol=1e-2)
        p2 = torch.allclose(ro.float(), ar.float(), rtol=5e-2, atol=5e-2)
        ok &= p1 and p2
        print(f"  T{T}K{K}: o={(o.float()-ro.float()).abs().max():.1e} chunk-vs-rec="
              f"{(ro.float()-ar.float()).abs().max():.1e} {'PASS' if p1 and p2 else 'FAIL'}")
    # varlen: B1 packed; cross-check vs per-sequence naive_recurrent (any-length, token-level)
    print("[varlen fp32] golden(B1,cu_seqlens) vs per-seq naive_recurrent (o,S atol/rtol 1e-3)")
    vlcases = [(2, 64), (4, 128), (16, 128)]  # (H, K=V) ; segment lens incl non-%64
    seglens = [64, 200, 320]                  # 200 -> partial last chunk
    for H, K in vlcases:
        V = K
        cu = [0]
        for L in seglens:
            cu.append(cu[-1] + L)
        Ttot = cu[-1]
        cu_t = torch.tensor(cu, device=dev, dtype=torch.int32)
        q, k, v, g, beta = _stable(1, Ttot, H, K, V, dev, torch.float32)
        S0 = (torch.randn(len(seglens), H, K, V, device=dev) * 0.1)
        # upstream ABI: golden consumes initial_state as [N,H,V,K]; feed S0^T so the internal
        # seed equals naive's [K,V] S0, and transpose golden's [V,K] output back to compare.
        o, S = chunk_kda_golden(q, k, v, g, beta, initial_state=S0.transpose(-2, -1),
                                output_final_state=True, cu_seqlens=cu_t)
        good = True
        for n, L in enumerate(seglens):
            sl = slice(cu[n], cu[n + 1])
            ro, rS = naive_rec(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
                               initial_state=S0[n:n + 1], output_final_state=True)
            good &= torch.allclose(o[:, sl], ro, 1e-3, 1e-3) and torch.allclose(S[n:n + 1].transpose(-2, -1), rS, 1e-3, 1e-3)
        ok &= good
        print(f"  H{H}K{K} segs{seglens}: {'PASS' if good else 'FAIL'}")
    print("[PRECISION_PASS]" if ok else "[PRECISION_FAIL]")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_validate())
