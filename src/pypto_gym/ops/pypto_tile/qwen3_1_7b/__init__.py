"""Qwen3-1.7B pypto kernel adapters.

This adapter wraps the per-kernel functions from ``pypto_gym/ops/qwen3_1_7b``
with the padding / cwd-isolation logic that the end-to-end network requires.

Each kernel runs in its own working directory to avoid kernel_aicpu/
ARG-name collisions when multiple JIT kernels co-exist in one process.

S-padding: kernels assume BS_TILE=8 outer loop. We pad inputs to a multiple
of BS_TILE so the AIC reads/writes never go past tensor end.
"""

import contextlib
import os
import sys
import tempfile
from pathlib import Path

import torch

# The op modules use sibling imports — add their directory to sys.path
# so `import qwen3_decode_attn`, `import qwen3_pre_attn_fused` etc. resolve.
# This file lives at  <repo>/src/pypto_gym/transformers/qwen3_1_7b/qwen3_pto_kernels/
# Ops live at        <repo>/src/pypto_gym/ops/qwen3_1_7b/
_OPS_DIR = (
    Path(__file__).resolve().parents[3]
    / "ops" / "qwen3_1_7b"
)
if str(_OPS_DIR) not in sys.path:
    sys.path.insert(0, str(_OPS_DIR))

from qwen3_iter1a_kernel import qwen3_pre_qkv_iter1a as _k1  # noqa: E402
from qwen3_k2_qk_rope import qwen3_qk_rope_q as _k2_q, qwen3_qk_rope_k as _k2_k  # noqa: E402
# K3 has a different weight layout in ops/qwen3_1_7b/ ([H, INT_SIZE]) — the
# end-to-end network passes torch Linear.weight directly ([INT_SIZE, H]),
# so we keep a network-compatible copy locally.
from .k3_post_attn import qwen3_post_attn_k3 as _k3  # noqa: E402
from qwen3_pre_attn_fused import qwen3_pre_attn_fused as _pre_attn_fused  # noqa: E402
from qwen3_decode_attn import qwen3_decode_attn as _decode_attn  # noqa: E402


_BASE_DIR = os.environ.get(
    "QWEN3_PTO_BUILD_DIR",
    os.path.join(tempfile.gettempdir(), "qwen3_pto_build"),
)
_K1_DIR = os.path.join(_BASE_DIR, "k1")
_K2Q_DIR = os.path.join(_BASE_DIR, "k2q")
_K2K_DIR = os.path.join(_BASE_DIR, "k2k")
_K3_DIR = os.path.join(_BASE_DIR, "k3")
_PREATTN_DIR = os.path.join(_BASE_DIR, "pre_attn_fused")
_DEC_ATTN_DIR = os.path.join(_BASE_DIR, "decode_attn")
for d in (_K1_DIR, _K2Q_DIR, _K2K_DIR, _K3_DIR, _PREATTN_DIR, _DEC_ATTN_DIR):
    os.makedirs(d, exist_ok=True)

_BS_TILE = 8


@contextlib.contextmanager
def _in_dir(path):
    prev = os.getcwd()
    if prev == path:
        yield
        return
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev)


def _pad_s(t, pad_to):
    """Pad first dim with zeros up to multiple of pad_to. Returns (padded, orig_S)."""
    S = t.shape[0]
    if S % pad_to == 0:
        return t.contiguous(), S
    pad_to_S = ((S + pad_to - 1) // pad_to) * pad_to
    pad_shape = list(t.shape)
    pad_shape[0] = pad_to_S - S
    pad = torch.zeros(*pad_shape, device=t.device, dtype=t.dtype)
    return torch.cat([t, pad], dim=0).contiguous(), S


def rmsnorm_qkv(x, w_in_norm, Wq, Wk, Wv, *, Nq=16, Nkv=8, D=128):
    x_p, S = _pad_s(x, _BS_TILE)
    Sp = x_p.shape[0]
    device, dtype = x.device, x.dtype
    q = torch.empty(Sp, Nq * D, device=device, dtype=dtype)
    k = torch.empty(Sp, Nkv * D, device=device, dtype=dtype)
    v = torch.empty(Sp, Nkv * D, device=device, dtype=dtype)
    with _in_dir(_K1_DIR):
        _k1(x_p, w_in_norm, Wq, Wk, Wv, q, k, v)
    return q[:S], k[:S], v[:S]


def qk_norm_rope_q(q_3d, cos, sin, w_q_norm):
    q_p, S = _pad_s(q_3d, _BS_TILE)
    cos_p, _ = _pad_s(cos, _BS_TILE)
    sin_p, _ = _pad_s(sin, _BS_TILE)
    out = torch.empty(q_p.shape, device=q_p.device, dtype=q_p.dtype)
    with _in_dir(_K2Q_DIR):
        _k2_q(q_p, cos_p, sin_p, w_q_norm, out)
    return out[:S]


def qk_norm_rope_k(k_3d, cos, sin, w_k_norm):
    k_p, S = _pad_s(k_3d, _BS_TILE)
    cos_p, _ = _pad_s(cos, _BS_TILE)
    sin_p, _ = _pad_s(sin, _BS_TILE)
    out = torch.empty(k_p.shape, device=k_p.device, dtype=k_p.dtype)
    with _in_dir(_K2K_DIR):
        _k2_k(k_p, cos_p, sin_p, w_k_norm, out)
    return out[:S]


def post_attn(attn_in, x_res, Wo, w_post_norm, Wgate, Wup, Wdown):
    attn_p, S = _pad_s(attn_in, _BS_TILE)
    xres_p, _ = _pad_s(x_res, _BS_TILE)
    y = torch.empty(attn_p.shape, device=attn_p.device, dtype=attn_p.dtype)
    with _in_dir(_K3_DIR):
        _k3(attn_p, xres_p, Wo, w_post_norm, Wgate, Wup, Wdown, y)
    return y[:S]


def pre_attn_fused(x, cos, sin, w_in_norm, Wq, Wk, Wv, w_q_norm, w_k_norm,
                   *, Nq=16, Nkv=8, D=128):
    """Combined RMSNorm + QKV + Q/K-norm + RoPE. Returns (q_3d, k_3d, v_3d)."""
    x_p, S = _pad_s(x, _BS_TILE)
    cos_p, _ = _pad_s(cos, _BS_TILE)
    sin_p, _ = _pad_s(sin, _BS_TILE)
    Sp = x_p.shape[0]
    device, dtype = x.device, x.dtype
    q = torch.empty(Sp, Nq, D, device=device, dtype=dtype)
    k = torch.empty(Sp, Nkv, D, device=device, dtype=dtype)
    v = torch.empty(Sp, Nkv, D, device=device, dtype=dtype)
    with _in_dir(_PREATTN_DIR):
        _pre_attn_fused(x_p, cos_p, sin_p, w_in_norm, Wq, Wk, Wv,
                        w_q_norm, w_k_norm, q, k, v)
    return q[:S], k[:S], v[:S]


_S2_TILE = 64


def decode_attn(q, k_full, v_full, cur_skv):
    """Decode attention v5: K/V as [Nkv=8, Skv_p, D]; kernel does GQA internally."""
    Nq_local = 16
    Nkv_local = 8
    GROUPS = Nq_local // Nkv_local
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    Skv_p = k_full.shape[1]
    mask = torch.zeros(Skv_p, dtype=torch.float32, device=q.device)
    if cur_skv < Skv_p:
        mask[cur_skv:] = -1e30
    mask_3d = mask.view(1, 1, Skv_p).expand(Nkv_local, GROUPS, Skv_p).contiguous()
    with _in_dir(_DEC_ATTN_DIR):
        _decode_attn(q.contiguous(), k_full, v_full, mask_3d, out)
    return out
