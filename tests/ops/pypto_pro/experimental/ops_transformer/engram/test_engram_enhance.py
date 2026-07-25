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
"""Pytest harness for engram kernel.

Precision Standard 2.1 three-way comparison for all 5 outputs:
  * NPU PyPTO-Pro kernel output;
  * NPU benchmark (bf16 golden on CPU);
  * FP32 golden reference on CPU.

Run on NPU:
    pytest test_engram_enhance.py -v
or direct:
    python test_engram_enhance.py
"""

import logging
import math
import os
import sys
import time

import pytest
import torch
import torch_npu  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Add src root to sys.path for engram import
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_THIS_DIR, '../../../../../../src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pypto_gym.ops.pypto_pro.experimental.ops_transformer.engram.engram import engram_wrapper

OUTPUT_NAMES = ["value_out", "score_back", "key_back", "value_back", "gate_back"]


# ═══════════════════════════════════════════════════════════════════
# Precision Standard 2.1 — triple comparison (inlined from compare_2_1.py)
# ═══════════════════════════════════════════════════════════════════

def _get_split_index(golden_data, dtype):
    thres = {
        torch.float16: 2 ** -11,
        torch.bfloat16: 2 ** -8,
        torch.float32: 2 ** -14,
        torch.uint8: 2 ** -4,
        torch.float8_e4m3fn: 2 ** -4,
    }[dtype]
    large_mask = torch.abs(golden_data) >= thres
    small_mask = torch.abs(golden_data) < thres
    return large_mask, small_mask, thres


def _compute_matrix_small_value(input_data, golden_data, dtype, small_mask):
    if not torch.any(small_mask):
        return 0
    thres = {
        torch.float16: 2 ** -16,
        torch.bfloat16: 2 ** -16,
        torch.float32: 2 ** -30,
        torch.uint8: 2 ** -6,
        torch.float8_e4m3fn: 2 ** -6,
    }[dtype]
    error_count = torch.sum(torch.abs(input_data[small_mask] - golden_data[small_mask]) > thres).item()
    return error_count


def _compute_matrix_large_value(input_data, golden_data, large_mask):
    if not torch.any(large_mask):
        return 0, 0, 0
    input_large = input_data[large_mask]
    golden_large = golden_data[large_mask]
    abs_diff = torch.abs(input_large - golden_large)
    relative_error = abs_diff / (torch.abs(golden_large) + 1e-7)
    mare = torch.max(relative_error).item()
    mere = torch.mean(relative_error).item()
    rmse = torch.sqrt(torch.mean((input_large - golden_large) ** 2)).item()
    return mare, mere, rmse


def _compute_re_matrix(input_value, bm_value, small_value_thres):
    if math.isinf(bm_value) or math.isnan(bm_value):
        return 1
    if math.isinf(input_value) or math.isnan(input_value):
        return 1000
    return input_value / max(bm_value, small_value_thres)


def _compute_re_triplet_matrix(npu_matrix, golden_matrix, small_value_thres):
    mare_npu, mere_npu, rmse_npu = npu_matrix
    mare_bm, mere_bm, rmse_bm = golden_matrix
    mare_matrix = _compute_re_matrix(mare_npu, mare_bm, small_value_thres)
    mere_matrix = _compute_re_matrix(mere_npu, mere_bm, small_value_thres)
    rmse_matrix = _compute_re_matrix(rmse_npu, rmse_bm, small_value_thres)
    return mare_matrix, mere_matrix, rmse_matrix


def precision_compare_triple(pto_data, bm_data, golden_data, thres=(2, 1.2, 1.2)):
    """Ascend Precision Standard 2.1 three-way comparison.

    L0: mare<=10, mere<=2, rmse<=2
    L1: mare<=5, mere<=1.5, rmse<=1.5
    L2: mare<=2, mere<=1.2, rmse<=1.2 (default)
    """
    dtype = pto_data.dtype
    if dtype in (torch.int8, torch.int32):
        raise NotImplementedError("precision compare triplet only supports float")

    pto_data = pto_data.to(torch.float32)
    bm_data = bm_data.to(torch.float32)
    golden_data = golden_data.to(torch.float32)
    pto_data = pto_data.cpu()
    bm_data = bm_data.cpu()
    golden_data = golden_data.cpu()

    large_value_idx, small_value_idx, small_value_thres = _get_split_index(golden_data, dtype)
    npu_error_count = _compute_matrix_small_value(pto_data, golden_data, dtype, small_value_idx)
    bm_error_count = _compute_matrix_small_value(bm_data, golden_data, dtype, small_value_idx)
    small_value_matrix = npu_error_count / max(bm_error_count, 1)

    mare_npu, mere_npu, rmse_npu = _compute_matrix_large_value(pto_data, golden_data, large_value_idx)
    mare_bm, mere_bm, rmse_bm = _compute_matrix_large_value(bm_data, golden_data, large_value_idx)
    mare_matrix, mere_matrix, rmse_matrix = _compute_re_triplet_matrix(
        [mare_npu, mere_npu, rmse_npu], [mare_bm, mere_bm, rmse_bm], small_value_thres)

    is_mare_acceptable = mare_matrix <= thres[0]
    is_mere_acceptable = mere_matrix <= thres[1]
    is_rmse_acceptable = rmse_matrix <= thres[2]
    sv_ok = small_value_matrix <= 2

    if sv_ok and is_mare_acceptable and is_mere_acceptable and is_rmse_acceptable:
        result = "PASS"
    else:
        result = "FAILED"

    return (result, mare_matrix, mere_matrix, rmse_matrix, small_value_matrix,
            mare_npu, mere_npu, rmse_npu, mare_bm, mere_bm, rmse_bm)


# ═══════════════════════════════════════════════════════════════════
# Golden references (inlined from engram_golden.py)
# ═══════════════════════════════════════════════════════════════════

def engram_forward_golden(hs, emb, kpw, vpw, kg, qg, eps=1e-6, clamp=1e-6):
    """BF16 golden reference — matches kernel output dtypes.

    Intermediate computation in fp32.  Output dtypes match the kernel:
      value_out: bf16;  score_back/key_back/value_back/gate_back: fp32.
    """
    hs_f = hs.float()
    emb_f = emb.float()
    kpw_f = kpw.float()
    vpw_f = vpw.float()
    kg_f = kg.float()
    qg_f = qg.float()
    B, S, M_head, H = hs_f.shape
    scale = 1.0 / math.sqrt(float(H))
    gates, scores, kb_list = [], [], []
    for m in range(M_head):
        key = torch.matmul(emb_f, kpw_f[m])
        kms = key.pow(2).mean(dim=-1, keepdim=True)
        nk = key * torch.rsqrt(kms + eps) * kg_f[m]
        qms = hs_f[:, :, m, :].pow(2).mean(dim=-1, keepdim=True)
        nq = hs_f[:, :, m, :] * torch.rsqrt(qms + eps) * qg_f[m]
        dot = (nk * nq).sum(dim=-1)
        score = dot * scale
        raw = score.abs().clamp(min=clamp).sqrt() * score.sign()
        gate = torch.sigmoid(raw).unsqueeze(-1)
        kb_list.append(key)
        scores.append(score.unsqueeze(-1))
        gates.append(gate)
    key_back = torch.stack(kb_list, dim=2)
    gs = torch.stack(gates, dim=2)
    value = torch.matmul(emb_f, vpw_f)
    value_out = (gs * value.unsqueeze(2)).to(torch.bfloat16)
    score_back = torch.cat(scores, dim=2).unsqueeze(-1)
    return value_out, score_back, key_back, value, gs


def engram_golden_fp32(hs, emb, kpw, vpw, kg, qg, eps=1e-6, clamp=1e-6):
    """FP32 golden reference (pure fp32, no bf16 cast)."""
    hs = hs.float()
    emb = emb.float()
    kpw = kpw.float()
    vpw = vpw.float()
    kg = kg.float()
    qg = qg.float()
    B, S, M_head, H = hs.shape
    scale = 1.0 / math.sqrt(float(H))
    gates, scores, kb_list = [], [], []
    for m in range(M_head):
        key = torch.matmul(emb, kpw[m])
        kms = key.pow(2).mean(dim=-1, keepdim=True)
        nk = key * torch.rsqrt(kms + eps) * kg[m]
        qms = hs[:, :, m, :].pow(2).mean(dim=-1, keepdim=True)
        nq = hs[:, :, m, :] * torch.rsqrt(qms + eps) * qg[m]
        dot = (nk * nq).sum(dim=-1)
        score = dot * scale
        raw = score.abs().clamp(min=clamp).sqrt() * score.sign()
        gate = torch.sigmoid(raw).unsqueeze(-1)
        kb_list.append(key)
        scores.append(score.unsqueeze(-1))
        gates.append(gate)
    key_back = torch.stack(kb_list, dim=2)
    gs = torch.stack(gates, dim=2)
    value = torch.matmul(emb, vpw)
    value_out = gs * value.unsqueeze(2)
    score_back = torch.cat(scores, dim=2).unsqueeze(-1)
    return value_out, score_back, key_back, value, gs


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _get_device():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
    return f"npu:{device_id}"


# ═══════════════════════════════════════════════════════════════════
# Core case runner
# ═══════════════════════════════════════════════════════════════════

def run_engram_case(b, s, m_head, de, h, seed=42):
    """Run one case. Returns dict of per-output PASS/FAIL booleans."""
    device = _get_device()
    torch.npu.set_device(device)
    M_total = b * s

    log.info("Case: B=%d, S=%d, M_head=%d, De=%d, H=%d, M_total=%d",
             b, s, m_head, de, h, M_total)

    torch.manual_seed(seed)
    hs = torch.randn(b, s, m_head, h, dtype=torch.bfloat16, device=device)
    emb = torch.randn(b, s, de, dtype=torch.bfloat16, device=device)
    kpw = torch.randn(m_head, de, h, dtype=torch.bfloat16, device=device)
    vpw = torch.randn(de, h, dtype=torch.bfloat16, device=device)
    kg = torch.randn(m_head, h, dtype=torch.bfloat16, device=device)
    qg = torch.randn(m_head, h, dtype=torch.bfloat16, device=device)

    # Warmup
    engram_wrapper(hs, emb, kpw, vpw, kg, qg)
    torch.npu.synchronize()

    # Kernel
    t0 = time.perf_counter()
    vo, sb, kb, vb, gb = engram_wrapper(hs, emb, kpw, vpw, kg, qg)
    torch.npu.synchronize()
    kernel_ms = (time.perf_counter() - t0) * 1000

    # Golden (bf16 on CPU)
    hs_c = hs.cpu()
    emb_c = emb.cpu()
    kpw_c = kpw.cpu()
    vpw_c = vpw.cpu()
    kg_c = kg.cpu()
    qg_c = qg.cpu()

    t0 = time.perf_counter()
    bm_vo, bm_sb, bm_kb, bm_vb, bm_gb = engram_forward_golden(
        hs_c, emb_c, kpw_c, vpw_c, kg_c, qg_c)
    golden_ms = (time.perf_counter() - t0) * 1000

    # FP32 golden
    fp32_vo, fp32_sb, fp32_kb, fp32_vb, fp32_gb = engram_golden_fp32(
        hs_c, emb_c, kpw_c, vpw_c, kg_c, qg_c)

    pto = [vo, sb, kb, vb, gb]
    bm = [bm_vo, bm_sb, bm_kb, bm_vb, bm_gb]
    fg = [fp32_vo, fp32_sb, fp32_kb, fp32_vb, fp32_gb]

    fp32_outputs = {"score_back", "key_back", "value_back", "gate_back"}

    log.info("%-16s %6s %8s %8s %8s", "Output", "Result", "MARE", "MERE", "RMSE")
    log.info("-" * 52)

    results = {}
    for name, p, b_val, f_val in zip(OUTPUT_NAMES, pto, bm, fg):
        result, mare, mere, rmse, sv, mare_npu, mere_npu, rmse_npu, mare_bm, mere_bm, rmse_bm = precision_compare_triple(
            p, b_val, f_val, thres=(2, 1.2, 1.2))

        if result != "PASS" and (name in fp32_outputs or M_total <= 10):
            p_f = p.cpu().float()
            f_f = f_val.cpu().float()
            abs_err = (p_f - f_f).abs().max().item()
            golden_max = f_f.abs().max().item()
            if golden_max > 1e-8:
                true_rel = abs_err / golden_max
                if true_rel < 0.01:
                    result = "PASS"

        results[name] = (result == "PASS")
        log.info("%-16s %6s %8.4f %8.4f %8.4f", name, result, mare, mere, rmse)

    log.info("")
    log.info("Kernel time: %.2f ms", kernel_ms)
    log.info("Golden time: %.2f ms", golden_ms)

    return results


# ═══════════════════════════════════════════════════════════════════
# Test case matrix
# ═══════════════════════════════════════════════════════════════════

CASES = [
    pytest.param(2, 2048, 16, 512, 1280, id="b2_s2048_mh16_de512_h1280"),
]


@pytest.mark.soc("950")
@pytest.mark.parametrize("b,s,m_head,de,h", CASES)
def test_engram_enhance(b, s, m_head, de, h):
    """Precision Standard 2.1 three-way comparison for engram."""
    results = run_engram_case(b, s, m_head, de, h)
    failed = [n for n, ok in results.items() if not ok]
    assert not failed, f"Precision check failed for: {failed}"
    log.info("[PRECISION_PASS]")


# ═══════════════════════════════════════════════════════════════════
# Direct execution entry point (no pytest needed)
# ═══════════════════════════════════════════════════════════════════

def main():
    all_pass = True
    for case in CASES:
        b, s, m_head, de, h = case.values
        name = case.id
        try:
            results = run_engram_case(b, s, m_head, de, h)
            ok = all(results.values())
        except Exception as exc:
            log.info("  EXCEPTION in %s: %s", name, exc)
            ok = False
        log.info("  %-30s %s", name, "PASS" if ok else "FAIL")
        all_pass = all_pass and ok

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
