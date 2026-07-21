#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

"""精度对比引擎 — 方案A 混合容差标准。

依据《生态算子精度标准》§2（experimental_standard.md）实现：
  - 逐元素通过条件: |actual - golden| <= atol + rtol * |golden|
  - 整体通过条件: matched_ratio >= required_matched_ratio AND max_abs_error <= max_abs_error_limit
  - max_abs_error_limit = max(fixed_limit, 32 * ULP)  ('or' 语义取较大者)

阈值表硬编码，不接受外部参数（防作弊）。
纯 CPU + torch 实现，不依赖 torch_npu。
"""

from typing import List, Optional, Tuple, Union

import torch

# ============================================================================
# 阈值表 (experimental_standard.md §2.2)
# ============================================================================

_REQUIRED_MATCHED_RATIO = 0.99

# 每种 dtype 的阈值元组含义: (rtol, atol, fixed_max_abs_limit)
_THRESHOLDS = {
    torch.float16: (2.0**-9, 2.0**-9, 1e-1),  # 1.95e-3, 1.95e-3, 0.1
    torch.bfloat16: (2.0**-6, 2.0**-6, 1e-0),  # 1.56e-2, 1.56e-2, 1.0
    torch.float32: (2.0**-10, 2.0**-16, 1e-2),  # 9.77e-4, 1.53e-5, 0.01
    torch.float64: (2.0**-10, 2.0**-16, 1e-2),  # 同 float32
}

# HiFloat32 / Float8 类型（torch 可能不支持，用字符串映射）
_DTYPE_NAME_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "float": torch.float32,
    "double": torch.float64,
}


def _get_threshold(dtype):
    """查表返回 (rtol, atol, required_matched_ratio, max_abs_error_limit)。

    对于整数类型，返回 None 表示走精确匹配路径。
    """
    if dtype in _THRESHOLDS:
        rtol, atol, fixed_limit = _THRESHOLDS[dtype]
        ulp = _compute_ulp(dtype)
        max_abs_limit = max(fixed_limit, 32.0 * ulp)
        return rtol, atol, _REQUIRED_MATCHED_RATIO, max_abs_limit
    # 整数类型 → 精确匹配
    return None


def _compute_ulp(dtype):
    """计算给定 dtype 的 ULP (Unit in the Last Place)。

    ULP = 2^(exponent - mantissa_bits)，取 1.0 附近的 ULP 值。
    """
    try:
        one = torch.tensor(1.0, dtype=dtype)
        nxt = torch.nextafter(one, torch.tensor(2.0, dtype=dtype))
        return (nxt - one).item()
    except (TypeError, RuntimeError):
        # 降级：用 finfo 推算
        fi = torch.finfo(dtype)
        return fi.eps


def _to_cpu(tensor):
    """转到 CPU，不改变 dtype。"""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu() if tensor.is_cuda or "npu" in str(tensor.device) else tensor
    return tensor


def _is_npu(tensor):
    if isinstance(tensor, torch.Tensor):
        return "npu" in str(tensor.device)
    return False


# ============================================================================
# 核心对比逻辑
# ============================================================================

def _compare_single_tensor(actual: torch.Tensor, golden: torch.Tensor) -> Tuple[bool, str, dict]:
    """对比单个 tensor，返回 (passed, summary, metrics)。

    依据 experimental_standard.md §2:
      - 浮点: 混合容差 + matched_ratio + max_abs_error_limit
      - 整数: torch.equal 精确匹配
      - NaN 位置必须一致
      - Inf: 同号视为匹配；异号 FAIL；一方 Inf 一方有限值→替换为 finfo.max 后正常判定
    """
    # 统一到 CPU
    actual = _to_cpu(actual)
    golden = _to_cpu(golden)

    # 形状检查
    if actual.shape != golden.shape:
        return False, f"shape mismatch: actual={tuple(actual.shape)} vs golden={tuple(golden.shape)}", {}

    total = actual.numel()
    if total == 0:
        return True, "empty tensor", {}

    # 整数类型 → 精确匹配
    threshold = _get_threshold(actual.dtype)
    if threshold is None:
        passed = torch.equal(actual, golden)
        if passed:
            return True, "exact match (integer)", {"matched_ratio": 1.0, "max_abs_error": 0.0}
        mismatch = (actual != golden).sum().item()
        return False, f"integer mismatch: {mismatch}/{total} elements differ", {"matched_ratio": 1 - mismatch / total}

    rtol, atol, req_ratio, max_abs_limit = threshold

    # 转到 float64 做高精度比较（避免自身舍入误差）
    a = actual.to(torch.float64)
    g = golden.to(torch.float64)

    # NaN 处理: 位置必须一致
    a_nan = torch.isnan(a)
    g_nan = torch.isnan(g)
    nan_mismatch = (a_nan != g_nan).sum().item()
    if nan_mismatch > 0:
        return False, f"NaN position mismatch: {nan_mismatch} positions differ", {}

    # Inf 处理
    a_inf = torch.isinf(a)
    g_inf = torch.isinf(g)
    both_inf = a_inf & g_inf
    inf_same_sign = both_inf & (torch.sign(a) == torch.sign(g))
    inf_diff_sign = both_inf & (torch.sign(a) != torch.sign(g))
    one_inf = a_inf ^ g_inf  # 一方 Inf 一方有限值

    if inf_diff_sign.any().item():
        cnt = inf_diff_sign.sum().item()
        return False, f"Inf sign mismatch: {cnt} positions", {}

    # 替换一方 Inf 为 finfo(actual).max（保留符号）
    fi = torch.finfo(actual.dtype)
    a_clean = a.clone()
    g_clean = g.clone()

    # 一方 Inf 一方有限值: 替换 Inf 为 ±finfo.max
    a_one_inf = a_inf & ~g_inf
    g_one_inf = g_inf & ~a_inf
    a_clean[a_one_inf & (a > 0)] = fi.max
    a_clean[a_one_inf & (a < 0)] = fi.min
    g_clean[g_one_inf & (g > 0)] = fi.max
    g_clean[g_one_inf & (g < 0)] = fi.min

    # 同号 Inf: 视为匹配，从后续比较中排除
    match_mask = inf_same_sign | a_nan | g_nan  # 这些位置视为通过
    compare_mask = ~match_mask

    if compare_mask.any():
        diff = (a_clean - g_clean).abs()
        tolerance = atol + rtol * g_clean.abs()
        element_pass = diff <= tolerance

        # 在可比较位置上的统计
        compare_count = compare_mask.sum().item()
        pass_count = (element_pass & compare_mask).sum().item() + match_mask.sum().item()
        matched_ratio = pass_count / total

        max_abs_error = diff[compare_mask].max().item() if compare_mask.any() else 0.0
    else:
        # 所有位置都是 NaN 或同号 Inf
        matched_ratio = 1.0
        max_abs_error = 0.0

    # 判定
    ratio_ok = matched_ratio >= req_ratio
    max_abs_ok = max_abs_error <= max_abs_limit
    passed = ratio_ok and max_abs_ok

    summary = (
        f"matched_ratio={matched_ratio:.6f} (req>={req_ratio}), "
        f"max_abs_error={max_abs_error:.6e} (limit={max_abs_limit:.6e})"
    )

    metrics = {
        "matched_ratio": matched_ratio,
        "max_abs_error": max_abs_error,
        "required_matched_ratio": req_ratio,
        "max_abs_error_limit": max_abs_limit,
        "rtol": rtol,
        "atol": atol,
    }

    return passed, summary, metrics


def _normalize_outputs(output) -> List[Optional[torch.Tensor]]:
    """将输出标准化为 tensor 列表。

    支持: 单个 tensor / tuple / list / None 占位。
    """
    if output is None:
        return [None]
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        return [o if isinstance(o, torch.Tensor) else None for o in output]
    return [None]


def check_precision(
    actual: Union[torch.Tensor, tuple, list],
    golden: Union[torch.Tensor, tuple, list],
    dtype_str: Optional[str] = None,
) -> Tuple[bool, str]:
    """精度对比主入口（方案A混合容差标准）。

    Args:
        actual: 算子输出（NPU tensor 或 CPU tensor，单输出或多输出）
        golden: golden 参考输出（CPU FP32 更高精度）
        dtype_str: 可选，指定 dtype 名称字符串（默认从 actual 推断）

    Returns:
        (passed, summary): passed=True/False, summary=人类可读的精度指标摘要

    标准 (experimental_standard.md §2):
      - 逐元素: |actual - golden| <= atol + rtol * |golden|
      - 整体: matched_ratio >= 0.99 AND max_abs_error <= max_abs_error_limit
      - max_abs_error_limit = max(fixed_limit, 32 * ULP)
      - 整数: 精确匹配 (torch.equal)
      - NaN 位置必须一致
      - Inf: 同号匹配, 异号 FAIL, 一方Inf→替换为 finfo.max
    """
    actuals = _normalize_outputs(actual)
    goldens = _normalize_outputs(golden)

    if len(actuals) != len(goldens):
        return False, f"output count mismatch: actual={len(actuals)} vs golden={len(goldens)}"

    all_passed = True
    summaries = []

    for i, (a, g) in enumerate(zip(actuals, goldens)):
        if a is None and g is None:
            continue
        if a is None or g is None:
            all_passed = False
            summaries.append(f"output[{i}]: one side is None")
            continue

        passed, summary, _ = _compare_single_tensor(a, g)
        if not passed:
            all_passed = False
        prefix = "" if len(actuals) == 1 else f"output[{i}]: "
        summaries.append(f"{prefix}{summary}")

    return all_passed, "; ".join(summaries)
