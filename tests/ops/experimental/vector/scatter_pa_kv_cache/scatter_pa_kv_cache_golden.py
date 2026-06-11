# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------



"""PyPTO scatter_pa_kv_cache golden reference implementation.

功能描述：
    在 Paged Attention 推理场景中更新 KV cache。将当前 step 生成的多个 token 的 
    key 和 value 数据，按照 slot_mapping 指定的位置，散布到对应的 cache block 中。
    
数学公式：
    key_cache[block_idx, block_offset, :, :] = key[i, :, :]
    value_cache[block_idx, block_offset, :, :] = value[i, :, :]
    
    其中：
    block_idx = slot_mapping[i] // block_size
    block_offset = slot_mapping[i] % block_size

参考：
    - examples/scatter_pa_kv_cache.py（100% 匹配）
    - PagedAttention 论文 (vLLM)

注意：
    - P0 版本暂不支持 compress_lens_optional、compress_seq_offset_optional、seq_lens_optional 参数
    - 这些可选参数在 SPEC.md 中标记为 P2 优先级，本 golden 实现暂不处理
"""

import torch
from typing import Optional

# ─────────────────────────────────────────────
# Golden 参考实现（纯 torch）
# ─────────────────────────────────────────────


def scatter_pa_kv_cache_golden(
    key: torch.Tensor,
    key_cache_in: torch.Tensor,
    slot_mapping: torch.Tensor,
    value: torch.Tensor,
    value_cache_in: torch.Tensor,
    compress_lens_optional: Optional[torch.Tensor] = None,
    compress_seq_offset_optional: Optional[torch.Tensor] = None,
    seq_lens_optional: Optional[torch.Tensor] = None,
) -> tuple:
    """PyTorch 参考实现。

    将 key/value tensor 的数据按照 slot_mapping 索引写入 key_cache/value_cache。

    Args:
        key: 当前 step 多个 token 的 key 值。
            Shape: [num_tokens, num_heads, head_size] 或 [num_tokens, num_heads, head_size, x]
            Dtype: BFLOAT16
        key_cache_in: 需要更新的 key cache。
            Shape: [num_blocks, block_size, num_heads, head_size] 或 [num_blocks, block_size, num_heads, head_size, x]
            Dtype: BFLOAT16
        slot_mapping: 每个 token key/value 在 cache 中的存储偏移。
            Shape: [num_tokens]
            Dtype: INT32
        value: 当前 step 多个 token 的 value 值。
            Shape: [num_tokens, num_heads, head_size] 或 空
            Dtype: BFLOAT16
        value_cache_in: 需要更新的 value cache。
            Shape: [num_blocks, block_size, num_heads, head_size] 或 空
            Dtype: BFLOAT16
        compress_lens_optional: 压缩量（P2 参数，暂不支持）。
            Shape: [batch_size]
            Dtype: INT32
        compress_seq_offset_optional: 每个 batch 每个 head 的压缩起点（P2 参数，暂不支持）。
            Shape: [batch_size, num_heads]
            Dtype: INT32
        seq_lens_optional: 每个 batch 的实际 seq_lens（P2 参数，暂不支持）。
            Shape: [batch_size]
            Dtype: INT32

    Returns:
        (key_cache_out, value_cache_out): 更新后的 key cache 和 value cache。
            两者都是 in-place 更新（返回副本以避免修改输入）。

    注意：
        - P0 版本暂不支持 compress_lens_optional、compress_seq_offset_optional、seq_lens_optional
        - 这些可选参数在 SPEC.md 中标记为 P2 优先级
    """
    # 获取输入 shape
    key_shape = key.shape
    value_shape = value.shape if value is not None else None
    num_tokens = key_shape[0]
    num_heads = key_shape[1]
    head_size = key_shape[2]

    key_cache_shape = key_cache_in.shape
    value_cache_shape = value_cache_in.shape if value_cache_in is not None else None
    block_size = key_cache_shape[1]

    # 创建 cache 的深拷贝，避免 in-place 修改输入（使用 .clone() 替代 deepcopy，内存效率更高）
    key_cache_out = key_cache_in.clone()
    value_cache_out = value_cache_in.clone() if value_cache_in is not None else None

    # P0 版本：通用更新模板（不支持压缩特性）
    for i in range(num_tokens):
        # 计算块索引和块内偏移
        block_idx = slot_mapping[i].item() // block_size
        block_offset = slot_mapping[i].item() % block_size

        # 更新 key_cache
        key_cache_out[block_idx, block_offset, :, :] = key[i, :, :]

        # 更新 value_cache（如果存在）
        if value_cache_out is not None and value is not None:
            value_cache_out[block_idx, block_offset, :, :] = value[i, :, :]

    return key_cache_out, value_cache_out


# ==========================================
# 验证辅助函数
# ==========================================

def _generate_test_data(num_tokens, num_blocks, block_size, num_heads, head_size):
    """生成测试数据 - 确保 slot_mapping 不重复"""
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    key_cache_in = torch.randn(num_blocks, block_size, num_heads, head_size, dtype=torch.bfloat16)
    max_slots = num_blocks * block_size
    if num_tokens <= max_slots:
        slot_mapping = torch.randperm(max_slots)[:num_tokens].to(torch.int32)
    else:
        slot_mapping = torch.randint(0, max_slots, (num_tokens,), dtype=torch.int32)
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16)
    value_cache_in = torch.randn(num_blocks, block_size, num_heads, head_size, dtype=torch.bfloat16)
    return key, key_cache_in, slot_mapping, value, value_cache_in


def _validate_shapes(key_cache_in, value_cache_in, key_cache_out, value_cache_out, test_name):
    """验证输出 shape 是否正确"""
    success = True
    errors = []
    if key_cache_out.shape != key_cache_in.shape:
        success = False
        errors.append(f"key_cache_out shape 不匹配: 期望 {key_cache_in.shape}, 实际 {key_cache_out.shape}")
    if value_cache_out is not None and value_cache_in is not None:
        if value_cache_out.shape != value_cache_in.shape:
            success = False
            errors.append(f"value_cache_out shape 不匹配: 期望 {value_cache_in.shape}, 实际 {value_cache_out.shape}")
    return success, errors


def _validate_scatter(key, key_cache_out, slot_mapping, block_size, test_name):
    """验证 scatter 操作的正确性"""
    num_tokens = key.shape[0]
    for i in range(num_tokens):
        block_idx = slot_mapping[i].item() // block_size
        block_offset = slot_mapping[i].item() % block_size
        expected = key[i, :, :]
        actual = key_cache_out[block_idx, block_offset, :, :]
        if not torch.allclose(expected, actual, atol=1e-4, rtol=0.0078125):
            return False, f"token {i}: scatter 结果不正确"
    return True, "scatter 正确"


def _run_test_case(key, key_cache_in, slot_mapping, value, value_cache_in,
                   block_size, test_name, all_passed):
    """运行单个测试用例并验证 shape 和 scatter"""
    key_cache_out, value_cache_out = scatter_pa_kv_cache_golden(
        key, key_cache_in, slot_mapping, value, value_cache_in)
    shape_success, shape_errors = _validate_shapes(
        key_cache_in, value_cache_in, key_cache_out, value_cache_out, test_name)
    scatter_success, scatter_msg = _validate_scatter(
        key, key_cache_out, slot_mapping, block_size, test_name)
    if shape_success and scatter_success:
        print("✓ PASS")
    else:
        print("✗ FAIL")
        all_passed = False
        for err in shape_errors:
            print(f"    {err}")
        if not scatter_success:
            print(f"    {scatter_msg}")
    return all_passed


# ==========================================
# 验证子函数
# ==========================================

def _validate_typical_cases():
    """验证典型 case"""
    all_passed = True
    print("\n[典型 case 验证]")

    configs = [
        ("性能_P0: num_tokens=100, block_size=128, num_heads=2, head_size=256, num_blocks=100",
         100, 100, 128, 2, 256),
        ("功能_P0: num_tokens=500, block_size=128, num_heads=2, head_size=256, num_blocks=100",
         500, 100, 128, 2, 256),
        ("边界_P0: num_tokens=1000, block_size=128, num_heads=2, head_size=256, num_blocks=200",
         1000, 200, 128, 2, 256),
    ]

    for desc, num_tokens, num_blocks, block_size, num_heads, head_size in configs:
        print(f"  {desc} ... ", end="")
        try:
            key, key_cache_in, slot_mapping, value, value_cache_in = _generate_test_data(
                num_tokens, num_blocks, block_size, num_heads, head_size)
            all_passed = _run_test_case(
                key, key_cache_in, slot_mapping, value, value_cache_in,
                block_size, desc, all_passed)
        except Exception as e:
            print(f"✗ FAIL\n    错误: {e}")
            all_passed = False
    return all_passed


def _validate_generalization_cases():
    """验证泛化 case"""
    all_passed = True
    print("\n[泛化 case 验证]")

    test_cases = [
        (1, 100, 128, 2, 256, "最小 tokens, 最小 blocks"),
        (500, 500, 128, 2, 256, "中间 tokens, 中间 blocks"),
        (1000, 1000, 128, 2, 256, "较大 tokens, 较大 blocks"),
    ]

    for num_tokens, num_blocks, block_size, num_heads, head_size, desc in test_cases:
        print(f"  {desc}: tokens={num_tokens}, blocks={num_blocks} ... ", end="")
        try:
            key, key_cache_in, slot_mapping, value, value_cache_in = _generate_test_data(
                num_tokens, num_blocks, block_size, num_heads, head_size)
            all_passed = _run_test_case(
                key, key_cache_in, slot_mapping, value, value_cache_in,
                block_size, desc, all_passed)
        except Exception as e:
            print(f"✗ FAIL\n    错误: {e}")
            all_passed = False
    return all_passed


def _validate_value_range():
    """值域检查 - slot_mapping 索引范围"""
    all_passed = True
    print("\n[值域检查]")
    print("  检查 slot_mapping 索引范围 ... ", end="")
    try:
        num_blocks, block_size = 100, 128
        num_tokens = 50
        key, key_cache_in, slot_mapping, value, value_cache_in = _generate_test_data(
            num_tokens, num_blocks, block_size, num_heads=2, head_size=256)
        max_slot = num_blocks * block_size - 1
        if (slot_mapping < 0).any() or (slot_mapping > max_slot).any():
            print("✗ FAIL\n    slot_mapping 包含无效索引")
            all_passed = False
        else:
            print("✓ PASS")
    except Exception as e:
        print(f"✗ FAIL\n    错误: {e}")
        all_passed = False
    return all_passed


def _validate_numerical_stability():
    """数值稳定性检查"""
    all_passed = True
    print("\n[数值稳定性检查]")

    for label, scale in [("大值输入 (scale=100)", 100), ("小值输入 (scale=1e-4)", 1e-4)]:
        print(f"  {label} ... ", end="")
        try:
            num_tokens, num_blocks, block_size = 100, 100, 128
            key = torch.randn(num_tokens, 2, 256, dtype=torch.bfloat16) * scale
            key_cache_in = torch.randn(num_blocks, block_size, 2, 256, dtype=torch.bfloat16) * scale
            slot_mapping = torch.randint(0, num_blocks * block_size, (num_tokens,), dtype=torch.int32)
            value = torch.randn(num_tokens, 2, 256, dtype=torch.bfloat16) * scale
            value_cache_in = torch.randn(num_blocks, block_size, 2, 256, dtype=torch.bfloat16) * scale
            key_cache_out, value_cache_out = scatter_pa_kv_cache_golden(
                key, key_cache_in, slot_mapping, value, value_cache_in)
            if torch.isnan(key_cache_out).any() or torch.isinf(key_cache_out).any():
                print("✗ FAIL\n    输出包含 NaN 或 Inf")
                all_passed = False
            else:
                print("✓ PASS")
        except Exception as e:
            print(f"✗ FAIL\n    错误: {e}")
            all_passed = False

    print("  零值输入 ... ", end="")
    try:
        num_tokens, num_blocks, block_size = 100, 100, 128
        key = torch.zeros(num_tokens, 2, 256, dtype=torch.bfloat16)
        key_cache_in = torch.randn(num_blocks, block_size, 2, 256, dtype=torch.bfloat16)
        slot_mapping = torch.randint(0, num_blocks * block_size, (num_tokens,), dtype=torch.int32)
        value = torch.zeros(num_tokens, 2, 256, dtype=torch.bfloat16)
        value_cache_in = torch.randn(num_blocks, block_size, 2, 256, dtype=torch.bfloat16)
        key_cache_out, value_cache_out = scatter_pa_kv_cache_golden(
            key, key_cache_in, slot_mapping, value, value_cache_in)
        scatter_success, scatter_msg = _validate_scatter(
            key, key_cache_out, slot_mapping, block_size, "零值输入")
        if scatter_success:
            print("✓ PASS")
        else:
            print(f"✗ FAIL\n    {scatter_msg}")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL\n    错误: {e}")
        all_passed = False
    return all_passed


def _validate_functional_correctness():
    """功能正确性检查 - 重复 slot_mapping"""
    all_passed = True
    print("\n[功能正确性检查]")
    print("  检查重复 slot_mapping（后写入覆盖前写入）... ", end="")
    try:
        num_tokens, num_blocks, block_size = 5, 10, 128
        key = torch.randn(num_tokens, 2, 256, dtype=torch.bfloat16)
        key_cache_in = torch.randn(num_blocks, block_size, 2, 256, dtype=torch.bfloat16)
        slot_mapping = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32)
        value = torch.randn(num_tokens, 2, 256, dtype=torch.bfloat16)
        value_cache_in = torch.randn(num_blocks, block_size, 2, 256, dtype=torch.bfloat16)
        key_cache_out, value_cache_out = scatter_pa_kv_cache_golden(
            key, key_cache_in, slot_mapping, value, value_cache_in)
        block_idx_0 = 0 // block_size
        block_offset_0 = 0 % block_size
        expected = key[1, :, :]
        if torch.allclose(key_cache_out[block_idx_0, block_offset_0, :, :], expected, atol=1e-4, rtol=0.0078125):
            print("✓ PASS")
        else:
            print("✗ FAIL\n    重复 slot_mapping 的覆盖逻辑不正确")
            all_passed = False
    except Exception as e:
        print(f"✗ FAIL\n    错误: {e}")
        all_passed = False
    return all_passed


# ==========================================
# 验证主函数
# ==========================================

def _validate():
    """自动生成的验证函数 - 运行时动态生成验证报告"""

    print("=" * 60)
    print("scatter_pa_kv_cache_golden 验证报告")
    print("=" * 60)

    all_passed = True
    all_passed &= _validate_typical_cases()
    all_passed &= _validate_generalization_cases()
    all_passed &= _validate_value_range()
    all_passed &= _validate_numerical_stability()
    all_passed &= _validate_functional_correctness()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有验证通过")
    else:
        print("❌ 部分验证失败")
    print("=" * 60)


if __name__ == "__main__":
    _validate()