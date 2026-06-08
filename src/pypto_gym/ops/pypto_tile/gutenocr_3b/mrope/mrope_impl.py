#!/usr/bin/env python3
# coding: utf-8
"""
gutenocr_3b Mrope PyPTO Kernel（基于 test_lightning_indexer_prolog.py）

关键实现：
1. rotate_half: 使用 pypto.view + pypto.concat（参考 test_line 153-164）
2. mrope_split: 使用多个 pypto.view 替代 torch.split
3. mrope_concat: 使用 pypto.concat 替代 torch.cat
4. 整体算子序列: function context + loop + view + concat + assemble

不使用 @pypto.frontend.jit decorator（正确模式）
"""

import torch
import logging
import sys
import os

os.environ['TILE_FWK_DEVICE_ID'] = '0'

_logger = logging.getLogger(__name__)

PTO_KERNEL_AVAILABLE = False  # 默认不可用，测试成功后改为 True

_pypto = None


def _import_pypto():
    """延迟导入PyPTO"""
    global _pypto
    if _pypto is None:
        sys.path.insert(0, '/data/h00520348/optimize525/pypto/python')
        import pypto
        _pypto = pypto
    return _pypto


def rotate_half_pto(input_tensor):
    """
    Rotate half 实现（参考 test_lightning_indexer_prolog.py line 153-164）

    使用 pypto.view 替代切片，pypto.concat 替代 torch.cat
    """
    pypto = _import_pypto()

    shape = list(input_tensor.shape)
    shape_size = len(shape)
    assert shape_size >= 1
    assert shape[shape_size - 1] % 2 == 0

    shape[shape_size - 1] //= 2  # 对半切

    offset1 = [0] * shape_size
    offset2 = [0] * shape_size
    offset2[shape_size - 1] = shape[shape_size - 1]  # 后半部分的 offset

    x1 = pypto.view(input_tensor, shape, offset1)  # 前半部分
    x2 = pypto.view(input_tensor, shape, offset2)  # 后半部分

    # 使用 concat 拼接（替代 torch.cat）
    return pypto.concat([x2 * (-1.0), x1 + 0.0], dim=-1)


def mrope_pto_correct(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Mrope PyPTO kernel（正确实现）

    参数:
        q: [batch, num_heads, seq_len, head_dim] - torch.Tensor
        k: [batch, num_kv_heads, seq_len, head_dim] - torch.Tensor
        cos: [3, batch, seq_len, head_dim] - torch.Tensor
        sin: [3, batch, seq_len, head_dim] - torch.Tensor
        mrope_section: [16, 24, 24]

    返回:
        q_embed, k_embed - torch.Tensor
    """
    if not PTO_KERNEL_AVAILABLE:
        _logger.info("PyPTO Mrope kernel 未启用，使用 torch fallback")
        return mrope_torch_fallback(q, k, cos, sin, mrope_section, unsqueeze_dim)

    try:
        pypto = _import_pypto()

        # 初始化 runtime
        pypto.runtime._device_init()

        batch = q.shape[0]
        num_heads = q.shape[1]
        seq_len = q.shape[2]
        head_dim = q.shape[3]

        mrope_section_expanded = [s * 2 for s in mrope_section]  # [32, 48, 48]

        # 定义阶段：创建 PyPTO tensor
        dtype = pypto.DT_FP32

        q_tensor = pypto.tensor([batch, num_heads, seq_len, head_dim], dtype, "Q_INPUT")
        k_tensor = pypto.tensor([batch, num_heads, seq_len, head_dim], dtype, "K_INPUT")
        cos_tensor = pypto.tensor([3, batch, seq_len, head_dim], dtype, "COS_INPUT")
        sin_tensor = pypto.tensor([3, batch, seq_len, head_dim], dtype, "SIN_INPUT")
        q_output = pypto.tensor([batch, num_heads, seq_len, head_dim], dtype, "Q_OUTPUT")
        k_output = pypto.tensor([batch, num_heads, seq_len, head_dim], dtype, "K_OUTPUT")

        # 在 function context 中定义算子（不使用 JIT decorator）
        with pypto.function("MROPE_KERNEL", q_tensor, k_tensor, cos_tensor, sin_tensor, q_output, k_output):

            pypto.set_vec_tile_shapes(32, 128)

            # Step 1: 使用 view 替代 torch.split
            # torch版本: cos.split([32, 48, 48], dim=-1)
            # PyPTO版本: 多个 view

            cos_parts = []
            sin_parts = []
            offset = 0

            for i, section_size in enumerate(mrope_section_expanded):
                # 从 cos_tensor 的第 i 个 batch（维度0）创建视图
                # cos_tensor shape: [3, batch, seq_len, head_dim]

                # 创建 shape（匹配 tensor 维度）
                view_shape = [1, batch, seq_len, section_size]  # 维度0取单个batch

                # offset: [batch_idx, 0, 0, head_dim_offset]
                offsets = [i, 0, 0, offset]

                cos_view = pypto.view(cos_tensor, view_shape, offsets)
                sin_view = pypto.view(sin_tensor, view_shape, offsets)

                # 去掉第0维（batch维度）
                # view_shape: [1, batch, seq_len, section_size] -> [batch, seq_len, section_size]
                cos_view_3d = pypto.reshape(cos_view, [batch, seq_len, section_size])
                sin_view_3d = pypto.reshape(sin_view, [batch, seq_len, section_size])

                cos_parts.append(cos_view_3d)
                sin_parts.append(sin_view_3d)

                offset += section_size

            # Step 2: 使用 concat 拼接
            cos_new = pypto.concat(cos_parts, dim=-1)  # [batch, seq_len, head_dim]
            sin_new = pypto.concat(sin_parts, dim=-1)

            # Step 3: Unsqueeze
            cos_new = pypto.unsqueeze(cos_new, dim=1)  # [batch, 1, seq_len, head_dim]
            sin_new = pypto.unsqueeze(sin_new, dim=1)

            # Step 4: Cast to FP32（参考 test_line 195-199）
            q_cast = pypto.cast(q_tensor, pypto.DT_FP32)
            k_cast = pypto.cast(k_tensor, pypto.DT_FP32)

            if q_tensor.dtype == pypto.DT_FP32:
                q_cast = q_cast + 0.0  # 参考 test_line 197

            if k_tensor.dtype == pypto.DT_FP32:
                k_cast = k_cast + 0.0

            cos_cast = pypto.cast(cos_new, pypto.DT_FP32)
            sin_cast = pypto.cast(sin_new, pypto.DT_FP32)

            # Step 5: Apply RoPE（参考 test_line 217）
            q_rotated = rotate_half_pto(q_cast)
            k_rotated = rotate_half_pto(k_cast)

            q_embed = q_cast * cos_cast + q_rotated * sin_cast
            k_embed = k_cast * cos_cast + k_rotated * sin_cast

            # Step 6: Cast back to original dtype
            q_embed_cast = pypto.cast(q_embed, q_tensor.dtype)
            k_embed_cast = pypto.cast(k_embed, k_tensor.dtype)

            # Step 7: Assemble results（使用 loop 结构）
            # 参考 test_vector_operation_part_one.py

            # 使用 loop 分块处理（简化版）
            num_blocks = 1  # 简化，不分块

            for block_idx in pypto.loop(num_blocks, name="LOOP_MROPE", idx_name="block_idx"):  # pylint: disable=unused-loop-variable
                # 创建临时 tensor
                tmp_q = pypto.tensor([batch, num_heads, seq_len, head_dim], dtype)
                tmp_k = pypto.tensor([batch, num_heads, seq_len, head_dim], dtype)

                # Move 结果到临时 tensor
                tmp_q.move(q_embed_cast)
                tmp_k.move(k_embed_cast)

                # Assemble 到输出
                pypto.assemble(tmp_q, [0, 0, 0, 0], q_output)
                pypto.assemble(tmp_k, [0, 0, 0, 0], k_output)

        # 执行阶段：创建 torch tensor
        q_torch_data = q.detach().clone()
        k_torch_data = k.detach().clone()
        cos_torch_data = cos.detach().clone()
        sin_torch_data = sin.detach().clone()

        q_out_torch = torch.zeros_like(q)
        k_out_torch = torch.zeros_like(k)

        # 转换为 PyPTO tensor（执行时）
        q_pto_exec = pypto.from_torch(q_torch_data, "q_exec")
        k_pto_exec = pypto.from_torch(k_torch_data, "k_exec")
        cos_pto_exec = pypto.from_torch(cos_torch_data, "cos_exec")
        sin_pto_exec = pypto.from_torch(sin_torch_data, "sin_exec")
        q_out_pto = pypto.from_torch(q_out_torch, "q_out_exec")
        k_out_pto = pypto.from_torch(k_out_torch, "k_out_exec")

        # 执行 kernel
        pypto.runtime._device_run_once_data_from_host(
            q_pto_exec, k_pto_exec, cos_pto_exec, sin_pto_exec, q_out_pto, k_out_pto
        )

        # 结果已经在 q_out_torch, k_out_torch 中
        pypto.runtime._device_fini()

        return q_out_torch, k_out_torch

    except Exception as e:
        _logger.warning(f"PyPTO Mrope kernel 失败: {e}")
        import traceback
        traceback.print_exc()

        try:
            pypto.runtime._device_fini()
        except Exception:
            pass

        return mrope_torch_fallback(q, k, cos, sin, mrope_section, unsqueeze_dim)


def mrope_torch_fallback(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Torch fallback 实现"""
    mrope_section_expanded = mrope_section * 2

    cos_new = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section_expanded, dim=-1))],
                        dim=-1).unsqueeze(unsqueeze_dim)
    sin_new = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section_expanded, dim=-1))],
                        dim=-1).unsqueeze(unsqueeze_dim)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos_new) + (rotate_half(q) * sin_new)
    k_embed = (k * cos_new) + (rotate_half(k) * sin_new)

    return q_embed, k_embed


class MRoPEWrapperPyPTO:
    """Mrope Wrapper 类（PyPTO版本）"""

    @staticmethod
    def apply(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        """调用 PyPTO kernel"""
        return mrope_pto_correct(q, k, cos, sin, mrope_section, unsqueeze_dim)


__all__ = [
    "mrope_pto_correct",
    "mrope_torch_fallback",
    "MRoPEWrapperPyPTO",
    "PTO_KERNEL_AVAILABLE",
    "rotate_half_pto",
]