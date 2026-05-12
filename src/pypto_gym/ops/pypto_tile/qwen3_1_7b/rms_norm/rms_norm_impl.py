#!/usr/bin/env python3
"""
Qwen3 RMSNorm PyPTO Kernel 实现

使用 PyPTO 内置 rms_norm API，适配 Qwen3 的多种调用场景：
- input_layernorm (hidden_size=2048)
- q_norm (num_heads=16, head_dim=128)
- k_norm (num_kv_heads=8, head_dim=128)
"""

import os
import sys
import pypto
import torch


def get_device_id():
    """获取环境变量中的 device ID"""
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("  export TILE_FWK_DEVICE_ID=<chip_id>")
        return None
    return int(os.environ['TILE_FWK_DEVICE_ID'])


@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128},
    pass_options={"cube_l1_reuse_setting": {-1: 4}},
)
def rms_norm_qwen3_impl(
    hidden_states: pypto.tensor(),
    weight: pypto.tensor(),
    output: pypto.tensor(),
    eps: float
):
    """
    RMSNorm Kernel
    
    Args:
        hidden_states: 输入tensor，shape=[..., hidden_size] 或 [..., num_heads, head_dim]
        weight: 归一化权重，shape=[hidden_size] 或 [head_dim]
        output: 输出tensor，shape与输入相同
        eps: 数值稳定性常数
    """
    rank = hidden_states.dim
    tile_shapes = [128 for _ in range(rank)]
    pypto.set_vec_tile_shapes(*tile_shapes)
    
    y = pypto.rms_norm(hidden_states, weight, eps)
    output[:] = y


def rms_norm_qwen3_wrapper(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    Wrapper 函数：接受 torch tensor，返回 torch tensor
    
    Args:
        hidden_states: 输入tensor (torch.Tensor)
        weight: 归一化权重 (torch.Tensor)
        eps: 数值稳定性常数
    
    Returns:
        归一化后的 tensor (torch.Tensor)
    """
    output = torch.empty_like(hidden_states)
    rms_norm_qwen3_impl(hidden_states, weight, output, eps)
    return output


if __name__ == "__main__":
    # 测试：input_layernorm 场景
    print("=" * 60)
    print("测试 RMSNorm 实现")
    print("=" * 60)
    
    device_id = get_device_id()
    if device_id is None:
        sys.exit(1)
    
    import torch_npu
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    
    # 测试场景1：input_layernorm
    x = torch.randn(1, 1, 2048, dtype=torch.float16, device=device)
    gamma = torch.ones(2048, dtype=torch.float16, device=device)
    
    output_pto = rms_norm_qwen3_wrapper(x, gamma, eps=1e-6)
    
    # 对比 Golden
    from rms_norm_golden import rms_norm_golden
    output_golden = rms_norm_golden(x.cpu(), gamma.cpu(), eps=1e-6).to(device)
    
    max_diff = (output_pto - output_golden).abs().max().item()
    print(f"[input_layernorm] shape={x.shape}, max_diff={max_diff:.6f}")
    assert max_diff < 1e-3, f"精度差异过大: {max_diff}"
    
    # 测试场景2：q_norm
    x = torch.randn(1, 1, 16, 128, dtype=torch.float16, device=device)
    gamma = torch.ones(128, dtype=torch.float16, device=device)
    
    output_pto = rms_norm_qwen3_wrapper(x, gamma, eps=1e-6)
    output_golden = rms_norm_golden(x.cpu(), gamma.cpu(), eps=1e-6).to(device)
    
    max_diff = (output_pto - output_golden).abs().max().item()
    print(f"[q_norm] shape={x.shape}, max_diff={max_diff:.6f}")
    assert max_diff < 1e-3, f"精度差异过大: {max_diff}"
    
    # 测试场景3：k_norm
    x = torch.randn(1, 1, 8, 128, dtype=torch.float16, device=device)
    gamma = torch.ones(128, dtype=torch.float16, device=device)
    
    output_pto = rms_norm_qwen3_wrapper(x, gamma, eps=1e-6)
    output_golden = rms_norm_golden(x.cpu(), gamma.cpu(), eps=1e-6).to(device)
    
    max_diff = (output_pto - output_golden).abs().max().item()
    print(f"[k_norm] shape={x.shape}, max_diff={max_diff:.6f}")
    assert max_diff < 1e-3, f"精度差异过大: {max_diff}"
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过")
    print("=" * 60)