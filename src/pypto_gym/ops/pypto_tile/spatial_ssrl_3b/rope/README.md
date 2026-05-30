# Spatial-SSRL-3B RoPE 算子集成

## 概述

将 RoPE（Rotary Position Embedding）算子从原始 torch 实现替换为支持 aclgraph 的版本，提升 NPU 上的推理性能。

## 实现策略

**场景A**：使用 PyTorch 原生算子 + @allow_in_graph

- 原始实现仅使用基础 PyTorch 算子（cat, mul, add, view等）
- 这些算子在 torch.compile 中自动支持
- 无需复杂的 PyPTO kernel 实现
- @allow_in_graph 修饰后自动支持 aclgraph

## 文件结构

```
pypto_gym/ops/pypto_tile/spatial_ssrl_3b/
└── rope/
    ├── rope_impl.py              # PyPTO 实现（原生算子 + allow_in_graph）
    └── README.md                 # 本文档
```

## 测试验证

```bash
cd tests/ops/spatial_ssrl_3b

python3 test_rope.py
```

### 测试用例来源

从 Spatial-SSRL-3B 模型打点采集的真实 shape/dtype：

**Multimodal RoPE (Language Model)**:
- q: [1, 16, 31, 128] (batch, num_heads, seq_len, head_dim)
- k: [1, 2, 31, 128] (batch, num_kv_heads, seq_len, head_dim)
- cos/sin: [3, 1, 31, 128] (num_sections, batch, seq_len, head_dim)
- mrope_section: [16, 24, 24]

**Vision RoPE (Vision Encoder)**:
- q: [seq_len, num_heads, head_dim]
- k: [seq_len, num_heads, head_dim]
- cos/sin: [seq_len, head_dim // 2]

## 技术说明

### 包含两个算子

1. **apply_rotary_pos_emb_vision_impl**: Vision RoPE（2D）
   - 用于视觉编码器（Qwen2_5_VLVisionAttention）
   - 标准 2D 旋转位置编码

2. **apply_multimodal_rotary_pos_emb_impl**: Multimodal RoPE（3D）
   - 用于语言模型（Qwen2_5_VLAttention）
   - 多模态场景，支持 3D 位置编码（时间、高度、宽度）

### 核心算法

```python
# rotate_half: 旋转半维度
x1 = x[..., :head_dim // 2]
x2 = x[..., head_dim // 2 :]
return torch.cat([-x2, x1], dim=-1)

# RoPE 应用
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

## 性能收益

- 减少内核调用开销
- 支持 aclgraph 图模式编译
- 预计性能提升 15-25%

## 状态

✅ 环境验证
✅ 网络基线验证
✅ Golden 编写
✅ 单算子精度验证
✅ 模型集成
✅ 端到端验证