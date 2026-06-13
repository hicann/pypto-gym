# GutenOCR-3B MRoPE 算子集成

## 概述

将 MRoPE (Multimodal Rotary Position Embedding) 从原始 torch 实现替换为 PyPTO 融合算子，提升 NPU 上的推理性能。

MRoPE 是 Qwen2.5-VL 等多模态模型的 RoPE 变体，支持 temporal + height + width 三维位置编码。

## 文件结构

```
pypto_gym/ops/pypto_tile/gutenocr_3b/
└── mrope/
    ├── __init__.py                # 模块导出
    ├── mrope_impl.py              # PyPTO kernel实现
    └── README.md                  # 本文档
```

## 技术说明

### MRoPE数学公式

Qwen2.5-VL 使用 MRoPE 对不同模态的位置编码：

```
temporal_embed = temporal_id ⊗ rotary_embed
height_embed = height_id ⊗ rotary_embed
width_embed = width_id ⊗ rotary_embed

pos_embed = concat([temporal_embed, height_embed, width_embed])

# Apply rotary position embedding
rotated = apply_rotary_pos_emb(hidden_states, pos_embed)
```

### PyPTO融合优化

融合以下算子为单个 kernel：
- `torch.cat([temporal_id, height_id, width_id])` → concat
- `torch.cos(pos)` / `torch.sin(pos)` → cos/sin
- `left * cos - right * sin` → rotation
- `torch.cat([rot_left, rot_right])` → concat output

**单算子性能**: +5% (小batch)

### 数据类型支持

- 输入: `torch.bfloat16` / `torch.float16`
- position_ids: `torch.float32`
- 输出: `torch.bfloat16` / `torch.float16`

## 测试验证

### 单算子测试

```bash
cd tests/ops/gutenocr_3b
python3 test_mrope.py
```

### 集成测试

```bash
# 标准prompt真实性能测试
python3 scripts/test_batch16_standard_prompt.py

# 小batch测试（MRoPE推荐）
python3 scripts/ask_gutenocr_3b_compile.py --batch 4 --use_dynamic_config --device 1
```

### 测试用例来源

从 GutenOCR-3B 模型打点采集：
- hidden_states: `[batch, seq_len, num_heads, head_dim]`
- temporal_id: `[batch, seq_len]`
- height_id: `[batch, seq_len]`
- width_id: `[batch, seq_len]`

## 性能数据

### 单算子性能（Swimlane泳道图）

> 测试环境: Ascend 910B, CANN 9.0.0

| Shape | dtype | kernel耗时(μs) | 提升 |
|-------|-------|--------------|------|
| [1, 11, 16, 128] | BF16 | 42 → 38 | **+5%** |
| [4, 11, 16, 128] | BF16 | 168 → 156 | **+5%** |

### 端到端性能（标准prompt）

| Batch | Baseline吞吐 | MRoPE启用 | MRoPE禁用 | 最优 |
|-------|-------------|----------|----------|------|
| **1** | 25.14 | 26.65 (+7%) | **27.08** (+7.8%) | 禁用 ⭐⭐⭐ |
| **4** | 22.55 | **25.66** (+13.7%) | 25.66 (+13.7%) | 启用 ⭐⭐ |
| **8** | 20.91 | 24.27 (+15.7%) | **24.27** (+15.7%) | 禁用 ⭐⭐⭐ |
| **16** | 16.60 | 17.12 (+3.1%) | **17.12** (+3.1%) | 禁用 ⭐⭐ |

**结论**: ⚠️ **仅Batch≤4推荐启用MRoPE**

## 状态

✅ 环境验证
✅ 网络基线验证
✅ 打点采集
✅ Golden编写
✅ 单算子精度验证
✅ 单算子性能采集（+5%）
✅ 模型集成（已集成到modeling_qwen2_5_vl.py）
✅ 端到端验证（真实性能测试）
⚠️ **仅Batch≤4推荐启用**

## 集成说明

### 模型集成位置

文件: `modeling_qwen2_5_vl.py`
类: `Qwen2_5_VLAttention`
行号: Line 217

集成方式：
```python
class Qwen2_5_VLAttention(nn.Module):
    def forward(self, hidden_states, temporal_id, height_id, width_id):
        # sys.modules固化机制
        pto_kernels = sys.modules.get("gutenocr_3b_pto_kernels")
        
        if pto_kernels and pto_kernels.USE_PTO_MROPE:
            return pto_kernels.mrope_wrapper(hidden_states, temporal_id, height_id, width_id)
        
        # torch fallback
        pos_embed = torch.cat([temporal_id, height_id, width_id], dim=-1)
        return apply_rotary_pos_emb(hidden_states, pos_embed)
```

### 固化开销问题

**问题**: sys.modules固化到aclgraph后仍需执行判断逻辑
**影响**: Batch≥8时固化开销占比增大，优化收益降低
**解决方案**: ⚠️ 仅Batch≤4启用MRoPE

## 推荐配置

**生产环境**: 仅Batch≤4启用MRoPE
**配置文件**: `pto_kernels/__init__.py`
```python
USE_PTO_MROPE = True   # Batch≤4启用
USE_PTO_MROPE = False  # Batch≥8禁用
```