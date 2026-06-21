# Qwen3-1.7B PyPTO 融合算子

Qwen3-1.7B 模型的 PyPTO 融合 kernel 算子库。当前实际集成的算子为 Q/K channel 的 RMSNorm + RoPE 部分融合 kernel。


## 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 目录结构

```
qwen3_1_7b/
├── __init__.py                          # 顶层入口: USE_PTO_ROPE, qwen3_qk_rope_q, qwen3_qk_rope_k
├── rms_norm_rope/                       # Q/K RMSNorm + RoPE 融合 kernel 实现
│   ├── __init__.py                      # 子包入口, 重导出 qwen3_qk_rope_q / qwen3_qk_rope_k
│   └── rrms_norm_rope_impl.py           # kernel 工厂与具体实现
└── rms_norm/                            # 独立 RMSNorm kernel (source 未提交, 仅 pycache)
```

## 算子列表

| 算子名称 | 融合范围 | 输入 shape | 输出 shape | 精度 | 对应 eager 代码 |
|---------|---------|-----------|-----------|------|---------------|
| `qwen3_qk_rope_q` | Q per-head RMSNorm + RoPE | `[S, 16, 128]` | `[S, 16, 128]` | BF16 | `rms_norm_per_head()` + `rotate_half()` + RoPE multiply |
| `qwen3_qk_rope_k` | K per-head RMSNorm + RoPE | `[S, 8, 128]` | `[S, 8, 128]` | BF16 | 同上, N_kv=8 |

两个 kernel 均由 `_make_qk_rope_kernel(N)` 工厂函数生成, 区别仅在于 head 数量 (Q: N_q=16, K: N_kv=8)。其余维度、精度、tile 大小均一致。

## 输入 / 输出详解

每个 kernel 接受 4 个输入 tensor 和 1 个输出 tensor:

| 参数 | shape | dtype | 含义 |
|------|-------|-------|------|
| `x` | `[S, N, 128]` | BF16 | q_proj 或 k_proj 的输出, S 为 sequence length, N=16(Q) 或 8(K) |
| `cos` | `[S, 128]` | BF16 | RoPE cosine 值, 由 position embedding 预计算 |
| `sin` | `[S, 128]` | BF16 | RoPE sine 值 |
| `w_norm` | `[128]` | BF16 | q_norm 或 k_norm 的权重向量 (`torch.nn.Parameter`) |
| `out` | `[S, N, 128]` | BF16 | 输出: 经过 RMSNorm + RoPE 处理后的 Q/K |

### 计算流程

kernel 内部沿 sequence 维度以 tile 方式遍历 (BS_TILE=8), 每个 tile 依次执行:

1. **RMSNorm**: `x_fp32 -> sq -> mean -> rsqrt(mean+eps) -> x * rsqrt -> x * w_norm`, 沿最后一维 (D=128) 归一化
2. **RoPE**: 将 normed 结果拆分为左右两半 `[BS_TILE, N, 64]`, 按 RoPE 公式旋转: `o1 = x_left*cos - x_right*sin`, `o2 = x_right*cos + x_left*sin`, 再 concat 回 `[BS_TILE, N, 128]`
3. 写回 BF16 输出

## 融合范围

```
q_proj / k_proj (PyTorch)  -->  [RMSNorm + RoPE] (PyPTO)  -->  Attention (PyTorch)
```

- 在 kernel 中融合: Q/K per-head RMSNorm + RoPE
- 留在 PyTorch 中: q_proj/k_proj/v_proj 线性投影, pre-attention RMSNorm, attention 计算

## 开关变量

| 变量 | 默认值 | 定义位置 | 含义 |
|------|--------|---------|------|
| `USE_PTO_ROPE` | `False` | `__init__.py` 第 24 行 | 控制模型是否启用 PyPTO 融合 RoPE kernel。设为 `True` 时, Attention 层用 `qwen3_qk_rope_q` / `qwen3_qk_rope_k` 替代 eager 的 per-head RMSNorm + RoPE |

## 关键实现细节

- **JIT 编译**: `_make_qk_rope_kernel()` 返回的是 `@pypto.frontend.jit` 装饰的 kernel, 首次调用时 JIT 编译
- **runtime 配置**: `stitch_function_max_num=128`, `device_sched_mode=1`
- **Tile 大小**: `BS_TILE=8` (sequence 维度), `D=128` (head dim), `HALF_D=64` (RoPE 半维度)
- **Epsilon**: `EPS=1e-6`, 与 Qwen3Config 中的 `rms_norm_eps` 一致
- **中间精度**: 计算以 FP32 进行, 输入/输出为 BF16
- **动态 shape**: sequence 维度标记为 `pypto.DYNAMIC`, 支持变长输入

## 对应测试

测试代码位于 `tests/model_ops/qwen3_1_7b/`:

| 文件 | 说明 |
|------|------|
| `test_rope.py` | RoPE kernel 精度测试, 覆盖 Q/K 两种 kernel, 8 个测试用例 (S=1,32,128,4,7) |
| `test_rope.json` | 测试用例配置, 定义 shape / dtype / tolerance |
| `rope_golden.py` | RoPE golden 参考实现 (PyTorch eager 等价代码) |
| `rms_norm_golden.py` | 独立 RMSNorm golden 参考实现 |
| `test_rms_norm.py` | 独立 RMSNorm 精度测试 |

### 运行测试

```bash
# 先设置环境
export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa
export TILE_FWK_DEVICE_ID=0

# 运行所有 RoPE 测试
python tests/model_ops/qwen3_1_7b/test_rope.py

# 只测 Q kernel
python tests/model_ops/qwen3_1_7b/test_rope.py --kernel q

# 只测 K kernel
python tests/model_ops/qwen3_1_7b/test_rope.py --kernel k

# 列出用例
python tests/model_ops/qwen3_1_7b/test_rope.py --list

# 通过 pytest 运行
pytest tests/ops/qwen3_1_7b -v
```

精度要求: rtol=1e-2, atol=1e-2 (BF16 精度)

## 模型集成

模型代码位于 `src/pypto_gym/transformers/qwen3_1_7b/`:

| 文件 | 说明 |
|------|------|
| `modeling_qwen3.py` | Qwen3 模型实现 (HuggingFace transformers 风格, 已适配 NPU) |
| `configuration_qwen3.py` | Qwen3Config, 含 Qwen3-1.7B 超参 (hidden_size=4096, num_layers=32, etc.) |

当 `USE_PTO_ROPE=True` 时, 在 Attention 层中将 eager 的 `q_norm` + `k_norm` + `apply_rotary_pos_emb` 替换为 `qwen3_qk_rope_q` / `qwen3_qk_rope_k` 融合 kernel。

## 端到端推理

推理脚本与基准测试位于 `modeling/transformers/qwen3_1_7b/`:

```bash
# 基线推理
python modeling/transformers/qwen3_1_7b/ask_Qwen3-1.7B.py --device 1

# 基准测试
bash modeling/transformers/qwen3_1_7b/bench_Qwen3-1.7B.sh
```

## 配置常量速查

| 常量 | 值 | 位置 |
|------|-----|------|
| D (head_dim) | 128 | `rrms_norm_rope_impl.py:35` |
| HALF_D | 64 | `rrms_norm_rope_impl.py:36` |
| EPS | 1e-6 | `rrms_norm_rope_impl.py:37` |
| BS_TILE | 8 | `rrms_norm_rope_impl.py:38` |
| USE_PTO_ROPE | False | `__init__.py:24` |
| N_q (Q heads) | 16 | `rrms_norm_rope_impl.py:166` |
| N_kv (KV heads) | 8 | `rrms_norm_rope_impl.py:167` |
