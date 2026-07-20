# 多组合策略对比与推荐

对于有多种实现方案的 B 类算子，以下是对比分析和推荐方案。

> **口径说明**：本文含少量未在 [torch-pypto-op-mapping.md](torch-pypto-op-mapping.md) 单列的融合/派生算子（如 `RMSNorm`、`SwiGLU`、`repeat_kv`），它们是映射表中基础算子（`layer_norm`、`silu`、`repeat` 等）的常见变体，仅在此做选型对比，不重复计入映射表编号。

## 1. mean

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: sum + div** | `pypto.sum(x, dim) / pypto.full(N)` | **PASS** (max_diff=0) | 1次reduce + 1次elementwise | ✅ **推荐** |
| B: cast + sum + div + cast | `cast(BF16→FP32) → sum → div → cast(FP32→BF16)` | 中间FP32 | 2次cast开销 | BF16场景 |

**推荐理由**：精度精确，步骤最少，性能最优。对于 BF16 输入，方案 A 本身就需要在 sum 前做 cast（因为 `pypto.sum` 的 FP32 硬约束），所以方案 B 本质上是方案 A 在 BF16 场景下的自然扩展。

## 2. RMSNorm

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: pypto.rms_norm** | `pypto.rms_norm(x, weight, eps)` | **PASS** (max_diff=2.38e-07) | 框架优化, 1步 | ✅ **推荐** |
| B: 手动组合 | `mul→sum→div→add→rsqrt→mul→mul` | **PASS** (max_diff=0) | 多步开销 | 自定义/调试 |

**推荐理由**：单步调用，代码简洁，性能最优（框架内部融合减少了内存搬运）。仅在需要自定义变体（如 Gated RMSNorm）或非标准维度时使用方案 B。

## 3. SiLU

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: x * sigmoid(x)** | `pypto.mul(x, pypto.sigmoid(x))` | **PASS** (max_diff=1.19e-07) | 2步 | ✅ **推荐** |
| B: x / (1 + exp(-x)) | `pypto.div(x, pypto.add(pypto.full(1.0), pypto.exp(pypto.neg(x))))` | 等价 | 4步, 更慢 | 不推荐 |

**推荐理由**：步骤数仅为方案 B 的一半，性能优势明显。`pypto.sigmoid` 是框架内置的硬件优化实现，比手动组合 `exp` 更可靠。

## 4. SwiGLU

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: silu(gate) * up** | `pypto.mul(pypto.mul(gate, pypto.sigmoid(gate)), up)` | **PASS** (max_diff=2.38e-07) | 3步 | ✅ **推荐** |
| B: gate * sigmoid(gate) * up | 数学等价于方案 A | 相同 | 相同 | 等价 |

**推荐理由**：方案 A 和 B 在 pypto 层面完全等价，推荐以 `silu(gate) * up` 的语义表达，代码可读性更好。

## 5. GELU

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: tanh 近似** | `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))` | **PASS** (max_diff=2.38e-07) | 多步 | ✅ **推荐**(通用) |
| B: erf 精确 | `x * 0.5 * (1 + erf(x / sqrt(2)))` | 精确 | erf可能慢 | 精度优先 |

**推荐理由**：`pypto.tanh` 有硬件加速，而 `pypto.erf` 的硬件加速状态不确定。在 BF16/FP16 推理场景下，tanh 近似的数学误差远小于量化误差，不影响最终精度。

## 6. Softmax

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: pypto.softmax** | `pypto.softmax(x, dim=-1)` | **PASS** (max_diff=0) | 内置, 1步 | ✅ **推荐** |
| B: 手动组合 | `amax→sub→exp→sum→div` | 等价 | 5步 | 自定义 |

**推荐理由**：精度完美，单步调用，性能最优。手动组合的 attention 实现已验证为 FAIL，说明在复杂场景下手动管理 softmax 的 tile shape 切换极其困难。

## 7. RoPE

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: 标准组合** | `cos/sin→mul→neg→concat→mul→add` | **PASS** (max_diff=0) | 标准 | ✅ **推荐** |
| B: fused RMSNorm+RoPE | `pypto.rms_norm` + RoPE 融合为单 kernel | 等同或更优 | 融合减少内存搬运 | 性能优化 |

**推荐理由**：精度完美，通用性最强。当 RMSNorm 和 RoPE 紧邻且模型架构允许时（如 Qwen3 1.7B），优先使用方案 B (fused kernel) 以获得更好的性能。

## 8. Attention

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| A: 标准组合 | `matmul→scale→softmax→matmul` | **FAIL** | 标准 | ❌ 不推荐 |
| **B: online softmax + 分块** | `amax→sub→exp→sum→div` + `is_loop_begin/end` 状态管理 | **PASS** (max_diff=0) | O(1)内存 | ✅ **推荐**(decode) |
| **C: flash attention kernel** | `sparse_flash_attention` 融合实现 | 未验证 | 最优 | ✅ **推荐**(生产) |

**推荐理由**：方案 A 已验证 FAIL，不推荐直接使用。方案 B 已验证 PASS（参考 gym gemma4 `gqa_decode_attn` 实现），是 decode 阶段的标准做法，数值稳定性好，内存效率高。方案 C 在已有成熟 production kernel 的场景下是最优选择。

## 9. MoE Routing

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: sigmoid + topk + gather + normalize** | `sigmoid(router) → topk → gather → sum → div` | **PASS** (max_diff=0) | 5步 | ✅ **推荐** |
| B: softmax + topk + gather | `softmax(router) → topk → gather` | 未验证 | 3步 | softmax gating 模型 |

**推荐理由**：精度完美，在 pypto-gym 覆盖的 MoE 模型（GLM、LLaDA2、MiniMax）中，sigmoid gating 是主流选择。sigmoid 是 elementwise 操作，比 softmax 更适合分布式计算。

## 10. Linear (with bias)

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| A: matmul + add | `pypto.add(pypto.matmul(x, w, dtype, b_trans=True), bias)` | **PASS** (各步骤) | 2次kernel | 简单场景 |
| **B: matmul with extend_params** | `pypto.matmul(x, w, dtype, extend_params={"bias_tensor": bias})` | 未验证 | 1次kernel, 融合bias | ✅ **推荐** |

**推荐理由**：融合实现减少一次 GM 读写，性能更优（matmul 输出直接在 Cube 单元内加上 bias）。代码更简洁，1 行完成。

## 11. Embedding

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: gather** | `pypto.gather(weight, dim=0, indices=input_ids)` | **PASS** (max_diff=0) | 查表 | ✅ **推荐** |
| B: index_select | `pypto.index_select(weight, dim=0, index=input_ids)` | **PASS** (max_diff=0) | 查表 | 1D index 场景 |

**推荐理由**：`gather` 支持多维 indices，在 Embedding 场景中可直接处理 [B, S] 的 input_ids 而无需 flatten/reshape，代码更简洁。

## 12. repeat_kv

| 方案 | 实现 | 精度 | 性能 | 推荐 |
|------|------|------|------|------|
| **A: unsqueeze + expand_clone + reshape** | `unsqueeze(kv, 2) → expand_clone([B,N_kv,G,S,D]) → reshape([B,N_q,S,D])` | **PASS** (expand_clone) | expand_clone 实际分配内存并复制 | ✅ **推荐**(默认) |
| B: 循环展开 | 在 attention 的 head loop 中，每 G 个 Q head 共用同一 KV head | 未验证 | 零额外内存 | 长序列 prefill |

**推荐理由**：代码简洁，decode 阶段（S=1）内存开销可忽略。在 prefill 阶段或长序列场景，KV 数据量大，方案 B 的零额外内存优势显著，可在 Stage 7 性能优化阶段切换。
