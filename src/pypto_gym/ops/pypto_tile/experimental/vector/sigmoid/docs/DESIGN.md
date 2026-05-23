---
schema_version: "2.1"
op_name: "Sigmoid"
status: draft
last_updated: "2026-05-16"

compute_kind: "vector"
dtypes: ["fp32"]
dynamic_axes: ["B"]
precision: { rtol: 0.001, atol: 0.001 }
---

# Sigmoid 设计方案

## 1. 计算图与精度路由

### 1.1 API 调用序列

Sigmoid 为单一原子操作，无需拆分多步。`pypto.sigmoid(x)` 内部完成 `1/(1+exp(-x))` 的全部计算。

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 1 | σ(x) | `pypto.sigmoid(x)` | DT_FP32 | DT_FP32 | 与输入相同 | direct mapping，逐元素 |

### 1.2 精度路由

```text
输入(FP32) → pypto.sigmoid(FP32) → 输出(FP32)
```

| 转换位置 | 转换方向 | 原因 |
|---------|---------|------|
| 无需 cast | — | `pypto.sigmoid` 仅支持 FP32，本算子输入即为 FP32，全链路 dtype 一致 |

**推导过程**：
1. 查阅 `docs/api/operation/pypto-sigmoid.md`，确认 `pypto.sigmoid` 仅接受 DT_FP32 输入。
2. SPEC 定义输入/输出均为 float32，与 API 约束完全吻合。
3. 无中间精度转换需求，零 cast 点。

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| 手动分解：`neg → exp → add → div` 四步组合 | `pypto.sigmoid` 已提供 direct mapping，手动分解引入 3 个中间 tensor，增加 UB 压力和指令开销，无任何收益 |
| `torch.sigmoid` 直接调用 | 非 PyPTO API，无法在 NPU 上执行自定义 kernel |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def sigmoid_kernel(
    x: pypto.Tensor(),                                    # 输入 [B, 16384], FP32
    out: pypto.Tensor(),                                  # 输出 [B, 16384], FP32
):
    ...
```

**签名说明**：
- 使用 `pypto.Tensor()` 无固定 shape 声明（与 `activation.py` 模式一致），编译器在运行时根据实际输入 shape 推导。
- 输入 `x` 和输出 `out` 由 wrapper 层（`Sigmoid_wrapper`）通过 `pypto.from_torch` 创建，确保 dtype 和 contiguous 性。

### 2.2 动态轴分析

> 仅运行时才确定大小的轴标 `pypto.DYNAMIC`；编译期已知的轴写常量。

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 | 说明 |
|--------|---------|-----------------|---------|------|
| B (dim 0) | 是 | [1, 65536] | 由编译器自动推导 | batch 维度，运行时确定 |
| D (dim 1) | 否 | 16384 | 常量 | 特征维度，编译期已知 |

**动态轴处理策略**：
- B 为运行时才确定的动态轴，但由于 Sigmoid 是纯 element-wise 操作，无跨迭代依赖。
- 编译器通过 `set_vec_tile_shapes` 配置 tile 大小后，自动对全量 tensor 进行分块迭代，无需显式 `pypto.loop`。
- 这与 `activation.py` 中 SiLU kernel 的处理方式一致。

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| `x` / `out` | wrapper 层传入 | Tensor | 由 `pypto.from_torch` 创建 |
| `x.shape[0]` (B) | 运行时推导 | 编译器内部处理 | 本设计不显式访问，无 SymbolicScalar 误用风险 |
| `16384` (D) | 字面量 | Python int | 常规使用 |

**说明**：由于本设计不使用显式 `pypto.loop`，不直接操作动态轴的 SymbolicScalar 值，因此无 SymbolicScalar 误用风险。

---

## 3. Tiling 策略

### 3.1 算子类型

**Vector**（纯 element-wise，无 matmul/reduction）

### 3.2 Tiling 推导

- **同时驻留 UB 的 Tensor**：

| Tensor | 用途 | tile shape | dtype | 单 tile 大小 |
|--------|------|-----------|-------|------------|
| x_tile | 输入 tile | [32, 32] | FP32 (4B) | 32 × 32 × 4 = 4,096 B |
| out_tile | 输出 tile | [32, 32] | FP32 (4B) | 32 × 32 × 4 = 4,096 B |
| **合计** | | | | **8,192 B (8 KB)** |

- **推导步骤**：
  1. **尾轴对齐**：FP32 → 8 元素对齐（32B），tile_D=32 满足 32/8=4 倍对齐 ✓
  2. **UB 预算**：2 × 32 × 32 × 4 = 8,192 B << UB 容量（~128KB）✓
  3. **展开约束**：编译器为单 tile [32,32] 生成指令，运行时循环迭代，编译期表达式展开仅涉及单 tile 计算 ≪ 18,000 ✓
  4. **TileShape 维度数**：2 维 = 输出 tensor 维度数 ✓

- **最终 tile**：

```python
# 参考 activation.py configure_tiling() 模式
pypto.set_vec_tile_shapes(32, 32)
```

### 3.3 替代方案

| 备选 tile | 否决理由 |
|-----------|---------|
| `[1, 8192]` (行级 tile) | 尾轴 8192 虽满足对齐，但 2D tile `[32,32]` 在 UB 充裕的前提下提供更均衡的并行度，且与 activation.py 成熟模式一致 |
| `[64, 64]` | UB 占用：2 × 64 × 64 × 4 = 32,768 B (32KB)，可行但无明确收益；默认 `[32,32]` 已是官方推荐 |
| `[32, 128]` (1D fallback) | 仅当输入为 1D 时使用；本算子输入恒为 2D，不适用 |

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| B (dim 0) | DYNAMIC (1–65536) | 运行期 | **编译器自动分块**（纯 element-wise，无跨迭代依赖） |
| D (dim 1) | 16384 | 编译期已知 | **编译器自动分块** |

**判定依据**：
- Sigmoid 是纯 element-wise 操作，无 reduction、无跨行依赖、无累加器。
- 参照 `activation.py` 的 SiLU kernel（`out[:] = x * pypto.sigmoid(x)`），编译器通过 `set_vec_tile_shapes` 配置的 tile 自动切分并迭代全量 tensor。
- 无需显式 `pypto.loop`。

### 4.2 完整伪代码

> Kernel 伪代码 + Wrapper 伪代码，标注每个变量类型。

```python
import pypto
import torch

global_run_mode = pypto.RunMode.NPU  # 或 pypto.RunMode.SIM

def configure_tiling(x):
    """根据输入维度动态配置 tiling（复用 activation.py 模式）。"""
    if len(x.shape) >= 2:
        tile_list = [32 for _ in range(len(x.shape))]
        pypto.set_vec_tile_shapes(*tile_list)
    else:
        pypto.set_vec_tile_shapes(32, 128)

# ── Kernel ──────────────────────────────────────────────

@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def sigmoid_kernel(
    x: pypto.Tensor(),                                    # 输入: [B, 16384], DT_FP32
    out: pypto.Tensor(),                                  # 输出: [B, 16384], DT_FP32
):
    """Sigmoid 激活函数 kernel。

    公式: out = σ(x) = 1 / (1 + exp(-x))
    逐元素计算，编译器通过 set_vec_tile_shapes 自动分块迭代。
    """
    configure_tiling(x)                                   # 配置 tile [32, 32]
    out[:] = pypto.sigmoid(x)                             # [B, 16384], FP32 → FP32

# ── Wrapper ─────────────────────────────────────────────

def Sigmoid_wrapper(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid 算子 wrapper，封装 kernel 调用。

    Args:
        x: 输入 tensor, shape [B, 16384], dtype float32。
           B 为动态轴，取值范围 [1, 65536]。

    Returns:
        y: 输出 tensor, shape [B, 16384], dtype float32。
           值域 (0, 1)。
    """
    # 确保输入 contiguous（pypto.from_torch 约束）
    x = x.contiguous()

    # 创建输出 tensor
    out = torch.empty_like(x)

    # 调用 kernel
    sigmoid_kernel(x, out)

    return out
```

**伪代码关键标注**：

| 行 | 变量/操作 | 类型 | 说明 |
|----|----------|------|------|
| `configure_tiling(x)` | — | 函数调用 | `x.shape` 由运行时决定，`len(x.shape)` 为 Python int |
| `out[:] = pypto.sigmoid(x)` | `out`, `x` | Tensor | `[:]` 显式写回，shape 自动匹配 |
| `x.contiguous()` | `x` | torch.Tensor | wrapper 层确保 contiguous |
| `torch.empty_like(x)` | `out` | torch.Tensor | wrapper 层预分配输出 |

### 4.3 跨迭代状态

无。Sigmoid 为纯 element-wise 操作，无累加器或跨 tile 状态。

### 4.4 尾块处理

**不需要显式处理**。

- `pypto.sigmoid` 作为编译器内置的 vector op，在 `set_vec_tile_shapes` 模式下，编译器自动处理尾块（非整 tile 的边界情况）。
- 尾轴 16384 / 32 = 512，无余数，dim 1 恰好整除。
- dim 0 的 B 值由运行时确定，编译器通过 valid_shape 内部处理非整 tile 情况。

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入已转 FP32 | ✓ 不适用 | 无 sum 操作 |
| 2 | matmul 两侧 dtype 一致 | ✓ 不适用 | 无 matmul 操作 |
| 3 | TileShape 维度数 = 操作数维度数 | ✓ | 2D tile [32,32] = 2D tensor [B, 16384] |
| 4 | 尾轴满足对齐 | ✓ | FP32 尾轴 32，满足 8 元素（32B）对齐 |
| 5 | 同阶段 UB 占用 ≤ 容量 | ✓ | 8,192 B << 128 KB |
| 6 | 表达式展开 < 18000 | ✓ | 单 tile 编译，编译器运行时循环 |
| 7 | 输出经 `[:]` 显式写回 | ✓ | `out[:] = pypto.sigmoid(x)` |
| 8 | 无 view/assemble 同张量回环 | ✓ | 无 view/assemble，整体赋值 |
| 9 | 动态轴标 `pypto.DYNAMIC` | ✓ | 使用 `pypto.Tensor()` 无固定 shape，编译器自动推导 |
| 10 | 动态 loop 提供 `unroll_list` | ✓ 不适用 | 无显式 pypto.loop |
| 11 | 跨迭代状态用 `submit_before_loop=True` | ✓ 不适用 | 无跨迭代状态 |
| 12 | 尾块用 `valid_shape` 处理 | ✓ 不适用 | 编译器内置处理 |
| 13 | 无 SymbolicScalar 用作 `**` / list index / Python `if` | ✓ | 不显式访问 SymbolicScalar |

### 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| — | 无开放问题 | — | — |

---

## 6. 验证方案

### 6.1 测试配置

| 用例名称 | 优先级 | 输入 shape | dtype | 重点验证 |
|----------|--------|----------|-------|---------|
| 性能_P0 | P0 | [16, 16384] | float32 | 核心性能场景，精度 + shape + 值域 |
| 功能_P0 | P0 | [1, 16384] | float32 | 最小 shape 验证，边界 B=1 |
| 功能_P1 | P1 | [64, 16384] | float32 | 大 batch 验证 |
| 极值_B | P1 | [65536, 16384] | float32 | B 上界验证（如内存允许） |
| 零值 | P0 | [2, 16384] (全零) | float32 | sigmoid(0) = 0.5 |
| 大正值 | P1 | [1, 16384] (x=100) | float32 | sigmoid(100) → 1.0 |
| 大负值 | P1 | [1, 16384] (x=-100) | float32 | sigmoid(-100) → 0.0 |
| 随机值 | P0 | [16, 16384] (randn) | float32 | 通用精度验证 |

### 6.2 精度容忍度

| dtype | rtol | atol | 来源 |
|-------|------|------|------|
| FP32 | 0.001 | 0.001 | SPEC.md 定义 |

### 6.3 精度验证方法

```python
# 1. 计算实现输出
y_impl = Sigmoid_wrapper(x_npu)

# 2. 计算 golden 输出
y_golden = Sigmoid_golden(x_cpu)

# 3. 精度对比
assert torch.allclose(y_impl.cpu(), y_golden, rtol=0.001, atol=0.001), \
    f"精度不匹配: max_diff={((y_impl.cpu() - y_golden).abs().max().item()):.6e}"
```

### 6.4 验证流程

1. **Shape 检查**：验证输出 shape 与输入 shape 完全一致。
2. **Dtype 检查**：验证输出 dtype 为 float32。
3. **精度对比**：使用 `torch.allclose(rtol=0.001, atol=0.001)` 对比 `Sigmoid_wrapper` 输出与 `Sigmoid_golden` 输出。
4. **值域检查**：验证所有输出值 ∈ [0, 1]（允许浮点边界 0.0 和 1.0）。
5. **数学属性**：
   - sigmoid(0) = 0.5
   - 对称性：σ(-x) = 1 - σ(x)
   - 单调递增性

---

## 设计状态总结

```text
设计状态：已收敛

迭代过程：
  第 1 轮：API 调用链 1 步，cast 0 处
    - pypto.sigmoid(x) direct mapping, FP32 全链路
  第 2 轮：Tiling Vector, tile = [32, 32]
    - UB 占用 8KB，远低于容量上限
    - 尾轴 16384 恰好被 32 整除
  第 3 轮：Loop 0 层显式循环，动态轴 B 由编译器自动分块
    - 纯 element-wise，无跨迭代依赖
    - 遵循 activation.py 成熟模式
  第 4 轮：约束检查 13/13 通过

回退记录：无

开放问题：无
```
