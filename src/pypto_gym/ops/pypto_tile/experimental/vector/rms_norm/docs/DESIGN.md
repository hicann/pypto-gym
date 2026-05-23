---
schema_version: "2.1"
op_name: "RMSNorm"
status: draft
last_updated: "2026-05-16"

compute_kind: "vector"
dtypes: ["fp32"]
dynamic_axes: ["B"]
precision: { rtol: 1e-3, atol: 1e-3 }
---

# RMSNorm 设计方案

## 1. 计算图与精度路由

### 1.1 API 调用序列

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 1 | 平方 | `pypto.mul(x, x)` 或 `x * x` | FP32 | FP32 | [tb, 64, th, tw] | 逐元素平方 |
| 2 | 沿 dim=1 求和 | `pypto.sum(sq, dim=1, keepdim=True)` | FP32 | FP32 | [tb, 1, th, tw] | 沿特征维度归约，keepdim 保留维度 |
| 3 | 除以 C | `pypto.div(s, C)` 或 `s / C` | FP32 | FP32 | [tb, 1, th, tw] | C=64 为编译期常量 |
| 4 | 加 epsilon | `pypto.add(mean_sq, eps)` 或 `mean_sq + eps` | FP32 | FP32 | [tb, 1, th, tw] | eps 为标量参数 |
| 5 | 开方 | `pypto.sqrt(mean_sq_eps)` | FP32 | FP32 | [tb, 1, th, tw] | 计算均方根 |
| 6 | 归一化 | `pypto.div(x, rms)` 或 `x / rms` | FP32 | FP32 | [tb, 64, th, tw] | 单轴广播 (dim=1: 1→64) |

**结论**：全部 6 步操作均有直接 PyPTO API 支持，全链路 FP32，无需 cast。

### 1.2 精度路由

```text
输入(FP32) → mul(FP32) → sum(FP32) → div(FP32) → add(FP32) → sqrt(FP32) → div(FP32) → 输出(FP32)
```

| 转换位置 | 转换方向 | 原因 |
|---------|---------|------|
| 无 | — | 全链路 FP32，`pypto.sum` 原生支持 FP32，无需类型转换 |

**关键优势**：本算子输入输出均为 FP32，所有原子操作（sum、mul、div、add、sqrt）均直接支持 FP32，避免了 BF16→FP32→BF16 的 cast 开销。

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| `pypto.rms_norm` 内置 API | 该 API 的 reduction 维度固定为 `dim=-1`（最后一维），而本算子需沿 `dim=1`（特征维度）归约，维度不匹配，无法直接使用 |
| 先 transpose 再用 dim=-1 | 引入额外的 transpose 操作，增加计算开销和内存搬运；手动分解实现更直接高效 |
| 使用 `mul(square, 1/C)` 替代 `div(sum, C)` | 生产参考（glm_v4_5）使用 `mul(square, mean_coff)` 模式，但本算子 div 的标量操作数 C=64 为编译期常量，div 性能可接受，且代码语义更清晰 |
| 使用 `reciprocal(rms) * x` 替代 `div(x, rms)` | 引入额外的 reciprocal 操作，增加计算步骤；直接 div 更简洁，且默认使用 HIGH_PRECISION 模式，精度有保障 |

---

## 2. 数据规格

### 2.1 Kernel 函数签名

```python
from dataclasses import dataclass

@dataclass
class RMSNormConfig:
    """RMSNorm 配置参数。"""
    eps: float = 1e-5
    num_features: int = 64


@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rms_norm_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 64, 256, 256], pypto.DT_FP32),    # 输入 [B, C, H, W]
    output: pypto.Tensor([pypto.DYNAMIC, 64, 256, 256], pypto.DT_FP32),  # 输出 [B, C, H, W]
    config: RMSNormConfig,                                              # eps 等配置
):
    ...
```

**Wrapper 函数**（供外部调用）：

```python
def RMSNorm_wrapper(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm wrapper: torch 输入 → torch 输出。"""
    B, C, H, W = x.shape
    config = RMSNormConfig(eps=eps, num_features=C)
    x_pto = pypto.from_torch(x, dynamic_axis=[0])
    out = torch.empty_like(x)
    rms_norm_kernel(x_pto, out, config)
    return out
```

### 2.2 动态轴分析

> 仅运行时才确定大小的轴标 `pypto.DYNAMIC`；编译期已知的轴写常量。

| 维度名 | 维度索引 | 是否动态 | 取值范围 / 常量 | 标注方式 | 说明 |
|--------|---------|---------|-----------------|---------|------|
| B | 0 | 是 | [1, 1024] | `pypto.DYNAMIC` | batch 维度，运行时变化 |
| C | 1 | 否 | 64 | 数值常量 | 特征维度，归约维度 |
| H | 2 | 否 | 256 | 数值常量 | 空间高度 |
| W | 3 | 否 | 256 | 数值常量 | 空间宽度 |

**动态轴选择理由**：
- B（batch 维度）天然可变，且**不是归约维度**（归约沿 dim=1），满足动态轴选择原则
- 归约维度 C=64 为编译期已知常量，TileShape 必须完整覆盖（tile_c=64），不可作为动态轴

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| `B` | `x.shape[0]`（动态轴） | SymbolicScalar | 不可用于 Python `if/range`，不可索引 list；仅用于 `pypto.loop` 和 `pypto.view` 的 offset |
| `b_idx` | `pypto.loop(B)` 迭代值 | SymbolicScalar | 仅用于 `pypto.view` / `pypto.assemble` 的 offset 参数 |
| `C` | 字面量 | int (64) | 编译期常量，常规使用 |
| `H` | 字面量 | int (256) | 编译期常量，常规使用 |
| `W` | 字面量 | int (256) | 编译期常量，常规使用 |
| `eps` | config.eps | float (1e-5) | 编译期常量标量，直接用于 `pypto.add` |
| `num_features` | config.num_features | int (64) | 编译期常量，用于 div 操作数 |

---

## 3. Tiling 策略

### 3.1 算子类型

**Vector** — 仅涉及逐元素运算（mul、div、add、sqrt）和归约运算（sum），无矩阵乘法。

### 3.2 Tiling 推导

**核心约束**：归约维度 dim=1（C=64）必须被 TileShape **完整覆盖**（tile_c = 64），以确保单次归约结果正确，无需跨 tile 累加。

- **同时驻留 UB 的 Tensor**（以单个 tile 为单位）：

| Tensor | 用途 | shape | dtype | 大小估算 |
|--------|------|-------|-------|---------|
| x_tile | 输入切片 | [1, 64, 1, 128] | FP32 | 32 KB |
| sq_tile | 平方中间结果 | [1, 64, 1, 128] | FP32 | 32 KB |
| out_tile | 输出切片 | [1, 64, 1, 128] | FP32 | 32 KB |
| s_tile | sum 结果 (keepdim) | [1, 1, 1, 128] | FP32 | 0.5 KB |
| rms_tile | sqrt 结果 | [1, 1, 1, 128] | FP32 | 0.5 KB |

- **推导步骤**：

  1. **归约维度约束**：dim=1 归约要求 tile_c = C = 64（完整覆盖）
  2. **尾轴对齐**：FP32 → 8 元素对齐（32B / 4B = 8），tile_w 必须为 8 的倍数
  3. **sum 64KB 约束**：`tile_b × 64 × tile_h × tile_w × 4 ≤ 65536`
     - tile_b=1, tile_h=1: `64 × tile_w × 4 ≤ 65536` → `tile_w ≤ 256`
     - 选取 tile_w = 128（保守，峰值 UB 占用更低）
  4. **UB 峰值估算**（编译器可能同时持有 x + sq + out）：
     - 保守峰值：3 × 32KB = 96KB（在 128KB–256KB UB 容量内 ✓）
     - 若编译器保留全部中间结果：~98KB（仍在安全范围 ✓）
  5. **展开约束**（单个 loop 迭代内）：
     - tiles = (1/1) × (64/64) × (256/1) × (256/128) = 512
     - 展开量 = 512 × (1 + 2) = 1536 ≤ 18000 ✓

- **最终 TileShape**：

```python
pypto.set_vec_tile_shapes(1, 64, 1, 128)
#                              ↑   ↑   ↑    ↑
#                              B   C   H    W
#                              |   |   |    └─ 尾轴 128 元素 (32B 对齐 ✓)
#                              |   |   └────── H 方向逐行处理
#                              |   └────────── 完整覆盖归约维度 (C=64)
#                              └────────────── B 由 pypto.loop 外部处理
```

### 3.3 替代方案

| 备选 tile | 否决理由 |
|-----------|---------|
| `[1, 64, 1, 256]` | tile_w=256 时 tile 大小恰好 64KB（sum 上限），且峰值 UB 达 192KB，接近 UB 容量上限，风险较高 |
| `[1, 64, 4, 64]` | tile 大小 = 1×64×4×64×4 = 65536 = 64KB，恰好踩 sum 上限；且 H 方向 4 行 tile 后的自动展开数仅略优，收益不大 |
| `[1, 64, 1, 64]` | tile_w=64 过小，每个 batch 需 1024 个 tile，展开量 3072，增加编译开销 |
| `[1, 32, 1, 128]` | tile_c=32 < C=64，无法单次完成 dim=1 归约，需跨 tile 累加中间结果，引入额外 loop 和累加器复杂度 |

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| B (dim=0) | DYNAMIC | 运行期 | `pypto.loop(B, name="LOOP_BATCH")` |
| C (dim=1) | 64 | 编译期已知 | 不需要 loop（TileShape 完整覆盖） |
| H (dim=2) | 256 | 编译期已知 | 编译器自动切分（256 个 tile） |
| W (dim=3) | 256 | 编译期已知 | 编译器自动切分（2 个 tile） |

**Loop 层次**：仅 1 层 `pypto.loop` 处理动态 B 轴，H/W 维度由编译器按 TileShape 自动切分。

### 4.2 数据流图

```
输入 x [B, 64, 256, 256] FP32
     │
     ├── pypto.loop(B) ──── 动态 batch 循环
     │     │
     │     ├── pypto.view(x, [1,64,256,256], [b_idx,0,0,0]) ──── 取第 b 个样本
     │     │     │
     │     │     │  ┌─ 编译器按 TileShape [1,64,1,128] 自动切分 ─┐
     │     │     │  │  (256×2 = 512 tiles, 每个 tile 独立计算)    │
     │     │     │  └──────────────────────────────────────────────┘
     │     │     │
     │     │     ├── Step 1: sq = x_slice * x_slice        [1,64,H,W] → [1,64,H,W]
     │     │     ├── Step 2: s = sum(sq, dim=1, keepdim=T)  [1,64,H,W] → [1, 1,H,W]
     │     │     ├── Step 3: mean_sq = s / 64               [1, 1,H,W] → [1, 1,H,W]
     │     │     ├── Step 4: mean_sq_eps = mean_sq + eps    [1, 1,H,W] → [1, 1,H,W]
     │     │     ├── Step 5: rms = sqrt(mean_sq_eps)        [1, 1,H,W] → [1, 1,H,W]
     │     │     └── Step 6: out = x_slice / rms            [1,64,H,W] / [1,1,H,W] → [1,64,H,W]
     │     │                                                   ↑ 单轴广播 dim=1
     │     │
     │     └── pypto.assemble(out, [b_idx,0,0,0], output) ──── 写回第 b 个结果
     │
     └── 输出 output [B, 64, 256, 256] FP32
```

### 4.3 完整伪代码

> 标注每个变量的类型（SymbolicScalar / int / Tensor），以及 view/assemble 的 offset。

```python
import pypto
import torch
from dataclasses import dataclass


# ─────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────
@dataclass
class RMSNormConfig:
    eps: float = 1e-5
    num_features: int = 64


# ─────────────────────────────────────────────────────
# 核心计算函数
# ─────────────────────────────────────────────────────
def rms_norm_core(x: pypto.Tensor, eps: float, num_features: int) -> pypto.Tensor:
    """RMSNorm 核心计算：纯 PyPTO API 实现。

    Args:
        x: 输入 tensor, shape [1, 64, 256, 256], FP32
        eps: 防止除零的小常数, float
        num_features: 特征维度大小, int (=64)

    Returns:
        归一化后的 tensor, shape 与输入一致
    """
    # Step 1: 逐元素平方
    sq = x * x                                               # [1, 64, 256, 256], FP32

    # Step 2: 沿特征维度 (dim=1) 求和, keepdim=True 保留维度
    s = pypto.sum(sq, dim=1, keepdim=True)                   # [1,  1, 256, 256], FP32

    # Step 3: 除以特征维度大小, 得到均方值
    mean_sq = s / num_features                                # [1,  1, 256, 256], FP32

    # Step 4: 加 epsilon 防止除零
    mean_sq_eps = mean_sq + eps                               # [1,  1, 256, 256], FP32

    # Step 5: 开方得到 RMS
    rms = pypto.sqrt(mean_sq_eps)                             # [1,  1, 256, 256], FP32

    # Step 6: 归一化: x / rms (rms 沿 dim=1 广播: 1→64)
    out = x / rms                                             # [1, 64, 256, 256], FP32

    return out


# ─────────────────────────────────────────────────────
# JIT Kernel
# ─────────────────────────────────────────────────────
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def rms_norm_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, 64, 256, 256], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 64, 256, 256], pypto.DT_FP32),
    config: RMSNormConfig,
):
    """RMSNorm JIT kernel, 支持动态 batch 维度。

    参数类型说明:
        x      : 输入 tensor, dim=0 为 pypto.DYNAMIC
        output : 输出 tensor, dim=0 为 pypto.DYNAMIC
        config : RMSNormConfig dataclass, 包含 eps 和 num_features
    """
    # ── 常量 (编译期已知) ──
    C = 64                # int, 特征维度 (归约维度)
    H = 256               # int, 空间高度
    W = 256               # int, 空间宽度
    eps = config.eps      # float, epsilon

    # ── 动态轴 ──
    B = x.shape[0]        # SymbolicScalar (动态轴 dim=0)

    # ── TileShape 配置 ──
    # [1, 64, 1, 128]: 完整覆盖归约维度 C=64, 尾轴 128 (32B 对齐)
    pypto.set_vec_tile_shapes(1, C, 1, 128)

    # ── 动态 batch 循环 ──
    for b_idx in pypto.loop(B, name="LOOP_BATCH", idx_name="b_idx"):
        # b_idx: SymbolicScalar

        # 取第 b_idx 个样本 (view, 只读)
        x_slice = pypto.view(x, [1, C, H, W], [b_idx, 0, 0, 0])
        # x_slice: [1, 64, 256, 256], FP32

        # 核心计算 (编译器按 TileShape 自动切分 H/W 维度)
        result = rms_norm_core(x_slice, eps, C)
        # result: [1, 64, 256, 256], FP32

        # 写回输出 (assemble, 只写)
        pypto.assemble(result, [b_idx, 0, 0, 0], output)


# ─────────────────────────────────────────────────────
# Wrapper 函数
# ─────────────────────────────────────────────────────
def RMSNorm_wrapper(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMSNorm 外部接口: torch.Tensor 输入 → torch.Tensor 输出。

    Args:
        x: 输入 tensor, shape [B, 64, 256, 256], dtype float32
        eps: 防止除零的小常数, 默认 1e-5

    Returns:
        RMS 归一化后的 tensor, shape 与输入一致
    """
    config = RMSNormConfig(eps=eps, num_features=x.shape[1])
    x_pto = pypto.from_torch(x, dynamic_axis=[0])
    out = torch.empty_like(x)
    rms_norm_kernel(x_pto, out, config)
    return out
```

### 4.4 跨迭代状态

| 状态名 | 初始化 | 更新方式 | submit_before_loop |
|--------|--------|---------|--------------------|
| 无 | — | — | — |

**说明**：每个 batch 样本的计算完全独立（无跨迭代数据依赖），`submit_before_loop` 不需要设为 True。编译器可将不同 batch 迭代并行分发到多核执行。

### 4.5 尾块处理

- **方案**：不需要显式 valid_shape 处理

**理由**：
- B 为动态轴，由 `pypto.loop(B)` 处理，每次迭代处理 1 个完整样本
- C=64、H=256、W=256 均为 TileShape 的整数倍：
  - C: 64 / 64 = 1（完整覆盖）
  - H: 256 / 1 = 256（整数倍）
  - W: 256 / 128 = 2（整数倍）
- 所有维度均无尾块不对齐问题

---

## 5. 约束检查与开放问题

### 5.1 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入已转 FP32 | ✓ | 全链路 FP32，输入即 FP32，无需 cast |
| 2 | matmul 两侧 dtype 一致 | N/A | 本算子无 matmul 操作 |
| 3 | TileShape 维度数 = 操作数维度数 | ✓ | TileShape [1,64,1,128] 为 4 维，操作 tensor 均为 4 维 |
| 4 | 尾轴满足对齐 | ✓ | tile_w=128，FP32 需 8 元素对齐 (128/8=16 ✓) |
| 5 | 同阶段 UB 占用 ≤ 容量 | ✓ | 峰值 ~96KB (3×32KB)，在 128KB-256KB UB 范围内 |
| 6 | 表达式展开 < 18000 | ✓ | 单迭代 512 tiles × 3 = 1536 < 18000 |
| 7 | 输出经 assemble 显式写回 | ✓ | `pypto.assemble(result, [b_idx,0,0,0], output)` |
| 8 | 无 view/assemble 同张量回环 | ✓ | view 读取 x，assemble 写入 output（不同 tensor）|
| 9 | 动态轴标 pypto.DYNAMIC | ✓ | dim=0 标注 `pypto.DYNAMIC`，dim=1/2/3 标注常量 |
| 10 | 动态 loop 提供 unroll_list | — | B 范围 [1, 1024]，编译器自动处理展开策略；如需优化可添加 `unroll_list` |
| 11 | 跨迭代状态 submit_before_loop | N/A | 无跨迭代状态，不需要 |
| 12 | 尾块用 valid_shape 处理 | N/A | 所有静态维度均为 TileShape 整数倍，无尾块 |
| 13 | 无 SymbolicScalar 禁止操作 | ✓ | `B` 和 `b_idx` 仅用于 `pypto.loop`、`pypto.view`/`assemble` 的 offset，未用于 `**`、list index、Python `if` |

### 5.2 API 可行性交叉验证

| 验证项 | 结果 | 详情 |
|--------|------|------|
| `pypto.sum(dim=1)` 支持性 | ✓ | 文档明确说明"支持任意单轴"，dim=1 合法 |
| `pypto.sum` keepdim=True TileShape 不变 | ✓ | 文档："keepdim=True 时保留被归约维度"，TileShape 无需重设 |
| `pypto.div` 广播支持 | ✓ | x [1,64,H,W] / rms [1,1,H,W]，单轴广播 (dim=1)，PyPTO 支持单轴广播 |
| `pypto.add(tensor, scalar)` | ✓ | `mean_sq + eps`，eps 为 Python float 标量，`pypto.add` 支持标量加法 |
| `pypto.mul(x, x)` 自乘 | ✓ | 参考实现 `rms_norm_core` 和 `rms_norm_denom` 均使用 `x * x` |
| `pypto.sqrt(FP32)` | ✓ | 文档：sqrt 支持 FP32 |

### 5.3 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| 1 | `pypto.sum(dim=1)` 在 4D tensor 上的实际运行表现 | 所有 dim=1 归约步骤 | 所有参考实现均使用 dim=-1；虽然文档声称支持任意轴，需在实现阶段首次验证 dim=1 的正确性。若不支持，回退方案：先 `pypto.transpose(x, [0,2,3,1])` 将 dim=1 移至末尾，再沿 dim=-1 归约 |
| 2 | TileShape [1,64,1,128] 在不同 NPU 型号上的 UB 容量适配性 | Tiling 策略 | 设计阶段按 128KB UB 估算，保守安全。若目标设备 UB 更大（如 256KB），可调大 tile_w 至 256 以减少 tile 数量，提升性能 |
| 3 | 编译器对 `x * x` 后复用 x 的优化策略 | 内存占用 (§4) | 步骤 1 (`sq = x * x`) 和步骤 6 (`out = x / rms`) 均需要 x。编译器可能选择保留 x 在 UB 或从 GM 重新加载。若编译器选择保留，峰值 UB 可能达 ~96KB；若选择重载，运行时稍慢但 UB 更省。两种策略均可行 |

---

## 6. 验证方案

### 6.1 测试配置

| 用例 | 输入 shape | dtype | 重点验证 | eps |
|------|----------|-------|---------|-----|
| 功能_P0 | [16, 64, 256, 256] | FP32 | 基本功能正确性，与 golden 输出对比 | 1e-5 |
| 性能_P0 | [16, 64, 256, 256] | FP32 | 端到端运行无报错，输出 shape 正确 | 1e-5 |
| 动态_B_min | [1, 64, 256, 256] | FP32 | B=1 边界：单 batch 是否正确 | 1e-5 |
| 动态_B_mid | [512, 64, 256, 256] | FP32 | B=512 中间值 | 1e-5 |
| 动态_B_max | [1024, 64, 256, 256] | FP32 | B=1024 上界：动态轴最大值是否正常 | 1e-5 |
| 数值_零输入 | [4, 64, 32, 32] | FP32 | 全零输入 → 全零输出 | 1e-5 |
| 数值_大值 | [4, 64, 32, 32] | FP32 | 大值输入 (scale=1e4) → 无 NaN/Inf | 1e-5 |
| 数值_小值 | [4, 64, 32, 32] | FP32 | 小值输入 (scale=1e-6) → 无 NaN/Inf | 1e-5 |
| 缩放不变性 | [4, 64, 32, 32] | FP32 | RMSNorm(k*x) ≈ RMSNorm(x) | 1e-5 |
| eps_边界 | [4, 64, 32, 32] | FP32 | 不同 eps (1e-8, 1e-2) 是否正常 | 变化 |

### 6.2 精度容忍度

| dtype | rtol | atol | 说明 |
|-------|------|------|------|
| FP32 | 1e-3 | 1e-3 | 按算子规格要求 (SPEC.md §6/§7) |

**验证方法**：
```python
# 对比 PyPTO 实现输出与 golden 参考输出
import torch
allclose = torch.allclose(pypto_output, golden_output, rtol=1e-3, atol=1e-3)
max_diff = (pypto_output - golden_output).abs().max().item()
```

### 6.3 参考实现与证据索引

| 信息 | 来源路径 |
|------|---------|
| PyPTO rms_norm_core 参考实现 | `examples/02_intermediate/basic_nn/layer_normalization/layer_norm.py` L158-165 |
| 生产级 rms_norm_denom (2D, dim=-1) | `models/deepseek_v4/hc_pre_impl.py` L22-28 |
| 生产级 rms_norm_bias (4D, dim=-1, 含 cast) | `models/glm_v4_5/glm_attention_fusion.py` L63-92 |
| pypto.sum API 文档 | `docs/api/operation/pypto-sum.md` |
| pypto.div API 文档 | `docs/api/operation/pypto-div.md` |
| pypto.set_vec_tile_shapes 文档 | `docs/api/config/pypto-set_vec_tile_shapes.md` |
| pypto.view API 文档 | `docs/api/operation/pypto-view.md` |
| pypto.assemble API 文档 | `docs/api/operation/pypto-assemble.md` |
| pypto.DYNAMIC 说明 | `docs/api/pypto-DYNAMIC.md` |
| pypto.loop 文档 | `docs/api/controlflow/pypto-loop.md` |
| pypto.from_torch 文档 | `docs/api/others/pypto-from_torch.md` |
| 循环与数据切分教程 | `docs/tutorials/development/loops.md` |
| Golden 参考实现 | `custom/level1/RMSNorm/RMSNorm_golden.py` |

---

## 设计状态总结

```text
设计状态：已收敛

迭代过程：
  第 1 轮：API 调用链 6 步，cast 0 处（全 FP32 无需类型转换）
  第 2 轮：Tiling Vector, tile = [1, 64, 1, 128]
  第 3 轮：Loop 1 层（B 动态轴），动态轴 [B(dim=0)]，跨迭代依赖 无
  第 4 轮：约束检查 13/13 通过（2 项 N/A）

开放问题：
  · pypto.sum(dim=1) 实际运行验证 — 若不支持需回退至 transpose+dim=-1 方案
  · TileShape 在不同 NPU 型号的 UB 适配性 — 可通过 config 参数化 tile_shape
  · 编译器对 x 复用的优化策略 — 两种策略（保留/重载）均可行
```

---
*生成时间: 2026-05-16*
*基于工件: SPEC.md + API_REPORT.md + RMSNorm_golden.py*
