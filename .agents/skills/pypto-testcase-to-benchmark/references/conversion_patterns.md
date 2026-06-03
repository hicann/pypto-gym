# PyPTO Testcase 转 Benchmark 详细参考

## 核心原则

**转换阶段只生成一个文件：`{N}_{OpName}.py`**。

REQUIRE.md、task_desc.py 由 benchmark 测试流程**自动生成**；SPEC.md、impl.py、golden.py、test.py、pypto_impl.py 等由 PyPTO 工作流生成，转换阶段不主动产生。

---

## 转换映射表

### 核心元素映射

| 源（已有 testcase） | 目标（benchmark case） | 关键注意点 |
|-------------------|----------------------|----------|
| `golden_*()` 函数 | `Model.forward()` | golden 逻辑 → PyTorch Module 前向方法 |
| `test_*()` 中构造的权重 | `Model.__init__` + `nn.Parameter` | 必须用 `nn.Parameter`，否则 `to(device)` 无效 |
| `test_*()` 中构造的输入 | `get_inputs()` 返回值 | 返回列表 `[x]`，数据类型需与 kernel 一致 |
| 无（test 中直接创建） | `get_init_inputs()` 返回值 | 提供 `Model.__init__` 所需的初始化参数 |
| 数学公式 | `FORMULA` 全局变量 | 文件顶层添加，case_loader 自动提取写入 REQUIRE.md |
| 动态维度 | `DYNAMIC_AXIS` 全局变量 | 文件顶层添加，case_loader 自动提取写入 REQUIRE.md |
| **in-place 输出 tensor** | **移除，改为 `return`** | KernelBench 要求 `forward()` 返回结果 |
| **控制参数（int/bool）** | **`Model.__init__` 参数** | 不能出现在 `get_inputs()` 中 |
| **NPU 特有 API** | **纯 PyTorch 近似** | `torch_npu.npu_xxx` 需替换为等价纯 PyTorch 实现 |

### 两种 testcase 结构

| 结构类型 | 特征 | 处理方式 |
|---------|------|----------|
| **标准结构** | 有独立 `golden_*()` 函数，清晰的输入/权重/输出分离 | 直接映射 golden → forward |
| **非标准结构** | golden 逻辑混在 `test_*()` 中，使用 `torch_npu` API，in-place 输出 | 从 test 中提取 golden 链，替换 NPU API，改为 return 形式 |

### 调用约定映射

```
KernelBench case:
  Model(*get_init_inputs())  →  Model.__init__(k=512, n=1024)
       ↓
  model.forward(*get_inputs())  →  Model.forward(x)
```

---

## 完整示例：FusedSwiGLU

### 已有 testcase 结构

```python
# test_fused_swiglu.py
def golden_fused_swiglu_fwd(x, w_g, w_fc, b_g, b_fc):
    gate = x.float() @ w_g.float() + b_g
    fc = x.float() @ w_fc.float() + b_fc
    gate_silu = gate * torch.sigmoid(gate)
    y = (gate_silu * fc).to(torch.bfloat16)
    return y

@pypto.frontend.jit(...)
def fused_swiglu_fwd_kernel(x, w_g, w_fc, b_g, b_fc, y):
    # ... kernel 实现
    pass

def test_fwd(m, k, n, device_id):
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    w_g = torch.randn(k, n, dtype=torch.bfloat16, device=device)
    # ... 其他权重
    y_golden = golden_fused_swiglu_fwd(x, w_g, w_fc, b_g, b_fc)
    fused_swiglu_fwd_kernel(x, w_g, w_fc, b_g, b_fc, y_out)
    assert_allclose(y_out, y_golden, rtol=0.0078125, atol=0.0001)
```

### 转换后产物（唯一产物）

#### KernelBench 用例 (101_FusedSwiGLU.py)

```python
import torch
import torch.nn as nn

FORMULA = "out[m, n] = SiLU(x[m, k] @ W_g[k, n] + b_g) * (x[m, k] @ W_fc[k, n] + b_fc)"
DYNAMIC_AXIS = ["M"]

class Model(nn.Module):
    def __init__(self, k: int = 512, n: int = 1024):
        super().__init__()
        self.w_g = nn.Parameter(torch.randn(k, n, dtype=torch.bfloat16) / math.sqrt(k))
        self.w_fc = nn.Parameter(torch.randn(k, n, dtype=torch.bfloat16) / math.sqrt(k))
        self.b_g = nn.Parameter(torch.randn(1, n, dtype=torch.bfloat16) / math.sqrt(n))
        self.b_fc = nn.Parameter(torch.randn(1, n, dtype=torch.bfloat16) / math.sqrt(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = x @ self.w_g + self.b_g
        fc = x @ self.w_fc + self.b_fc
        return gate * torch.sigmoid(gate) * fc

def get_inputs():
    return [torch.randn(220000, 512, dtype=torch.bfloat16)]

def get_init_inputs():
    return [512, 1024]
```

---

## FORMULA 与 DYNAMIC_AXIS 详解

### FORMULA — 数学公式

**用途**：`case_loader.py` 自动提取并写入 `REQUIRE.md` 的 `### 1.3 数学公式` 小节

**格式要求**：
- 简洁的数学表达式，用下标索引描述运算
- 必须是可被 `ast.literal_eval` 解析的字符串常量
- 不能是运行时计算结果

**示例**：
```python
FORMULA = "out[m, n] = SiLU(x[m, k] @ W_g[k, n] + b_g[0, n]) * (x[m, k] @ W_fc[k, n] + b_fc[0, n])"
```

### DYNAMIC_AXIS — 动态轴

**用途**：`case_loader.py` 自动提取并写入 `REQUIRE.md` front matter 的 `dynamic_axis` 字段

**格式要求**：
- 字符串列表，每个元素是动态维度的名称（大写字母）
- 必须是可被 `ast.literal_eval` 解析的列表常量
- 区分动态轴（运行时变化，如 batch 维度 M）和固定轴（如 K、N）

**示例**：
```python
DYNAMIC_AXIS = ["M"]
```

**兼容规则**：
- 老 KernelBench case 没有这两个变量时，不会输出对应内容
- 新 case 有其中一个变量时，只输出对应部分
- 变量必须是可被 `ast.literal_eval` 解析的常量表达式；不要写成运行时计算结果

---

## 特殊场景转换示例

### 场景 1: In-place 实现转 return 形式

原始 PyPTO 算子通常是 in-place 的（结果写入传入的输出 tensor）：

```python
# 原始 in-place 风格
def gate(hidden_states, gate_weight, router_logits_out):
    # 计算结果直接写入 router_logits_out
    router_logits_out[:] = torch.matmul(hidden_states, gate_weight.t())

# test 中调用
gate(hidden_states, weight, router_logits_out)
assert_allclose(router_logits_out, golden_result)
```

转换为 KernelBench 格式时，移除输出参数，改为 return：

```python
class Model(nn.Module):
    def __init__(self, num_router_experts: int = 160, hidden_size: int = 5120):
        super().__init__()
        self.gate_weight = nn.Parameter(
            torch.randn(num_router_experts, hidden_size, dtype=torch.float32) / math.sqrt(hidden_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden_states, self.gate_weight.t())

def get_inputs():
    return [torch.randn(64, 5120, dtype=torch.float32) / math.sqrt(5120)]

def get_init_inputs():
    return [160, 5120]
```

### 场景 2: 控制参数处理

原始实现中可能有 `int`/`bool` 控制参数：

```python
# 原始实现
def select_experts(router_logits, top_k, renormalize, topk_group, ...,
                   e_score_correction_bias, topk_weights, topk_ids):
    ...
```

控制参数移到 `__init__`，tensor 参数保留在 `forward`：

```python
class Model(nn.Module):
    def __init__(self, num_router_experts=160, top_k=8, renormalize=True):
        super().__init__()
        self.top_k = top_k
        self.renormalize = renormalize
        self.e_score_correction_bias = nn.Parameter(
            torch.randn(num_router_experts, dtype=torch.bfloat16) / math.sqrt(num_router_experts)
        )

    def forward(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = router_logits.sigmoid()
        # 使用 self.top_k, self.renormalize
        return topk_weights, topk_ids

def get_inputs():
    return [torch.randn(32, 160, dtype=torch.float32) / math.sqrt(160)]

def get_init_inputs():
    return [160, 8, True]
```

### 场景 3: NPU API 纯 PyTorch 近似

原始 testcase 使用 NPU 特有 API：

```python
# 原始 NPU 实现
x_quant = torch_npu.npu_quantize(x_g, x_scale, x_offset, torch.qint8, -1, False)
mm = torch_npu.npu_quant_matmul(x_quant, weight, deq_scale, bias=quant_bias, output_dtype=torch.bfloat16)
```

替换为纯 PyTorch 近似：

```python
# KernelBench golden 实现
x_scale = self.input_scale_reciprocal.float().unsqueeze(0)
x_offset = self.input_offset.float().unsqueeze(0)
x_quant = (x_g.float() * x_scale + x_offset).round().clamp(-128, 127).to(torch.int8)

mm = x_quant.float() @ self.weight.float()  # ⚠️ 用 float32 代替 int32，aclnn matmul 不支持 int32 输入
mm = mm + self.quant_bias.float().unsqueeze(0)  # ⚠️ 先加 bias
mm = mm * self.deq_scale.float().unsqueeze(0)   # ⚠️ 再乘 deq_scale（顺序不可交换，与 mm*deq_scale+bias 不等价）
mm_golden = mm.to(torch.bfloat16)
```

### 场景 4: 多输出算子

原始实现是 in-place 多输出：

```python
def attention_pre_quant(..., query, key, value, residual_res):
    # 结果写入 query, key, value, residual_res
    ...
```

转换为 return tuple：

```python
class Model(nn.Module):
    def __init__(self, hidden_size=5120, total_head_size=1792, ...):
        super().__init__()
        ...

    def forward(self, hidden_states, residual, cos, sin
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...
        return query, key, value, residual_out

def get_inputs():
    return [
        torch.randn(8, 5120, dtype=torch.bfloat16) / math.sqrt(5120),
        torch.randn(8, 5120, dtype=torch.bfloat16) / math.sqrt(5120),
        torch.randn(8, 1, 32, dtype=torch.bfloat16) / math.sqrt(32),
        torch.randn(8, 1, 32, dtype=torch.bfloat16) / math.sqrt(32),
    ]
```

---

## 常见陷阱详解

### 陷阱 1: 权重未注册为 Parameter

**现象**:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and npu:0!
```

**错误代码**:
```python
class Model(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.w_g = torch.randn(k, n)  # ❌ 普通张量
```

**修复**:
```python
class Model(nn.Module):
    def __init__(self, k, n):
        super().__init__()
        self.w_g = nn.Parameter(torch.randn(k, n))  # ✅ Parameter
```

### 陷阱 2: init/forward 参数混淆

**get_init_inputs vs get_inputs**:
- `get_init_inputs()` → `Model.__init__()` → 创建权重
- `get_inputs()` → `Model.forward()` → 计算输出

**正确映射**:
```python
def get_init_inputs():
    return [512, 1024]  # → Model.__init__(512, 1024)

def get_inputs():
    return [torch.randn(220000, 512)]  # → Model.forward(x)
```

### 陷阱 3: 输入未数值缩放导致 BF16 溢出

**现象**: benchmark 精度验证大面积失败（>50% 元素不匹配）

**原因**: 输入 tensor 数值过大，BF16 表示范围有限导致溢出

**修复**: 对输入进行数值缩放
```python
def get_inputs():
    M = 220000
    return [torch.randn(M, 512, dtype=torch.bfloat16) / math.sqrt(M)]
```

### 陷阱 4: 转换阶段越权生成产物

**现象**: 手动创建了 impl.py、golden.py、test.py、pypto_impl.py 等文件

**原因**: 误以为转换阶段需要生成这些文件

**修复**:
- **不要生成** impl.py、golden.py、test.py、pypto_impl.py
- 这些由 benchmark 测试流程（case_loader / pypto-op-orchestrator）**自动生成**
- 转换阶段唯一产物是 `{N}_{OpName}.py`

### 陷阱 5: 未处理 in-place 输出

**现象**: `forward()` 参数列表包含输出 tensor，case_loader 解析异常或验证失败

**原因**: 原始实现是 in-place 风格，但 KernelBench 要求 return 风格

**修复**:
```python
# 错误：保留了 in-place 输出参数
def forward(self, x, out):  # ❌
    out[:] = x @ self.w

# 正确：移除输出参数，return 结果
def forward(self, x) -> torch.Tensor:  # ✅
    return x @ self.w
```

### 陷阱 6: NPU API 未替换

**现象**: `torch_npu.npu_xxx` 在纯 CPU 环境或 verifier 阶段报错

**原因**: KernelBench golden 必须纯 PyTorch，不能依赖 NPU

**修复**: 用纯 PyTorch 近似等价实现

```python
# 错误：保留 NPU API
x_quant = torch_npu.npu_quantize(x, scale, offset, ...)  # ❌

# 正确：纯 PyTorch 近似
x_quant = (x * scale + offset).round().clamp(-128, 127).to(torch.int8)  # ✅
```

### 陷阱 7: 控制参数混在 get_inputs 中

**现象**: `get_inputs()` 返回 int/bool，case_loader 解析失败

**原因**: KernelBench 格式要求 `get_inputs()` 只返回 tensor

**修复**: 控制参数移到 `__init__`

```python
# 错误：get_inputs 返回非 tensor
def get_inputs():
    return [torch.randn(32, 160), 8, True]  # ❌

# 正确：控制参数移到 get_init_inputs
def get_inputs():
    return [torch.randn(32, 160)]  # ✅

def get_init_inputs():
    return [160, 8, True]  # ✅ → Model.__init__(160, 8, True)
```

---

## 验证检查清单

Phase 1 完成后，检查以下事项：

- [ ] `{N}_{OpName}.py` 已创建在正确的路径（以 benchmark/docs/add-new-case.md 为准）
- [ ] `Model` 类中权重使用 `nn.Parameter()`
- [ ] `get_inputs()` 返回列表 `[x]`
- [ ] `get_init_inputs()` 返回 `Model.__init__` 所需参数
- [ ] 输入 tensor 已进行数值缩放（如 `/ math.sqrt(M)`）
- [ ] `FORMULA` 已添加（如 benchmark/docs/add-new-case.md 要求）
- [ ] `DYNAMIC_AXIS` 已添加（如 benchmark/docs/add-new-case.md 要求）
- [ ] **原始实现如果是 in-place，已改为 return 形式**
- [ ] **多输出算子返回 `tuple[torch.Tensor, ...]`**
- [ ] **NPU 特有 API 已替换为纯 PyTorch 近似**
- [ ] **控制参数（int/bool）已移到 `__init__`，不在 `get_inputs` 中**
- [ ] 环境变量 `PTO_TILE_LIB_CODE_PATH` 和 `TILE_FWK_DEVICE_ID` 已设置

**注意**：不要检查 impl.py、golden.py、test.py、pypto_impl.py、SPEC.md、REQUIRE.md、task_desc.py — 这些由 benchmark + PyPTO 流程自动生成。
