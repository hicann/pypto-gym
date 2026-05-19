---
name: pypto-testcase-to-benchmark
description: >
  将 PyPTO 已有 testcase 转换为 benchmark 框架可用用例的 Skill。
  适用于用户已有算子 testcase（含 golden 函数、kernel 实现、测试脚本），需要将其集成到
  pypto-gym/benchmark 测试框架中以便统一验证 correctness + performance 的场景。
  触发场景：(1) "把这个 testcase 转成 benchmark case" (2) "已有 kernel 怎么接入 benchmark"
  (3) "生成 benchmark 用例" (4) 任何提到 testcase + benchmark 转换的需求。
  也适用于批量验证已有 golden case 与原始测试的一致性：生成对比验证脚本、输出结构化分析报告、
  定位并修复 golden 中的差异（量化路径、RoPE 约定、RMSNorm epsilon、结构简化等）。
  触发场景：(5) "验证这些 golden case 和原 test 是否等价" (6) "对 golden case 做质量排查"
  (7) "批量对比 golden 和原始 golden" (8) "生成 golden 验证报告"。
---

# PyPTO Testcase 转 Benchmark Skill

## 核心原则

1. **转换阶段**：只负责生成 benchmark case 文件（`{N}_{OpName}.py`），**不主动生成** REQUIRE.md、task_desc.py、SPEC.md、impl.py、golden.py、test.py、pypto_impl.py
2. **测试阶段**：benchmark 框架会自动生成 `REQUIRE.md`、`task_desc.py`；PyPTO 工作流会生成 `SPEC.md`、impl、golden、test、pypto_impl，这些不是转换阶段的职责
3. **问题定位**：测试失败时，先确认是 case 本身转换错误还是后续算子生成异常，转换错误自行修正，算子生成异常作为结果汇报给用户

## 前置环境检查

执行任何步骤前，必须先验证以下环境条件。任一条件不满足时，暂停转换并引导用户修复。

```bash
# 检查 1: CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 检查 2: PTO-ISA 源码路径（必须用源码版，因为8.5版本cann包缺少必要头文件）
export PTO_TILE_LIB_CODE_PATH=/workspace/project/pto-isa
# 检查 3: NPU 设备
test -n "$TILE_FWK_DEVICE_ID" || export TILE_FWK_DEVICE_ID=0
# 检查 4: benchmark 内置数据集存在
test -d benchmark/KernelBench
# 检查 5: PyPTO 可正常 import python3 -c "import pypto; print(pypto.__version__)"
```

## 转换流程概述

转换分为 **两个阶段**：

1. **Phase 1 — Benchmark Case 生成**（必选）：分析 testcase → 生成 KernelBench 用例 → **Golden 功能覆盖验证**（强制，1:1 对比原始 test）→ NPU 预检（唯一产物：`{N}_{OpName}.py`）
2. **Phase 2 — 回归验证**（可选）：运行 benchmark 测试，确认 case 本身没有问题

**转换开始前，必须先向用户确认**：

- 是否需要执行 Phase 2 回归验证？
- 如果需要，询问运行产物根目录（YAML `output.root_dir`，默认当前目录下临时目录）

如用户回答不需要回归验证，则仅完成 Phase 1 的 case 生成即可结束。

---

## Phase 1: Benchmark Case 生成

### Step 1: 读取 benchmark/docs/add-new-case.md 确认规则

**转换开始前，必须先读取 `pypto-gym/benchmark/docs/add-new-case.md`**，确认当前目标 case 文件的生成规则。该文档可能随框架版本更新而变化。

需要确认的关键信息：


| 确认项                   | 来源                  | 示例                                                                                                |
| --------------------- | ------------------- | ------------------------------------------------------------------------------------------------- |
| case 放置路径             | add-new-case.md 第1节 | 仓内 `benchmark/KernelBench/pto_case/` 或外部 `KernelBench/<level>/` |
| 命名格式                  | add-new-case.md 第1节 | `<N>_<CaseName>.py`                                                                               |
| 序号规则                  | add-new-case.md 第1节 | 从当前 level 未占用编号继续递增                                                                               |
| **FORMULA（数学公式）**     | add-new-case.md 第3节 | 文件顶层添加 `FORMULA = "out[m,n] = x[m,k] @ w[k,n]"`                                                   |
| **DYNAMIC_AXIS（动态轴）** | add-new-case.md 第3节 | 文件顶层添加 `DYNAMIC_AXIS = ["M", "B"]`                                                                |


**注意路径职责**：

- **仓内数据路径**：`benchmark/KernelBench/` 已完整内置上游 case 集；新增长期维护 case 可提交到 `benchmark/KernelBench/pto_case/`。
- **外部实验路径**：临时 case 可放到外部 `KernelBench/<level>/` 目录，并通过 YAML `bench_dir` 指向。
- **PyPTO 源码仓**：不保存 benchmark case 文件，只作为代码生成阶段的工作区。

当前转换应以 `benchmark/docs/add-new-case.md` 文档为准。

### Step 2: 分析现有 testcase

读取用户提供的 testcase 文件并探索关联路径下的相关文件，特别要关注README.md，提取以下关键信息。

** testcase 通常有两种结构，需分别处理：**

#### 标准结构（有独立 golden 函数）


| 提取项       | 来源                            | 示例                                     |
| --------- | ----------------------------- | -------------------------------------- |
| golden 函数 | `golden_*()` 或 `get_golden()` | `golden_fused_swiglu_fwd(x, w_g, ...)` |
| 输入张量      | `get_inputs()` 或 test 中构造     | `x: [M, K], w_g: [K, N]`               |
| 权重张量      | `__init`__ 或 test 中构造         | `w_g, w_fc, b_g, b_fc`                 |
| 数据类型      | dtype 参数                      | `torch.bfloat16`                       |
| 测试尺寸      | shape 常量                      | `M=220000, K=512, N=1024`              |
| 精度容差      | `assert_allclose` 参数          | `rtol=0.0078125, atol=0.0001`          |


#### 非标准结构（如模型中的 PyPTO 算子实现文件）

实际模型中的算子文件可能没有独立的 `golden_*()` 函数，而是将 golden 逻辑、kernel 调用、精度校验混在一起：


| 提取项       | 来源                                                    | 说明                                                         |
| --------- | ----------------------------------------------------- | ---------------------------------------------------------- |
| golden 逻辑 | `test_*()` 中的 PyTorch 参考实现链                           | 通常是 `assert_allclose` 之前的纯 PyTorch 计算                      |
| 输入张量      | `test_*()` 中构造的输入                                     | 注意区分输入 tensor 和输出 tensor（in-place 写入的目标）                   |
| 权重张量      | `check_args()` 中的 shape 断言、kernel 签名中的常量              | 如 `hidden_size=5120, num_router_experts=160`               |
| 控制参数      | 函数签名中的非 Tensor 参数                                     | `top_k`, `renormalize`, `topk_group`, `num_expert_group` 等 |
| 数据类型      | `torch.randn(..., dtype=...)` 或 `assert dtype == ...` | 注意不同参数可能有不同 dtype                                          |
| 精度容差      | `assert_allclose(..., rtol=..., atol=...)`            | 直接提取                                                       |


**提取技巧**：

- golden 逻辑通常是 `assert_allclose` 前的最后一串 PyTorch 操作
- 如果原始实现使用了 `torch_npu.npu_xxx` 等 NPU 特有 API，需要用纯 PyTorch 近似等价实现（见下方"NPU API 纯 PyTorch 近似"）

### Step 3: 生成 KernelBench 格式用例（唯一产物）

根据 Step 1 确认的放置路径，创建 `{N}_{OpName}.py`：

**强制规则**：

- `class Model(nn.Module)` 必须存在
- 权重必须用 `nn.Parameter()` 注册，否则 `model.to(device)` 不会迁移到 NPU
- `get_inputs()` 返回列表 `[x]`
- `get_init_inputs()` 返回 `Model.__init`__ 所需参数
- 如 benchmark/docs/add-new-case.md 要求，顶层添加 `FORMULA` 和 `DYNAMIC_AXIS`

```python
import torch
import torch.nn as nn

FORMULA = "out[m, n] = SiLU(x @ W_g) * (x @ W_fc)"
DYNAMIC_AXIS = ["M"]

class Model(nn.Module):
    def __init__(self, k: int = 512, n: int = 1024):
        super().__init__()
        self.w_g = nn.Parameter(torch.randn(k, n, dtype=torch.bfloat16) / math.sqrt(k))
        # ... 其他权重

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = x @ self.w_g + self.b_g
        # ... golden 计算逻辑
        return gate * torch.sigmoid(gate) * fc

def get_inputs():
    return [torch.randn(220000, 512, dtype=torch.bfloat16)]

def get_init_inputs():
    return [512, 1024]
```

**重要**：输入 tensor 应进行数值缩放（如 `/ math.sqrt(M)`），避免 BF16 数值溢出导致精度验证失败。

#### In-place 实现转 return 形式

原始 PyPTO 算子实现通常是 **in-place** 的（结果写入传入的输出 tensor）：

```python
# 原始 in-place 风格
def gate(hidden_states, gate_weight, router_logits_out):
    ...  # 结果写入 router_logits_out

def attention_pre_quant(..., query, key, value, residual_res):
    ...  # 结果写入 query/key/value/residual_res
```

KernelBench 格式要求 `forward()` **返回结果**，转换时需：

- 移除输出 tensor 参数
- 在 `forward()` 内创建输出 tensor 并返回
- 多输出时返回 `tuple[torch.Tensor, ...]`

```python
# 转换后（return 风格）
def forward(self, hidden_states):
    router_logits = torch.matmul(hidden_states, self.gate_weight.t())
    return router_logits

def forward(self, hidden_states, residual, cos, sin):
    ...
    return query, key, value, residual_out
```

#### 控制参数（非 Tensor 参数）处理

原始实现中的控制参数（`int`, `bool` 等标量）不能作为 `forward()` 的输入（KernelBench 格式要求 `get_inputs()` 只返回 tensor）。处理方式：

- 在 `Model.__init__()` 中接收并存储为 `self.xxx`
- 在 `forward()` 中通过 `self.xxx` 访问
- `get_init_inputs()` 返回这些参数的默认值

```python
class Model(nn.Module):
    def __init__(self, num_router_experts=160, top_k=8, renormalize=True):
        super().__init__()
        self.top_k = top_k
        self.renormalize = renormalize
        ...

    def forward(self, router_logits):
        if self.renormalize:
            ...
        return topk_weights, topk_ids

def get_init_inputs():
    return [160, 8, True]  # → Model.__init__(160, 8, True)
```

#### NPU 特有 API 的纯 PyTorch 近似

原始 testcase 可能使用 `torch_npu.npu_xxx` 等 NPU 特有 API，这些在 KernelBench golden 中不可用。需替换为纯 PyTorch 近似：


| NPU API                                                            | 纯 PyTorch 近似                                                              | 说明              |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------- | --------------- |
| `torch_npu.npu_quantize(x, scale, offset, dtype, axis, symmetric)` | `(x * scale + offset).round().clamp(-128, 127).to(torch.int8)`            | per-channel 量化  |
| `torch_npu.npu_quant_matmul(x_int8, weight, deq_scale, bias)`      | `x_int8.float() @ weight.float()` 然后 `* deq_scale + bias`（⚠️ 必须用 float32，aclnn 不支持 int32 MatMul） | 量化 MatMul + 反量化 |


注意：近似实现只需在 golden 精度可接受范围内即可，不需要 100% 等价。

#### 关于 FORMULA 和 DYNAMIC_AXIS（重点）

根据 benchmark/docs/add-new-case.md 要求，新增 case **应尽可能提供**这两个大写全局变量：

`**FORMULA` — 数学公式**

- 用途：`case_loader.py` 自动提取并写入 `REQUIRE.md` 的 `### 1.3 数学公式` 小节
- 格式：简洁的数学表达式，用下标索引描述运算
- 示例：
  ```python
  FORMULA = "out[m, n] = SiLU(x[m, k] @ W_g[k, n]) * (x[m, k] @ W_fc[k, n])"
  ```
- 要求：必须是可被 `ast.literal_eval` 解析的字符串常量，不能是运行时计算结果

`**DYNAMIC_AXIS` — 动态轴列表**

- 用途：`case_loader.py` 自动提取并写入 `REQUIRE.md` front matter 的 `dynamic_axis` 字段
- 格式：字符串列表，每个元素是动态维度的名称（大写字母）
- 示例：
  ```python
  DYNAMIC_AXIS = ["M", "B"]
  ```
- 要求：必须是可被 `ast.literal_eval` 解析的列表常量
- 注意：区分动态轴（运行时变化，如 batch 维度 M）和固定轴（如 K、N）

**兼容规则**：

- 老 KernelBench case 没有这两个变量时，不会输出对应内容
- 新 case 有其中一个变量时，只输出对应部分
- 变量必须是可被 `ast.literal_eval` 解析的常量表达式；不要写成运行时计算结果

#### 转换后检查清单

Phase 1 结束前，逐项确认：

- 原始实现如果是 **in-place** 的，已改为 `return` 形式
- 如果是 **多输出**，`forward()` 返回 `tuple[torch.Tensor, ...]`
- 是否使用了 **NPU 特有 API**？如有，是否已替换为纯 PyTorch 近似
- **控制参数**（int/bool）是否已在 `__init__` 中注册为 `self.xxx`
- 权重是否使用 `nn.Parameter()` 注册
- `get_inputs()` 返回列表 `[x]`
- `get_init_inputs()` 返回 `Model.__init__` 所需参数
- 输入 tensor 已进行数值缩放（如 `/ math.sqrt(M)`）
- `FORMULA` 已添加（如 benchmark/docs/add-new-case.md 要求）
- `DYNAMIC_AXIS` 已添加（如 benchmark/docs/add-new-case.md 要求）
- **`forward()` 中无 int32/int64 中间 MatMul（aclnn 不支持整数 MatMul，量化 golden 用 float() 代替）**

#### Golden 功能覆盖验证（强制 — 新增）

**检查清单通过后、NPU 预检之前**，必须执行以下 1:1 功能覆盖验证。目标是确保生成的 golden 与原 test 的实现范围完全一致，**不允许私自裁剪计算模块**。

##### 验证步骤：

**1. 提取原始 test 的计算模块清单**

读取原始 test 文件中的 golden 参考函数（通常是 `test_*()` 中的纯 PyTorch 计算链或独立的 `*_compute()` / `golden_*()` 函数），逐行梳理出所有计算步骤和中间变量，列出清单：

| 计算步骤 | 输入变量 | 输出变量 | 操作类型 |
|---------|---------|---------|---------|
| x @ w_dq | x, w_dq | q_a | MatMul |
| RMSNorm(q_a, γ) | q_a, gamma | q_norm | RMSNorm |
| ... | ... | ... | ... |

**注意**：
- 量化计算链（quant → matmul → dequant）应作为完整步骤检查，不能只覆盖 matmul 而遗漏 quant/dequant
- 多输入/多输出的分支路径（如 query path / key path）必须**全部覆盖**，不能只覆盖其中一个分支
- scatter_update / cache 更新等副作用操作也必须纳入检查，不能因为"非核心计算"而跳过

**2. 逐步骤对比生成的 golden**

将步骤 1 的清单，逐行与生成的 `Model.forward()` 对比，确认结构覆盖：

| 原 test 计算步骤 | Golden 中对应行 | 状态 |
|-----------------|----------------|------|
| x @ w_dq | L125-130 | ✅ 已覆盖 |
| RMSNorm(q_a) | L131 | ✅ 已覆盖 |
| ... | ... | ... |

**比对标准**：
- ✅ **已覆盖**：golden 中存在功能等价的纯 PyTorch 实现
- ⚠️ **简化覆盖**：golden 中路径存在但实现有裁剪（如跳过 quant/dequant、合并步骤）—— 需判断简化是否影响覆盖率
- ❌ **缺失**：golden 中完全没有对应计算——**必须补充**

**3. 运行时数值对比（强制）**

结构覆盖确认后，必须编写并执行运行时对比脚本，逐一验证 golden benchmark 与原始 test golden 在相同输入下输出一致。对比原则：**纯 PyTorch golden 函数对纯 PyTorch golden 函数，不调用 NPU kernel**。

##### 对比脚本模板

对每个 case，编写独立验证脚本并放到 `benchmark/KernelBench/pto_case/verification_reports/case{N}_verify.py`：

```python
#!/usr/bin/env python3
"""Case N: 验证 benchmark golden 与原始 test golden 的数值一致性。"""
import sys, math
import torch

# 1. 从原始 test 文件提取 golden 参考函数（纯 PyTorch 部分）
#    如 original_test.py 中的 golden_xxx() 或 xxx_compute()
def original_golden(inputs, weights):
    ...  # 逐行复制原始 golden 函数的核心计算

# 2. 从 benchmark golden case 的 Model.forward() 提取等效计算
#    可使用相同权重跑 benchmark 的 Model。或直接提取 forward() 中的纯计算逻辑

# 3. 用相同随机种子生成 3 组不同尺寸的输入
configs = [
    (42, "默认", 原始默认尺寸...),
    (142, "小规模", 缩小尺寸...),
    (242, "中等规模", 其他尺寸...),
]

# 4. 逐组运行两侧 golden，逐位比较输出
for seed, tag, dims in configs:
    torch.manual_seed(seed)
    inputs = generate_inputs(dims)
    out_bench = run_benchmark_golden(model, inputs)
    out_orig = original_golden(inputs, model_weights)
    max_diff = (out_bench.float() - out_orig.float()).abs().max().item()
    # INT8 输出额外计算精确匹配率
    match_rate = (out_bench == out_orig).float().mean().item()
    print(f"[{tag}] max_diff={max_diff:.6e} match={match_rate:.4f} {'PASS' if max_diff<1e-5 else 'FAIL'}")
```

**脚本编写原则**：

- **3 组配置**：原始默认尺寸 + 小规模（缩小 2-4 倍）+ 中等规模（换一组尺寸），每组不同 seed
- **仅比 golden**：不调用 NPU kernel，只比较原始 test 的 golden 函数与 benchmark golden 的输出
- **多输出**：每个输出独立报告 max_diff，量化输出（INT8）用 `==` 或精确匹配率，浮点输出用绝对差
- **遇到差异**：脚本末尾应打印差异最大的元素及其位置，便于定位根因
- **可独立运行**：`python3 benchmark/KernelBench/pto_case/verification_reports/case{N}_verify.py`

##### 当原始 test 内嵌在源码中时

部分 case 的 golden 参考函数直接写在源码文件（如 `src/.../gmm_mxfp8.py` 中的 `compute_golden_result()`）而非独立的 test 文件。对比脚本同样提取该函数，按照上述模板进行数值比对。

**4. 处理覆盖缺口与数值差异**

验证发现覆盖缺口或数值不匹配时，按以下优先级处理：

| 缺口/差异类型 | 处理方式 |
|------------|---------|
| 计算步骤缺失 | **自行补充到 golden**，确保 1:1 覆盖 |
| 简化过度导致功能丢失 | **自行修正**，恢复被裁剪的计算 |
| 数值差异（max_diff > 1e-3） | **分析根因并修正**，见下方常见差异表 |
| 无法在 KernelBench 格式中表达 | **向用户报告**，明确说明原因和建议 |
| NPU 特有 API 无法等价替换 | **向用户报告**，说明限制和替代方案 |

**常见数值差异根因及修复**：

| 差异现象 | 根因 | 修复方法 |
|---------|------|---------|
| RMSNorm 输出偏差 | benchmark 添加了 epsilon，原始无 epsilon | 移除 epsilon，使用 `sum(x² * 1/N)` 计算 |
| INT8 量化输出不匹配 | benchmark 用 `round().clamp().to(int8)`，原始用 `round().to(int32).to(float16).trunc().to(int8)` | 对齐至原始的 int32→float16→trunc 量化路径 |
| RoPE 输出差异大（MAE > 0.1） | benchmark 用半切模式 `chunk(2)`，原始用交织模式 `reshape→permute→rotate` | 实现原始的交织：`reshape(b,s,h,d//2,2).permute(0,1,2,4,3).reshape(...)` |
| k_nope 量化不匹配 | benchmark 用 per-channel quant，原始用 4 组 per-token quant | 改为 `reshape(t,4,kvl//4)` → `per_token_quantize` → `reshape(t,kvl)` |
| softmax 输出 ~1e-4 差异 | benchmark 用 matmul+softmax，原始用 online softmax | 实现原始的分块 online softmax（逐 K 迭代 max/sum/out 校正） |
| 结构简化（缺失计算分支） | benchmark 只实现了简化版算法 | **补全所有计算分支**，不允许私自裁剪 |
| 缺少权重缩放因子 | benchmark 遗漏了 `scale = (dim1^-0.5)*(dim2^-0.5)` | 补全原始 golden 中的缩放因子 |
| BF16 精度级差异（~1e-4） | 内存布局遍历方式（2D vs 4D）导致 BF16 舍入不同 | 对齐遍历方式（如 block-concat-then-slice 替代 per-position stack） |

**禁止行为**：
- ❌ 私自裁剪原始 test 中的计算模块（如：只生成 query path 而跳过 key path）
- ❌ 将量化/反量化步骤合并简化为直接 FP 计算而不告知用户
- ❌ 遇到覆盖缺口时静默跳过，不作为缺口汇报
- ❌ 以"简化 golden"、"降低复杂度"等理由省略计算步骤
- ❌ 发现数值差异后标记 PASS 而不修复
- ❌ 只对齐部分分支而遗漏其他分支

**验证通过条件**：所有计算步骤状态为 ✅，所有 3 组配置数值对比 max_diff=0（或 BF16 精度容差内），所有输入/输出匹配通过。

##### 验证报告（须输出为 MD 文件）

验证完成后，将报告输出到 `benchmark/KernelBench/pto_case/verification_reports/case{N}_{name}_report.md`：

```
# Case N：{算子中文名称} — 验证报告

## 源码映射
| 项目 | 路径 |
|------|------|
| Golden Case | benchmark/KernelBench/pto_case/{N}_{name}.py |
| 原始测试 | tests/ops/{path}/test_{name}.py |
| 源码实现 | src/pypto_gym/ops/pypto_tile/{path}/{name}_impl.py |

## 公式对比

两侧 golden 计算的核心公式（文本描述 + 关键代码行引用）。

## 运行时对比结果

| 配置 | 尺寸参数 | max_diff | 精确匹配率 | 状态 |
|------|---------|----------|-----------|------|
| 默认 | (m=..., k=..., n=...) | 0.0 | 100% | 通过 |
| 小规模 | ... | 0.0 | 100% | 通过 |
| 中等规模 | ... | 0.0 | 100% | 通过 |

## 差异分析与修复

[如有差异，逐项列出根因和修复内容]
[如无差异，标注"无需修复，两侧 golden 完全一致"]

## 结论

[通过 / 已修复后通过 / 存在差异需用户决策]
```

**多 case 场景下，额外生成汇总文件** `benchmark/KernelBench/pto_case/verification_reports/MASTER_SUMMARY.md`，包含逐 case 结果表、修复清单、NPU 验证结果（如有）。

#### Case NPU 预检（强制）

Case 文件生成并通过检查清单后，**必须**在 NPU 上执行预检，验证 case 的 `Model.forward()` 在 NPU 上可正常运行。

```bash
python .agents/skills/pypto-testcase-to-benchmark/scripts/validate_case_npu.py \
    {case文件路径} \
    --device ${TILE_FWK_DEVICE_ID:-0}
```

**三种结果处理**：

| 退出码 | 含义 | 处理 |
|--------|------|------|
| 0 ✅ | NPU 预检通过 | 继续 Phase 1 收尾 |
| 1 ❌ | NPU 预检失败 | 根据错误信息修复 case 后重新预检，常见失败见下方 |
| 2 ⚠️ | NPU 不可用 | **直接向用户报告风险**："当前环境无 NPU 设备，无法验证 case 兼容性。请在 NPU 环境中重新运行预检，或知悉以下风险：case 中使用了 NPU API 近似替换，可能存在 dtype/算子兼容性问题。" 等待用户指示后再继续 |

**退出码 1 常见失败及修复**：

| 错误信息 | 修复 |
|----------|------|
| `aclnn.*does not support.*int32` | 将 golden 中 int32 MatMul 改为 `float()` MatMul |
| `Expected all tensors.*same device` | 权重用 `nn.Parameter()` 注册 |
| `ModuleNotFoundError.*torch_npu` | NPU 环境未配置，按退出码 2 处理 |

**至此，Phase 1 结束。唯一的产物是 `{N}_{OpName}.py`。不生成任何其他文件。**

---

## Phase 2: 回归验证（可选）

### 用户确认

Phase 1 完成后，**必须向用户确认**是否执行回归验证：

> Phase 1（Benchmark Case 生成）已完成，已生成 `{N}_{OpName}.py`。
> 是否需要执行 Phase 2 回归验证？
>
> - 需要：运行 benchmark 测试验证 case 本身是否存在问题
> - 不需要：转换结束

### 环境变量配置（如用户确认需要）

**询问用户运行产物根目录**，默认生成在当前目录下：

```
请输入 output.root_dir（运行产物根目录，默认 ./benchmark_runs/<本次任务名>）：
```

生成一个临时 YAML 并执行 benchmark：

```yaml
# /tmp/pypto_case_regression.yaml
bench_dir: ""
cases: "pto_case={N}_{OpName}"
devices: [0]
concurrency: 1
output:
  root_dir: "/path/to/output/root"
pypto:
  timeout_sec: 10800
  pref_round: 3
  skip_pypto_gen: true
verifier:
  mode: "performance"
  verifier_mode: "direct"
  verify_rtol: 1.0e-2
  verify_atol: 2.5e-2
```

```bash
python -m benchmark run --config /tmp/pypto_case_regression.yaml --foreground
```

**说明**：

- `pypto.skip_pypto_gen=true`：复用现成 `custom/<op>/` 产物，只跑 verifier。回归验证推荐此模式。
- `pypto.skip_pypto_gen=false`：让 pypto-op-orchestrator 真跑 Stage 1-7 算子开发。仅当需要完整端到端集成测试时使用。
- `verifier.verifier_mode=direct`：不烧 LLM，直接调 KernelVerifier。快速回归推荐。
- `verifier.verifier_mode=opencode`：走 LLM skill 验证（含反作弊审阅）。需要配置 opencode CLI。

### 测试失败时的分析逻辑

**关键原则**：测试失败时，先区分是 case 本身的问题还是后续算子生成的问题。

#### 情况 A：Case 本身转换错误（需自行修正）

**判断特征**：

- verifier.log 中显示精度不匹配（VERIFY_FAILED），且 golden 与 impl 差异巨大（>1%）
- 错误出现在 verify 阶段，pypto 编译成功但结果不对
- 通常是以下 case 本身的问题：
  - golden 函数逻辑与原始 testcase 不一致
  - `get_inputs()` 返回的 shape/dtype 错误
  - `get_init_inputs()` 与 `Model.__init`__ 参数不匹配
  - 输入未数值缩放导致 BF16 溢出
  - `forward()` 返回值 shape/dtype 与预期不符

**处理**：

1. 对比原始 testcase 的 golden 函数和 case 中的 `Model.forward()`
2. 检查 `get_inputs()` 和 `get_init_inputs()` 的 shape、dtype、顺序
3. 修正 case 文件后重新运行测试

#### 情况 B：后续算子生成异常（作为结果汇报给用户）

**判断特征**：

- pypto 编译失败（如 aicore error、COMPILE_CODE_FAILED）
- pypto-op-orchestrator 阶段报错（Stage 1-7 中某阶段失败）
- impl.py / pypto_impl.py 生成异常
- 非 case 文件本身的问题

**处理**：

- 将错误日志整理后汇报给用户
- 说明这是 benchmark 框架/算子生成阶段的问题，非 case 转换错误
- 不尝试自行修改非 case 相关的文件

### 查看结果

执行完成后，报告位于：

```
<output.root_dir>/report/summary.md       # 总览报告
<output.root_dir>/report/summary.json     # JSON 结果
<output.root_dir>/logs/benchmark.out      # 后台 run stdout；--foreground 时在当前终端
```

单 case 详细结果：

```
<output.root_dir>/report/<level>/{OpName}/result.json
<output.root_dir>/report/<level>/{OpName}/verifier.log
```

---

## 产物清单


| 文件                | 来源               | 作用                                                        |
| ----------------- | ---------------- | --------------------------------------------------------- |
| `{N}_{OpName}.py` | **转换阶段新建（唯一产物）** | KernelBench 格式 case 文件，含 Model/get_inputs/get_init_inputs |


**以下文件不由转换阶段生成，由 benchmark 测试流程自动产生**：


| 文件                       | 来源             | 作用                     |
| ------------------------ | -------------- | ---------------------- |
| `REQUIRE.md`             | benchmark 自动生成 | Stage 1 用户需求输入          |
| `task_desc.py`           | benchmark 自动生成 | 原始用例缓存                 |
| `SPEC.md`                | PyPTO 工作流生成    | 算子需求规格                 |
| `{OpName}_impl.py`       | PyPTO 工作流生成    | PyPTO kernel + wrapper |
| `{OpName}_golden.py`     | PyPTO 工作流生成    | PyTorch 参考实现           |
| `test_{OpName}.py`       | PyPTO 工作流生成    | 精度测试脚本                 |
| `{OpName}_pypto_impl.py` | PyPTO 工作流生成    | ModelNew 包装（桥接入口）      |


## 常见陷阱


| 陷阱                      | 现象                                                  | 解决                                                          |
| ----------------------- | --------------------------------------------------- | ----------------------------------------------------------- |
| 未读 add-new-case.md      | case 放错路径、缺少 FORMULA/DYNAMIC_AXIS                   | 转换前必须先读文档确认规则                                               |
| 转换阶段越权生成产物              | 手动创建 impl/golden/test 等文件                           | **不要生成**，这些由 benchmark 流程自动处理                               |
| 混淆 case 错误与算子生成错误       | 误改 benchmark 自动生成的文件                                | 测试失败时先判断：是 case 本身问题还是算子生成问题                                |
| 权重未注册为 Parameter        | `RuntimeError: Expected all tensors on same device` | `nn.Parameter(torch.randn(...))`                            |
| init/forward 参数混淆       | Model 初始化错误或 forward 调用失败                           | `get_init_inputs()` 给 `__init`__，`get_inputs()` 给 `forward` |
| 输入未数值缩放                 | BF16 溢出导致精度验证大面积失败                                  | 输入 tensor 进行 `/ math.sqrt(M)` 等缩放                           |
| **未处理 in-place 输出**     | `forward()` 参数包含输出 tensor，但 KernelBench 期望 return   | 移除输出参数，在 `forward()` 内创建并返回结果                               |
| **NPU API 未替换**         | `torch_npu.npu_xxx` 在纯 CPU 环境报错                     | 用纯 PyTorch 近似等价实现（如量化用 `round().clamp()` 替代）                |
| **控制参数混在 get_inputs 中** | `get_inputs()` 返回 int/bool，case_loader 解析失败         | 控制参数移到 `__init`__，`get_inputs()` 只返回 tensor                 |
| **多输出未用 tuple**         | `forward()` 返回多个 tensor 但未被正确识别                     | 显式声明返回类型 `-> tuple[torch.Tensor, ...]`                      |
| **int32/int64 中间 MatMul 导致 aclnn 失败** | `aclnn matmul 不支持 int32 输入`、编译失败 | Golden 中避免 int32/int64 MatMul。量化 MatMul golden 用 `float()` 代替 `.to(torch.int32)`，精度差异在验证容差内 |
| **golden 简化过度** | 验证阶段发现 benchmark golden 与原始 golden 差异大 | 使用 Golden 功能覆盖验证流程逐步骤对比，补全所有缺失的计算模块 |
| **RoPE 约定不一致** | q_rope/k_rope 输出 MAE > 0.1 | 检查原始测试的 RoPE 实现（半切 vs 交织），对齐至原始约定 |
| **量化路径不同** | INT8 输出精确匹配率 < 100% | 对齐量化路径：`round→int32→float16→trunc→int8` |
| **RMSNorm epsilon 差异** | norm 输出微小差异 | 检查原始实现是否使用 epsilon，对齐即可 |


## 参考文档

- **新增 Case 规则**：`pypto-gym/benchmark/docs/add-new-case.md`（转换前必读）
- **转换映射表与完整示例**：见 `references/conversion_patterns.md`
