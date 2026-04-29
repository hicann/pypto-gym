---
name: pypto-fused-op-integration
description: PyPTO算子整网集成工作流。打点采集真实tensor→Golden验证→算子开发→模型集成→端到端验证。触发词：算子融合、整网集成、replace small ops、模型算子替换（GLM、LLaMA、Qwen、MoE、Attention等）。
---

# PyPTO 融合算子整网集成 Skill

将 PyPTO 融合算子替换到整网中，替代原始算子实现的完整工作流程。

---

## 工作流程概览

**阶段一：前置准备** → 环境验证 + 网络基线 ★ + 智能推荐
**阶段二：理解验证** → 打点采集 ★ + Golden编写（场景区分） + 场景验证 ⚠️
**阶段三：算子开发** → 设计方案 + 实现 + 单算子验证 ★
**阶段四：模型集成** → 目录结构 + 适配层 + 调用逻辑 + 缓存处理 ★
**阶段五：验证与提交** → 端到端验证（必须） + 性能调优 [可选] + 提交

> **★ 标注：** 关键步骤，必须完成  
> **⚠️ 标注：** 需特别注意  
> **[可选]：** 后续迭代进行

---

## 关联Skill获取方式

本skill位于 `pypto-gym` 仓库（`cann/pypto-gym`），与 `migrate-huggingface-to-npu` 同级。

本skill引用的其他skill不在本仓库中，**均位于 pypto 主仓库**：
```
https://gitcode.com/cann/pypto/tree/master/.agents/skills
```

| 被引用Skill | 用途 | 所在仓库 |
|------------|------|---------|
| `pypto-environment-setup` | 环境安装（步骤1） | cann/pypto |
| `pypto-api-explore` | 探索pypto API（步骤2） | cann/pypto |
| `pypto-golden-generate` | Golden生成（步骤4） | cann/pypto |
| `pypto-op-design` | 设计方案（步骤7） | cann/pypto |
| `pypto-op-develop` | 算子实现（步骤8） | cann/pypto |
| `pypto-precision-compare` | 精度对比（步骤9） | cann/pypto |
| `pypto-precision-debug` | 精度调试（步骤13） | cann/pypto |
| `pypto-aicore-error-locator` | aicore错误定位（步骤13） | cann/pypto |
| `pypto-host-stacktrace-analyzer` | 堆栈分析（步骤13） | cann/pypto |
| `pypto-op-perf-tune` | 性能调优（步骤14） | cann/pypto |
| `pypto-issue-creator` | 创建Issue（步骤15） | cann/pypto |
| `pypto-pr-creator` | 创建PR（步骤15） | cann/pypto |
| `pypto-intent-understand` | 需求理解 | cann/pypto |

> **获取方法：** 克隆或浏览 `https://gitcode.com/cann/pypto`，skill文件均在 `.agents/skills/` 目录下，每个skill对应一个子目录。

---

## 详细步骤指南

### 阶段一：前置准备

#### 步骤 0：确认模型信息 ★ 【必须最先执行】

**上游 Skill：** 若模型通过 `migrate-huggingface-to-npu` 迁移，
模型目录已包含 `scripts/ask_*.py` + `core/modeling_*.py` + `.git`，
环境基线已验证，可直接跳入步骤 2（需求分析）。

**必须询问：**
- **模型路径**：权重目录（目录或子目录下包含 safetensors/bin）
  - 若来自 `migrate-huggingface-to-npu`，直接沿用上游的 `model_weight_dir`

**自动查找（无需询问）：**
- 在 `{模型路径}/scripts/` 下查找 `*.py` 执行脚本
- 在 `{模型路径}/core/` 下查找 `modeling_*.py`（上游 skill 已准备）
- 若唯一 → 直接使用
- 若多个 → 询问用户选择
- 若无 → 询问用户提供

> **原则：** 最小化用户输入，最大化自动推断。

---

#### 步骤 1：前置验证（环境+网络基线）★

**目标：** 确认环境和网络可运行，作为后续工作基础。

**环境验证：**
- 推荐 Skill：`pypto-environment-setup`
- 检查 NPU 状态：`npu-smi info`

**网络基线验证：**
- 检查模型路径和文件完整性
- 执行完整推理，确认输出正常（无乱码/NaN/Inf）
- **关键成功标志：生成对 Prompt 的通顺回答**

> **为什么需要？** 网络必须可运行，否则后续工作基础不稳定。

**验证检查点：**
- ✅ NPU 驱动正常
- ✅ 模型可正常加载和推理
- ✅ 输出为自然语言（非乱码）

---

#### 步骤 2：需求分析（智能推荐）

**目标：** 分析网络可融合部分，推荐给用户确认。

**操作：**
1. **分析网络结构**：阅读模型代码，识别算子组合模式（Attention、MoE、FFN等）
2. **匹配 pypto 现有算子**：搜索 `models/` 目录下的实现案例
3. **推荐融合点**：
   
   向用户推荐候选融合点（示例格式）：
   ```
   发现可融合算子组合：
   | 位置 | 原始实现 | pypto算子 | 收益 |
   | Attention层 | Q/K/V投影+Softmax+Output | pypto.flash_attention | 减少3次matmul |
   | FFN层 | Gate+Up+SwiGLU+Down | pypto.swiglu_ffn | 减少中间存储 |
   
   请确认目标算子。
   ```
4. **无合适推荐**：直接询问用户融合位置和目标

**推荐 Skill：** `pypto-api-explore`（探索 pypto API）

---

### 阶段二：理解验证 ⚠️

> **为什么需要？** 推测的计算逻辑需 Golden 验证确认。

#### 步骤 3：打点采集真实 Tensor 信息 ★

**目标：** 从原始网络采集真实 shape/dtype，构造必须 pass 的测试用例。

**操作：**
1. 定位并插入一行 print（采集所有外部输入的 shape/dtype）
2. 运行原始网络采集
3. 采集完成后删除打印，恢复代码原状
4. 创建 test_cases.json

**注意事项：**
- ⚠️ **不要限制打印次数**：禁止使用计数器限制打印次数（如 `if counter < 5`），否则会漏掉不同场景（如不同 layer、不同 seq_len、不同 shape），导致测试用例不完整
- 应采集所有调用场景，覆盖不同 shape/dtype 组合

**格式参考：**
- 统一格式以 `pypto-op-develop/templates/test_cases-template.json` 为准
- 多输入算子补充说明见 `references/test-cases-template.md`

**输出物：** test_cases.json

**存放位置：** `models/{model_name}/pto_kernels/xxx/test/test_cases.json`

**关键原则：** 真实用例是必须 pass 的基准，覆盖所有调用场景。

---

#### 步骤 4：编写 Golden 脚本（场景区分）

**目标：** 编写 PyTorch 参考实现，精度对比基准。

**场景判断：** 检查**被替换逻辑**是否使用 torch_npu 融合算子（只看逻辑本身，不看文件import）。

---

| 场景 | 判断条件 | 策略 | 输出件 |
|------|---------|------|--------|
| **场景A** | 只使用基础算子（matmul、softmax等） | 直接复制原始代码，无需理解验证 | `xxx_golden.py` |
| **场景B** | 使用融合算子（flash_attention等） | 用 torch 重写等价实现，必须验证 | `xxx_golden.py` + `test_xxx_golden_correctness.py` |

**通用要点：**
- 纯 PyTorch，禁止引入 pypto/torch_npu
- 导出 `{op}_golden()` 函数
- 独立 `{op}_golden.py` 文件

**推荐 Skill：** `pypto-golden-generate`

---

#### 步骤 5：验证理解正确性（分场景验证）

**目标：** 验证 Golden 与原始实现等价。

---

**场景A：未引用 torch_npu**

**无需验证**：Golden = 原始代码，直接进入步骤7。

---

**场景B：引用 torch_npu ★ 必须验证**

**输出件清单：**

| 输出件 | 文件名 | 存放位置 | 要求 |
|--------|--------|---------|------|
| 测试用例 | `test_cases_golden.json` | `pto_kernels/test/` | 必须：torch_npu 参数信息 |
| 验证脚本 | `test_{op}_golden.py` | `pto_kernels/test/` | 必须：对比 Golden vs torch_npu |

**验证流程：**
1. **构造测试用例**（来自步骤3采集的 shape/dtype）
2. **对比 torch Golden 与 torch_npu**：
   ```python
   output_npu = torch_npu.flash_attention(query, key, value, **params)
   output_golden = xxx_golden(query, key, value, **params)
   assert_allclose(output_npu, output_golden, rtol=1e-3, atol=1e-3)
   print("[PRECISION_PASS] Golden 与 torch_npu 一致")
   ```
3. **一致后替换 Golden 到整网**，验证输出正确
4. **全部通过** → 进入步骤7

---

**验证检查点：**
- ✅ Golden 与 torch_npu 一致（diff < 1e-3）
- ✅ 整网替换后输出正常
- ✅ 无 NaN/Inf

---

#### 步骤 6：决策与迭代

- **通过 ✅**：
  - 场景A：无需验证 → 进入步骤7
  - 场景B：torch Golden 与 torch_npu 一致 + 整网验证通过 → 进入步骤7
- **失败 ❌**：
  - 场景A：不存在（Golden = 原始代码）
  - 场景B：重新理解融合算子语义，修正 Golden

---

### 阶段三：算子开发

#### 步骤 7：设计方案

**目标：** 设计 PyPTO 实现方案（API 映射、Tiling 策略）。

**推荐 Skill：** `pypto-op-design`

**⚠️ 阶段三常见陷阱：**

| 陷阱 | 原因 | 表现 | 预防 |
|------|------|------|------|
| 内置API不支持动态轴 | 内部`cast`拒绝dim=-1 | `FC0000: invalid shape value: -1` | 设计前先查API源码，含`cast`则走手动实现 |
| `set_vec_tile_shapes(x.shape[i])` | 返回值是SymbolicScalar | `F00002: Not concrete value` | 使用concrete常量(e.g. `set_vec_tile_shapes(1, 2048)`) |
| `pypto.mul(x, Element(...))` | `mul`内部二次包装Element | `TypeError: Element(Element)` | 传标量(float/int)，`mul`自动转换 |

---

#### 步骤 8：算子实现

**目标：** 编写 PyPTO 算子代码。

**推荐 Skill：** `pypto-op-develop`

---

#### 步骤 9：单算子验证

**目标：** 验证 PyPTO 实现正确性。

**编译环境：**
- 必须设置 `export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa`（kernel 编译需要）
- 设置 `export TILE_FWK_DEVICE_ID=<空闲 chip id>`

**关键说明：**
- 步骤 3 采集的真实用例是必须 pass 的基准
- 输出无 NaN/Inf，与 Golden 对齐（diff < 2e-3）

**推荐 Skill：** `pypto-precision-compare`

---

### 阶段四：模型集成

#### 步骤 10：调整目录结构

**目标：** 创建 PyPTO 算子库目录结构（按算子组织）。

**典型结构：**
```
pto_kernels/                        # 算子库顶层
├── __init__.py                     # USE_PTO开关 + 导入所有算子
│
├── xxx/                            # 算子目录（如 rms_norm、ffn、softmax）
│   ├── __init__.py                 # 导出 xxx_wrapper
│   ├── xxx_impl.py                 # PyPTO kernel（带前缀）
│   ├── xxx_golden.py               # Golden参考（带前缀）
│   ├── README.md                   # 算子文档
│   └── test/
│       ├── test_xxx.py             # 测试脚本（带前缀）
│       └── test_cases.json         # 测试用例
│
└── utils/                          # 通用工具（可选）
    └── DESIGN.md                   # 设计文档
```

**命名规则：**
- 目录名：抽象命名（如 `rms_norm`、`ffn`）
- 文件名：带算子前缀（如 `rms_norm_impl.py`）
- 模块名：`{model}_pto_kernels`（如 `qwen3_pto_kernels`），避免通用名称

---

#### 步骤 11：配置适配层

**目标：** 封装 PyPTO 算子调用。

**关键要点：**
- 开关设计：按算子粒度 `USE_PTO_{OP}`（如 `USE_PTO_RMS_NORM`），便于渐进式验证
- 函数命名：与原始算子同名，参数传递根据场景灵活设计
- 文档注释：包含目标文件、目标类、替换代码片段

**最佳实践（参考 GLM-Net）：**

| 方面 | 推荐做法 | 理由 |
|------|---------|------|
| **模块命名** | `{model}_pto_kernels` | 避免通用名称，提高可识别性 |
| **开关设计** | 按算子粒度 `USE_PTO_{OP}` | 渐进式验证，便于定位问题 |
| **函数命名** | 与原始算子同名 | 降低理解成本 |
| **参数传递** | 根据场景灵活设计（layer 对象或单独参数） | 适配不同调用位置 |
| **文档注释** | 包含替换代码片段 | 可直接复制，减少错误 |
| **allow_in_graph** | 适配层函数 `{op}_pto` 调用 `@allow_in_graph` 修饰的 `{op}_wrapper`，禁止越级调 JIT kernel | 确保 torch.compile / aclgraph 图捕获兼容 |

**参考案例：** https://gitcode.com/songle1/glm-net/blob/main/glm_pto_kernels/__init__.py

---

#### 步骤 12：修改模型调用逻辑 + sys.modules注入 ★

**目标：** 替换原始算子调用，通过 sys.modules注入绕过缓存。

**推荐方案：sys.modules注入**

**原理：** Python的 `sys.modules` 是全局模块注册表。脚本预导入算子库→注入→modeling自动获取，无缓存依赖。

**实施步骤：**

**步骤A：脚本注入（在transformers导入前）**
```python
parser.add_argument("--use-pto", action="store_true", help="启用PyPTO算子")

if args.use_pto:
    sys.path.insert(0, args.model_path)         # 1. 添加路径
    import {model}_pto_kernels                  # 2. 导入模块
    sys.modules["{model}_pto_kernels"] = module # 3. 注册全局
    module.USE_PTO_{OP} = True                   # 4. 启用算子开关

from transformers import AutoModelForCausalLM   # 之后加载模型
```

**步骤B：modeling获取**
```python
import sys

pto_kernels = sys.modules.get("{model}_pto_kernels")

def forward(self, hidden_states):
    if pto_kernels is not None and pto_kernels.USE_PTO_{OP}:
        return pto_kernels.{op}(self, hidden_states)
    # 原始torch实现（fallback）
    ...
```

**关键要点：**
- 注入位置：transformers 导入前
- 条件判断：`sys.modules.get()` + 开关启用（双重条件）
- **必须保留原始 torch 实现作为 fallback**

**验证检查点：**
- ✅ 使用 `--use-pto` → `RMS_PTO_AVAILABLE = True`（PTO生效）
- ✅ 不使用 → `RMS_PTO_AVAILABLE = False`（torch fallback）
- ✅ 本地修改算子库后即时生效

---

### 阶段五：验证与提交

#### 步骤 13：验证与排查

**目标：** 端到端精度验证 + 问题排查。

**精度验证：**
- 运行整网推理，确认输出正常
- 与原始实现对比

**问题排查：**
- 参考 Skill：`pypto-aicore-error-locator`、`pypto-host-stacktrace-analyzer`、`pypto-precision-debug`

---

#### 步骤 14：性能调优 [可选]

**目标：** 采集性能数据，分析瓶颈。

> **优先级：** 精度验证（必须） > 性能优化（可选）

**操作：**
- 采集泳道图、timeline
- 分析瓶颈（KV组装、MatMul、循环开销）
- 优化 Tile 配置、合图策略

**推荐 Skill：** `pypto-op-perf-tune`

---

#### 步骤 15：提交与文档

**目标：** 创建 Issue 和 PR。

**操作：**
1. 创建 Issue 跟踪变更
2. 提交 PR（含修改说明）

**推荐 Skill：** `pypto-issue-creator`、`pypto-pr-creator`

---

## 相关资源

### 参考模板
- **测试集模板**：`references/test-cases-template.md`

### 相关 Skill
- `pypto-environment-setup`：环境安装
- `pypto-intent-understand`：需求理解
- `pypto-golden-generate`：Golden 生成
- `pypto-op-design`：设计方案
- `pypto-op-develop`：算子实现
- `pypto-precision-compare`：精度对比
- `pypto-precision-debug`：精度调试
- `pypto-op-perf-tune`：性能调优
- `pypto-aicore-error-locator`：aicore 错误定位
- `pypto-host-stacktrace-analyzer`：堆栈分析
- `pypto-issue-creator`：创建 Issue
- `pypto-pr-creator`：创建 PR

---

**Skill 版本：** v2.4
**最后更新：** 2026-04-28
**维护者：** PyPTO Team
**更新说明：** 步骤 0 新增上游 migrate-huggingface-to-npu 感知，可直接跳入步骤 2