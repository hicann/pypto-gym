---
name: pypto-fused-op-integration
description: 将HuggingFace大语言模型迁移到华为昇腾NPU环境并集成PyPTO融合算子。完整工作流：NPU迁移→基线验证→算子融合→模型集成→端到端验证→性能分析→归档。触发词：NPU、昇腾、Ascend、华为AI处理器、算子融合、整网集成、replace small ops、模型算子替换（GLM、LLaMA、Qwen、MoE、Attention等）、HuggingFace模型部署、内存预估、auto_map、离线加载等。
---

## 环境约定

所有 `pip` / `npm` 命令必须使用国内源：
- **pip**：`-i https://mirrors.huaweicloud.com/repository/pypi/simple --trusted-host mirrors.huaweicloud.com`
- **npm**：`--registry=https://registry.npmmirror.com`

---

# PyPTO 融合算子整网集成 Skill

将HuggingFace大语言模型迁移到华为Ascend NPU环境，并将PyPTO融合算子替换到整网中替代原始算子实现的完整工作流程。

**强制要求：** 必须使用真实NPU硬件，最终验证推理脚本成功运行。

---

## 工作流程概览

```
阶段零：NPU 迁移与基线建立  →  模型下载 → 脚本生成 → 基线验证 → Git基线 → 代码部署
阶段一：前置准备           →  需求分析 + 环境验证 ★ + 智能推荐
阶段二：理解验证           →  打点采集 ★ + Golden编写 + 场景验证 ⚠️
阶段三：算子开发           →  设计方案 + 实现 + 单算子验证 ★
阶段四：模型集成           →  目录结构 + 适配层 + 调用逻辑 + 缓存处理 ★
阶段五：验证与提交        →  端到端验证（必须） + 性能采集与分析 [可选] + 归档 + 提交
```

> **★ 标注：** 关键步骤，必须完成  
> **⚠️ 标注：** 需特别注意  
> **[可选]：** 后续迭代进行

---

## 关联Skill获取方式

本skill位于 `pypto-gym` 仓库（`cann/pypto-gym`）。

本skill引用的其他skill不在本仓库中，**均位于 pypto 主仓库**：
```
https://gitcode.com/cann/pypto/tree/master/.agents/skills
```

| 被引用Skill | 用途 | 所在仓库 |
|------------|------|---------|
| `pypto-environment-setup` | NPU环境安装 | cann/pypto |
| `pypto-api-explore` | 探索pypto API | cann/pypto |
| `pypto-golden-generate` | Golden生成 | cann/pypto |
| `pypto-op-design` | 设计方案 | cann/pypto |
| `pypto-op-develop` | 算子实现 | cann/pypto |
| `pypto-precision-compare` | 精度对比 | cann/pypto |
| `pypto-precision-debug` | 精度调试 | cann/pypto |
| `pypto-aicore-error-locator` | aicore错误定位 | cann/pypto |
| `pypto-host-stacktrace-analyzer` | 堆栈分析 | cann/pypto |
| `pypto-op-perf-tune` | 性能调优 | cann/pypto |
| `pypto-issue-creator` | 创建Issue | cann/pypto |
| `pypto-pr-creator` | 创建PR | cann/pypto |
| `pypto-intent-understand` | 需求理解 | cann/pypto |

> **获取方法：** 克隆或浏览 `https://gitcode.com/cann/pypto`，skill文件均在 `.agents/skills/` 目录下，每个skill对应一个子目录。

---

## 详细步骤指南

---

### 阶段零：NPU 迁移与基线建立

---

#### 步骤 0：确认模型信息 ★

**必须询问：**
1. HuggingFace模型链接 → 提取 `repo_id`（如：Qwen/Qwen2-7B）和 `model_name`（如：Qwen2-7B）
2. 模型存放目录 `user_model_dir`（默认：/data/models）

**自动检测：**
- `pypto_repo`：当前仓库根目录（通过 pwd 或检测 .git 文件）

**变量定义（后续步骤统一使用）：**
```
model_weight_dir = {user_model_dir}/{model_name}           # 模型权重目录（运行时文件）
script_dir = {model_weight_dir}/scripts                    # 脚本目录
core_dir = {model_weight_dir}/core                         # 代码目录（transformers内置模式）
```

**检测已有模型：**

```bash
ls {model_weight_dir}/*.safetensors {model_weight_dir}/*.bin 2>/dev/null
```

- 存在权重 → 直接使用，跳过步骤 4
- 不存在 → **询问：**「无权重文件。有已下载的目录吗？（给路径则跳过下载，否则自动下载）」

若用户提供目录：`model_weight_dir` / `script_dir` / `core_dir` 同步更新。

---

#### 步骤 1：检查NPU环境与内存预估

**检查NPU状态：**
```bash
npu-smi info
```

**验证标准：** 输出显示NPU设备列表，至少一张卡可用。失败则使用 `pypto-environment-setup` skill。

Ascend910 单卡内存：**64GB HBM**

**预估模型内存占用：**

公式：模型内存(GB) ≈ 参数量(B) × 精度系数 + KV缓存(20%) + 系统开销(2GB)

| 模型参数 | float16 | float32 | int8 | int4 | 单卡64GB(float16) |
|---------|---------|---------|------|------|-------------------|
| 3B | ~7GB | ~14GB | ~3.5GB | ~1.8GB | ✅ 可以 |
| 7B | ~17GB | ~34GB | ~8.5GB | ~4.3GB | ✅ 可以 |
| 13B | ~30GB | ~60GB | ~15GB | ~7.5GB | ✅ 可以 |
| 30B | ~60GB | ~120GB | ~30GB | ~15GB | ⚠️ 勉强，建议量化 |
| 70B+ | ~140GB | ~280GB | ~70GB | ~35GB | ❌ 需多卡或量化 |

**KV缓存影响：** 长序列（如128K上下文）会显著增加KV缓存占用，需额外预留内存。

---

#### 步骤 2：安装依赖（版本匹配是关键）

**torch和torch-npu版本必须完全一致！**

```bash
pip install torch==2.7.1 torch-npu==2.7.1
pip install transformers accelerate sentencepiece protobuf

# 验证安装
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}'); print(f'NPU可用: {torch.npu.is_available()}')"
```

---

#### 步骤 3：项目目录结构

**最终目录结构（所有文件集中在模型权重目录）：**
```
{model_weight_dir}/                  # 默认：/data/models/{model_name}
├── config.json                      # 模型配置（auto_map指向core/下文件）
├── model.safetensors                # 模型权重
├── tokenizer.json                   # tokenizer
├── tokenizer.model                  # tokenizer模型
├── core/                            # 网络结构代码（仅transformers内置模式需要）
│   ├── modeling_xxx.py              # 导入：from transformers.xxx, from .configuration_xxx
│   └── configuration_xxx.py         # 导入：from transformers.xxx
└── scripts/                         # 脚本目录
    ├── ask_{model_name}.py          # 推理脚本（内嵌计时/显存采集）
    ├── bench_{model_name}.sh        # 两段式基准测试脚本（含JSON对比）
    ├── prof_{model_name}.sh         # msprof kernel级采集脚本 [可选]
    ├── sample_inputs.txt            # 示例提示词文件（bench使用）
    └── README.md                    # 迁移说明（必须）
```

**trust_remote_code模式：** core/目录不存在，网络结构代码已包含在HuggingFace下载的文件中。

---

#### 步骤 4：下载模型

> **前置条件：** 步骤 0 已确认 `{model_weight_dir}` 不存在权重文件。

**检测网络结构来源：**

检查仓库 config.json 的 auto_map 字段：
- 若含 auto_map：优先选择完整下载网络结构（trust_remote_code 模式）
- 若不含 auto_map：使用 transformers 内置实现

**创建目录并下载：**
```bash
mkdir -p {model_weight_dir}

export HF_ENDPOINT=https://hf-mirror.com

nohup python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='{repo_id}',
    local_dir='{model_weight_dir}',
    max_workers=10
)
" > {model_weight_dir}/download.log 2>&1 &
```

**检查下载进度：**
```bash
ps aux | grep snapshot_download
du -sh {model_weight_dir}/
ls -la {model_weight_dir}/
```

---

#### 步骤 5：创建ask脚本

**创建目录并生成脚本：**
```bash
mkdir -p {script_dir}

python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/generate_ask_script.py \
    --model-name "{model_name}" \
    --script-dir "{script_dir}" \
    --default-model-dir "{user_model_dir}"
```

**生成的脚本特性：**
- 默认模型路径：`{user_model_dir}/{model_name}`（可通过 `--model-path` 指定）
- 使用 `local_files_only=True` 离线加载，`trust_remote_code=True`
- 参数：`--prompt`（优先级最高）、`--sentence_file`（从文件读取提示词）、`--output_length`（默认100）、`--use_pypto`（PyPTO占位标记）、`--report-file`（JSON性能报告）
- `--prompt` 提供时优先使用，否则 `--sentence_file` 提供时读取文件，均不提供则使用默认 "你好"
- 内嵌 `time.perf_counter()` + `torch.npu.reset_peak_memory_stats()` 计时/显存采集

---

#### 步骤 5.5：生成benchmark脚本

生成两段式基准测试脚本，分别测试baseline和PyPTO模式：

```bash
python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/generate_bench_script.py \
    --model-name "{model_name}" --script-dir "{script_dir}" --model-dir "{model_weight_dir}"

cp {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/sample_inputs.txt {script_dir}/
chmod +x {script_dir}/bench_{model_name}.sh
```

**生成的脚本行为：**
- Phase 1：调用 `ask_{model_name}.py` 不加 `--use_pypto`（baseline），通过 `--report-file` 输出 JSON 指标
- Phase 2：调用 `ask_{model_name}.py` 加 `--use_pypto`（PyPTO占位模式），通过 `--report-file` 输出 JSON 指标
- 两阶段均使用 `sample_inputs.txt` 作为输入，`--output_length 100`
- 脚本末尾自动解析两份 JSON，输出对比表格（推理耗时、吞吐、峰值显存）

---

#### 步骤 5.6：生成msprof采集脚本 [可选]

```bash
python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/generate_prof_script.py \
    --model-name "{model_name}" --script-dir "{script_dir}" --model-dir "{model_weight_dir}"
chmod +x {script_dir}/prof_{model_name}.sh
```

生成的 `prof_{model_name}.sh` 使用 `msprof` 对 baseline 和 PyPTO 各采集一份 kernel trace，自动 diff 算子耗时。

---

#### 步骤 6：验证脚本运行 ★

```bash
python3 {script_dir}/ask_{model_name}.py
```

**验证通过标准：**
- 脚本成功加载模型到 NPU
- 输出显示正在使用 NPU 设备
- 生成回复并输出
- 无错误退出

---

#### 步骤 7：Git管理 — 提交1：基线 ★ [改造前必须]

在代码部署之前，将已验证跑通的基线代码纳入版本控制：

```bash
cd {model_weight_dir}

# 已有 .git 则跳过 init
[ ! -d .git ] && git init

cat > .gitignore << 'EOF'
*.safetensors
*.bin
*.pt
*.pth
tokenizer.json
tokenizer.model
vocab.json
merges.txt
__pycache__/
*.log
.cache/
output/
EOF

git add scripts/ *.json *.md .gitignore
git commit -m "migrate {model_name} to NPU — baseline before pto integration"
```

> 仅提交脚本/配置/文档，排除权重/tokenizer等大文件。
> 后续 PyPTO 产出的代码变更也通过此 git 仓库管理。

---

#### 步骤 8：本地完整代码部署（强制执行）

检测 `{model_weight_dir}/config.json` 的 auto_map 字段判断网络结构来源。

**README.md 必须包含的字段（强制）：**

| 字段 | 说明 |
|------|------|
| HuggingFace | 模型的 repo_id |
| 权重目录 | 模型权重存放的实际路径 |
| 代码来源 | trust_remote_code 或 transformers包 |
| 运行命令 | 执行脚本的具体命令 |

**情况B额外必须包含：**

| 字段 | 说明 |
|------|------|
| transformers版本 | 复制代码时的 transformers 版本号 |
| 代码位置 | 复制后代码的实际路径 |
| 修改内容 | 导入方式修改、auto_map添加等 |

---

**情况A：auto_map 存在（trust_remote_code 模式）**

网络结构已下载到模型目录，无需复制和修改：

1. 保持代码原位置
2. 创建 `{script_dir}/README.md`（包含必须字段，代码来源填"HuggingFace仓库自带"）

---

**情况B：auto_map 不存在（transformers 内置模式）**

从 transformers 包复制实现文件到 `{model_weight_dir}/core/` 目录：

1. **创建 core 目录并复制代码**
   ```bash
   mkdir -p {model_weight_dir}/core

   TRANSFORMERS_PATH=$(python3 -c "import transformers; print(transformers.__path__[0])")
   MODEL_TYPE=$(python3 -c "import json; print(json.load(open('{model_weight_dir}/config.json')).get('model_type',''))")

   cp $TRANSFORMERS_PATH/models/$MODEL_TYPE/modeling_$MODEL_TYPE.py {model_weight_dir}/core/
   cp $TRANSFORMERS_PATH/models/$MODEL_TYPE/configuration_$MODEL_TYPE.py {model_weight_dir}/core/
   ```

2. **修改导入方式**（在 `{model_weight_dir}/core/` 文件中）

   transformers 5.x 的导入格式复杂，需精确处理：

   **推荐方式：使用自动化脚本**
   ```bash
   python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/fix_imports.py {model_weight_dir}/core/modeling_{model_type}.py
   python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/fix_imports.py {model_weight_dir}/core/configuration_{model_type}.py
   ```

   **手动修改规则**（若脚本不可用）：
   - `from ...xxx import yyy` → `from transformers.xxx import yyy`（单行导入）
   - `from ...xxx import (` → `from transformers.xxx import (`（多行导入块开始）
   - `from .configuration_xxx` **保持不变**（同一目录相对导入）

   **注意**：简单的 `sed` 替换可能遗漏多行导入块中的续行，建议优先使用脚本。

3. **添加 auto_map 到 config.json**（指向 core 子目录）
   ```bash
   python3 -c "
   import json
   c = json.load(open('{model_weight_dir}/config.json'))
   mt = c.get('model_type', '')
   arch = c['architectures'][0]
   c['auto_map'] = {
       'AutoConfig': f'core/configuration_{mt}.{mt.capitalize()}Config',
       'AutoModelForCausalLM': f'core/modeling_{mt}.{arch}'
   }
   json.dump(c, open('{model_weight_dir}/config.json', 'w'), indent=2)
   print('已添加 auto_map（指向 core/ 目录）')
   "
   ```

4. 创建 `{script_dir}/README.md`（包含必须字段+情况B额外字段）

---

#### 步骤 9：Git管理 — 提交2：代码部署变更

代码部署完成后，将 core/ 和 config.json 的变更提交：

```bash
cd {model_weight_dir}
git add core/ config.json
git commit -m "code deployment: add core/ with fixed imports and auto_map"
```

---

### 阶段一：前置准备

---

#### 步骤 10：需求分析（智能推荐）

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

#### 步骤 11：前置验证（环境+网络基线）★

**目标：** 确认环境和网络可运行，作为后续工作基础。

**环境验证：**
- 推荐 Skill：`pypto-environment-setup`
- 检查 NPU 状态：`npu-smi info`
- 确认 torch/torch-npu 版本一致

**基线确认：**
- 阶段零的步骤6已验证模型可正常运行
- 若来自本skill的阶段零，直接确认即可
- 否则运行 `python3 {script_dir}/ask_{model_name}.py` 验证

**验证检查点：**
- ✅ NPU 驱动正常
- ✅ 模型可正常加载和推理
- ✅ 输出为自然语言（非乱码）

---

### 阶段二：理解验证 ⚠️

> **为什么需要？** 推测的计算逻辑需 Golden 验证确认。

#### 步骤 12：打点采集真实 Tensor 信息 ★

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

#### 步骤 13：编写 Golden 脚本（场景区分）

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

#### 步骤 14：验证理解正确性（分场景验证）

**目标：** 验证 Golden 与原始实现等价。

---

**场景A：未引用 torch_npu**

**无需验证**：Golden = 原始代码，直接进入步骤16。

---

**场景B：引用 torch_npu ★ 必须验证**

**输出件清单：**

| 输出件 | 文件名 | 存放位置 | 要求 |
|--------|--------|---------|------|
| 测试用例 | `test_cases_golden.json` | `pto_kernels/test/` | 必须：torch_npu 参数信息 |
| 验证脚本 | `test_{op}_golden.py` | `pto_kernels/test/` | 必须：对比 Golden vs torch_npu |

**验证流程：**
1. **构造测试用例**（来自步骤12采集的 shape/dtype）
2. **对比 torch Golden 与 torch_npu**：
   ```python
   output_npu = torch_npu.flash_attention(query, key, value, **params)
   output_golden = xxx_golden(query, key, value, **params)
   assert_allclose(output_npu, output_golden, rtol=1e-3, atol=1e-3)
   print("[PRECISION_PASS] Golden 与 torch_npu 一致")
   ```
3. **一致后替换 Golden 到整网**，验证输出正确
4. **全部通过** → 进入步骤16

---

**验证检查点：**
- ✅ Golden 与 torch_npu 一致（diff < 1e-3）
- ✅ 整网替换后输出正常
- ✅ 无 NaN/Inf

---

#### 步骤 15：决策与迭代

- **通过 ✅**：
  - 场景A：无需验证 → 进入步骤16
  - 场景B：torch Golden 与 torch_npu 一致 + 整网验证通过 → 进入步骤16
- **失败 ❌**：
  - 场景A：不存在（Golden = 原始代码）
  - 场景B：重新理解融合算子语义，修正 Golden

---

### 阶段三：算子开发

#### 步骤 16：设计方案

**目标：** 设计 PyPTO 实现方案（API 映射、Tiling 策略）。

**推荐 Skill：** `pypto-op-design`

**⚠️ 阶段三常见陷阱：**

| 陷阱 | 原因 | 表现 | 预防 |
|------|------|------|------|
| 内置API不支持动态轴 | 内部`cast`拒绝dim=-1 | `FC0000: invalid shape value: -1` | 设计前先查API源码，含`cast`则走手动实现 |
| `set_vec_tile_shapes(x.shape[i])` | 返回值是SymbolicScalar | `F00002: Not concrete value` | 使用concrete常量(e.g. `set_vec_tile_shapes(1, 2048)`) |
| `pypto.mul(x, Element(...))` | `mul`内部二次包装Element | `TypeError: Element(Element)` | 传标量(float/int)，`mul`自动转换 |

---

#### 步骤 17：算子实现

**目标：** 编写 PyPTO 算子代码。

**推荐 Skill：** `pypto-op-develop`

---

#### 步骤 18：单算子验证

**目标：** 验证 PyPTO 实现正确性。

**编译环境：**
- 必须设置 `export PTO_TILE_LIB_CODE_PATH=/path/to/pto-isa`（kernel 编译需要）
- 设置 `export TILE_FWK_DEVICE_ID=<空闲 chip id>`

**关键说明：**
- 步骤 12 采集的真实用例是必须 pass 的基准
- 输出无 NaN/Inf，与 Golden 对齐（diff < 2e-3）

**推荐 Skill：** `pypto-precision-compare`

---

### 阶段四：模型集成

#### 步骤 19：调整目录结构

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

#### 步骤 20：配置适配层

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

#### 步骤 21：修改模型调用逻辑 + sys.modules注入 ★

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

#### 步骤 22：验证与排查

**目标：** 端到端精度验证 + 问题排查。

**精度验证：**
- 运行整网推理，确认输出正常
- 与原始实现对比

**问题排查：**
- 参考 Skill：`pypto-aicore-error-locator`、`pypto-host-stacktrace-analyzer`、`pypto-precision-debug`

---

#### 步骤 23：性能采集与分析 [可选]

**目标：** PyPTO 融合完成后，量化算子替换带来的整网收益。

**前置：** 步骤 22 端到端验证通过

> **优先级：** 精度验证（必须） > 性能对比（可选）

##### 方法一：整网计时+显存对比

```bash
bash {model_weight_dir}/scripts/bench_{model_name}.sh
```

- ask 推理脚本内嵌 `time.perf_counter()` + `torch.npu.reset_peak_memory_stats()` / `torch.npu.max_memory_allocated()`
- 每次运行通过 `--report-file` 输出 JSON 指标报告
- bench 脚本自动解析两份 JSON，生成对比表格：

| 指标 | Baseline | PyPTO | Diff |
|------|----------|-------|------|
| 推理耗时 (s) | | | |
| 吞吐 (tokens/s) | | | |
| 峰值显存 (MB) | | | |

##### 方法二：msprof kernel 级采集

```bash
bash {model_weight_dir}/scripts/prof_{model_name}.sh
```

- 分别对 baseline 和 PyPTO 模式执行 `msprof --application=...`
- 输出文件：
  - `device_*/summary/op_statistic_*.csv` — 算子耗时汇总
  - `device_*/timeline/*.json` — Chrome trace 时间线
  - `device_*/aicore_metrics_*.csv` — AI Core 利用率
- 脚本自动 diff Top 15 算子耗时差异

##### 记录结果

将 **对比表格** 追加到 `{model_weight_dir}/scripts/README.md` 末尾，跟随 Git 提交保留。

##### 进一步算子级微调

**推荐 Skill：** `pypto-op-perf-tune`（Tile 配置、合图策略、向量化等微观优化）

---

#### 步骤 24：归档到 pypto-gym 仓库 [可选]

**前置：** 步骤 22 端到端验证通过  
**触发：** 询问用户是否归档

##### 变量定义
```
{model_dir}       = 部署模型权重目录
{pypto_gym_repo}  = pypto-gym 仓库根目录
{model_name}      = 模型名（pypto-gym 惯例：小写+下划线，如 qwen3_1_7b）
```

##### 文件映射

| 来源 (`{model_dir}/`) | 目标 (`{pypto_gym_repo}/`) | 操作 |
|---|---|---|
| `scripts/*` | `modeling/transformers/{model_name}/` | 新建 |
| `core/` 全部 .py | `src/pypto_gym/transformers/{model_name}/` | 覆盖 |
| `pto_kernels/` (除去 golden + test/) | `src/pypto_gym/ops/pypto_tile/{model_name}/` | 顶层 `__init__.py` 覆盖，其余新建 |
| `pto_kernels/*/golden*.py` + `test/` | `tests/ops/{model_name}/` | 新建 |

##### test 文件 import 改造

`test_rms_norm.py` 移至 `tests/ops/{model_name}/` 后，impl 不在同级目录，通过 `sys.path` 跨目录引用：

```python
from pathlib import Path; import sys
_IMPL = Path(__file__).resolve().parents[3] / "src/pypto_gym/ops/pypto_tile/{model_name}"
sys.path.insert(0, str(_IMPL))
from rms_norm_golden import rms_norm_golden      # 同级
from rms_norm.rms_norm_impl import rms_norm_impl  # sys.path 中找到
```

归档完成后运行 test_xxx.py 验证归档正确性：
```bash
python3 tests/ops/{model_name}/test_rms_norm.py
```
全部用例通过（`[PRECISION_PASS]` + `All tests passed!`）即确认归档成功。

##### 提交 PR [可选]

归档完成后询问用户是否需要提交 PR 将变更合入 pypto-gym 仓库。若需要，使用 `pypto-pr-creator` skill 创建 PR。

---

#### 步骤 25：还原重建指南 [可选]

将归档文件 + 下载的权重重建为可运行环境。

```bash
MODEL_DIR=/path/to/weights

mkdir -p $MODEL_DIR/core $MODEL_DIR/pto_kernels $MODEL_DIR/scripts

# 1. 复制归档文件
cp -r {pypto_gym_repo}/src/pypto_gym/transformers/{model_name}/*  $MODEL_DIR/core/
cp -r {pypto_gym_repo}/src/pypto_gym/ops/pypto_tile/{model_name}/* $MODEL_DIR/pto_kernels/
cp    {pypto_gym_repo}/modeling/transformers/{model_name}/* $MODEL_DIR/scripts/

# 将 golden 文件复制回对应算子目录（wrapper fallback 需要）
for f in {pypto_gym_repo}/tests/ops/{model_name}/*_golden.py; do
    op=$(basename "$f" | sed 's/_golden.py//')
    cp "$f" $MODEL_DIR/pto_kernels/$op/ 2>/dev/null || true
done

# 2. 参数化修复 auto_map（从 config.json 动态提取 model_type 和 architectures）
python3 -c "
import json, sys
cfg = json.load(open(sys.argv[1]))
mt = cfg['model_type']; arch = cfg['architectures'][0]
cfg['auto_map'] = {
    'AutoConfig': f'core/configuration_{mt}.{mt.capitalize()}Config',
    'AutoModelForCausalLM': f'core/modeling_{mt}.{arch}'
}
json.dump(cfg, open(sys.argv[1], 'w'), indent=2)
" $MODEL_DIR/config.json

# 3. 运行验证
python3 $MODEL_DIR/scripts/ask_{model_name}.py --prompt "你好"
python3 $MODEL_DIR/scripts/ask_{model_name}.py --prompt "你好" --use_pypto
```

---

#### 步骤 26：Git管理 — 提交3：PTO 融合变更

PTO 融合成功后，将算子库和修改后的代码提交：

```bash
cd {model_weight_dir}
git add pto_kernels/ core/ scripts/ config.json
git commit -m "pto integration: add fused kernel for {model_name}"
```

---

#### 步骤 27：提交与文档

**目标：** 创建 Issue 和 PR。

**操作：**
1. 创建 Issue 跟踪变更
2. 提交 PR（含修改说明）

**推荐 Skill：** `pypto-issue-creator`、`pypto-pr-creator`

---

## 常见问题解决

### Q1: transformers报错 "PyTorch >= 2.4 is required"
升级torch和torch-npu到2.7.1

### Q2: torch_npu报错 "undefined symbol"
确保torch和torch-npu版本完全一致

### Q3: 网络问题或模型加载慢
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
使用 `local_files_only=True` 从本地加载

### Q4: 导入错误（ImportError/FileNotFoundError）
检查导入语句：transformers模块用绝对导入，本地configuration用相对导入

## 错误恢复指南

### 下载中断恢复

若模型下载中断（网络问题、进程被杀），可重新执行下载：
```bash
# 检查已下载文件大小
du -sh {model_weight_dir}/

# 重新下载（resume模式会跳过已下载文件）
export HF_ENDPOINT=https://hf-mirror.com
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='{repo_id}', local_dir='{model_weight_dir}', max_workers=4)
"
```

### 导入修改失败恢复

若导入修改导致语法错误，可从备份恢复：
```bash
# 查看备份文件
ls -la {model_weight_dir}/core/*.bak

# 从备份恢复
cp {model_weight_dir}/core/modeling_{model_type}.py.bak {model_weight_dir}/core/modeling_{model_type}.py
cp {model_weight_dir}/core/configuration_{model_type}.py.bak {model_weight_dir}/core/configuration_{model_type}.py

# 重新使用脚本修复
python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/fix_imports.py {model_weight_dir}/core/modeling_{model_type}.py
```

---

## 相关资源

### 参考模板
- **测试集模板**：`references/test-cases-template.md`

### 相关 Skill
- `pypto-environment-setup`：环境安装
- `pypto-api-explore`：探索pypto API
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

**Skill 版本：** v3.0  
**最后更新：** 2026-04-30  
**维护者：** PyPTO Team  
**更新说明：** 合并 migrate-huggingface-to-npu 和 pypto-fused-op-integration 为统一 skill。阶段零为 NPU 迁移，阶段一~五为融合集成。
