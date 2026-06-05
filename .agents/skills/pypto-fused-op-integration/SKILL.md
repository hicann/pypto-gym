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
阶段三路线选择 🔀           →  Benchmark 自动化 或 经典手动开发（用户选择）★
├─ 阶段三-A：Benchmark 自动化  →  Benchmark Case生成 + Orchestrator Agent 自动开发 ★
└─ 阶段三-B：经典手动开发     →  design → develop → precision-compare → perf-tune
阶段四：模型集成           →  目录结构 + 适配层 + 调用逻辑 + 缓存处理 ★
阶段五：验证与提交        →  端到端验证（必须） + 性能采集与分析 [可选] + 归档 + 提交
```

> **★ 标注：** 关键步骤，必须完成  
> **⚠️ 标注：** 需特别注意  
> **🔀 标注：** 用户选择点，必须询问  
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
| `pypto-testcase-to-benchmark` | Golden → Benchmark Case 转换 | cann/pypto-gym |
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

## 参考文档索引

以下内容已移至 `references/` 子目录，需要时查阅：

| 文件 | 内容 | 使用场景 |
|------|------|---------|
| `references/memory_estimation.md` | NPU内存预估公式+各规模模型估算表 | 步骤1判断模型是否能在单卡运行 |
| `references/directory_structure.md` | 阶段零/四/归档目录结构 + README必须字段 + 变量定义 | 各步骤创建目录、写README时 |
| `references/troubleshooting.md` | 常见问题FAQ + 下载中断/导入修改失败恢复 | 遇到报错时查阅 |

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
- 变量定义见 `references/directory_structure.md`

**检测已有模型：**
```bash
ls {model_weight_dir}/*.safetensors {model_weight_dir}/*.bin 2>/dev/null
```

- 存在权重 → 直接使用，跳过步骤 4
- 不存在 → **询问：**「无权重文件。有已下载的目录吗？（给路径则跳过下载，否则自动下载）」

**⚠️ 副本/克隆目录需验证tokenizer文件：**

模型目录权重文件存在不等于tokenizer可用。从baseline复制或symlink模型时，必须同步检查：
```bash
ls {model_weight_dir}/tokenizer.json {model_weight_dir}/vocab.json {model_weight_dir}/merges.txt 2>/dev/null
```
缺失会导致 `tokenizer(prompt)` 返回 0 tokens，后续模型因空输入崩溃。

---

#### 步骤 1：检查NPU环境与内存预估

**检查NPU状态：**
```bash
npu-smi info
```

**验证标准：** 输出显示NPU设备列表，至少一张卡可用。失败则使用 `pypto-environment-setup` skill。

**内存预估：** 查阅 `references/memory_estimation.md` 判断模型是否能在单卡运行。

---

#### 步骤 2：安装依赖

**torch和torch-npu版本必须完全一致！**

```bash
pip install torch==2.7.1 torch-npu==2.7.1
pip install transformers accelerate sentencepiece protobuf

# 验证安装
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}'); print(f'NPU可用: {torch.npu.is_available()}')"
```

---

#### 步骤 3：下载模型

> **前置条件：** 步骤 0 已确认 `{model_weight_dir}` 不存在权重文件。

**检测网络结构来源：**
检查仓库 config.json 的 auto_map 字段：
- 含 auto_map → trust_remote_code 模式（优先）
- 不含 auto_map → transformers 内置实现

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
```

---

#### 步骤 4：生成脚本（ask + bench + prof）

> 合并原步骤 5/5.5/5.6，一键生成推理脚本、基准测试脚本和 msprof 采集脚本。

**生成推理脚本：**
```bash
mkdir -p {script_dir}

python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/generate_ask_script.py \
    --model-name "{model_name}" \
    --script-dir "{script_dir}" \
    --default-model-dir "{user_model_dir}"
```

生成的 `ask_{model_name}.py` 支持：`--prompt`、`--sentence_file`、`--output-length`（默认100）、`--use_pypto`、`--per-step-timing`、`--report-file`、`--profile-dir`（torch_npu profiler trace 输出目录）

**生成基准测试脚本：**
```bash
python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/generate_bench_script.py \
    --model-name "{model_name}" --script-dir "{script_dir}" --model-dir "{model_weight_dir}"

cp {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/sample_inputs.txt {script_dir}/
chmod +x {script_dir}/bench_{model_name}.sh
```

**生成msprof采集脚本 [可选]：**
```bash
python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/generate_prof_script.py \
    --model-name "{model_name}" --script-dir "{script_dir}" --model-dir "{model_weight_dir}"
chmod +x {script_dir}/prof_{model_name}.sh
```

---

#### 步骤 5：验证脚本运行 ★

```bash
python3 {script_dir}/ask_{model_name}.py
```

**验证通过标准：**
- 脚本成功加载模型到 NPU
- 输出显示正在使用 NPU 设备
- 生成回复并输出，无错误退出

---

#### 步骤 6：Git基线提交 ★

> 在代码部署之前，将已验证跑通的基线代码纳入版本控制。

```bash
cd {model_weight_dir}

[ ! -d .git ] && git init

cp {pypto_repo}/.agents/skills/pypto-fused-op-integration/templates/gitignore {model_weight_dir}/.gitignore

git add scripts/ *.json *.md .gitignore
git commit -m "migrate {model_name} to NPU — baseline before pto integration"
```

> 仅提交脚本/配置/文档，排除权重/tokenizer等大文件。

---

#### 步骤 7：代码部署 ★

检测 `{model_weight_dir}/config.json` 的 auto_map 字段判断网络结构来源。

**README必须字段：** 见 `references/directory_structure.md`

---

**情况A：auto_map 存在（trust_remote_code 模式）**

网络结构已下载到模型目录，无需复制和修改：
1. 保持代码原位置
2. 创建 `{script_dir}/README.md`（代码来源填"HuggingFace仓库自带"）

---

**情况B：auto_map 不存在（transformers 内置模式）**

从 transformers 包复制实现文件到 `{model_weight_dir}/core/` 目录：

1. **复制代码：**
   ```bash
   mkdir -p {model_weight_dir}/core

   TRANSFORMERS_PATH=$(python3 -c "import transformers; print(transformers.__path__[0])")
   MODEL_TYPE=$(python3 -c "import json; print(json.load(open('{model_weight_dir}/config.json')).get('model_type',''))")

   cp $TRANSFORMERS_PATH/models/$MODEL_TYPE/modeling_$MODEL_TYPE.py {model_weight_dir}/core/
   cp $TRANSFORMERS_PATH/models/$MODEL_TYPE/configuration_$MODEL_TYPE.py {model_weight_dir}/core/
   ```

2. **修改导入方式：** 使用自动化脚本（推荐）或手动修改
   ```bash
   python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/fix_imports.py {model_weight_dir}/core/modeling_{model_type}.py
   python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/fix_imports.py {model_weight_dir}/core/configuration_{model_type}.py
   ```

   手动修改规则：`from ...xxx import yyy` → `from transformers.xxx import yyy`，`from .configuration_xxx` 保持不变。

3. **添加 auto_map 到 config.json：**
   ```bash
   python3 -c "
   import json
   c = json.load(open('{model_weight_dir}/config.json'))
   mt = c.get('model_type', ''); arch = c['architectures'][0]
   c['auto_map'] = {
       'AutoConfig': f'core/configuration_{mt}.{mt.capitalize()}Config',
       'AutoModelForCausalLM': f'core/modeling_{mt}.{arch}'
   }
   json.dump(c, open('{model_weight_dir}/config.json', 'w'), indent=2)
   "
   ```

4. 创建 `{script_dir}/README.md`（包含必须字段+情况B额外字段）

---

#### 步骤 8：代码部署Git提交

```bash
cd {model_weight_dir}
git add core/ config.json
git commit -m "code deployment: add core/ with fixed imports and auto_map"
```

---

### 阶段一：前置准备

---

#### 步骤 9：需求分析（智能推荐）

**目标：** 分析网络可融合部分，推荐给用户确认。

**操作：**
1. 阅读模型代码，识别算子组合模式（Attention、MoE、FFN等）
2. 搜索 `models/` 目录下的实现案例，匹配 pypto 现有算子
3. 向用户推荐候选融合点（示例格式）：
   ```
   发现可融合算子组合：
   | 位置 | 原始实现 | pypto算子 | 收益 |
   | Attention层 | Q/K/V投影+Softmax+Output | pypto.flash_attention | 减少3次matmul |
   | FFN层 | Gate+Up+SwiGLU+Down | pypto.swiglu_ffn | 减少中间存储 |
   请确认目标算子。
   ```
4. 无合适推荐 → 直接询问用户融合位置和目标

**推荐 Skill：** `pypto-api-explore`

---

#### 步骤 10：前置验证（环境+网络基线）★

**环境验证：**
- 推荐 Skill：`pypto-environment-setup`
- 检查 NPU 状态：`npu-smi info`
- 确认 torch/torch-npu 版本一致

**基线确认：**
- 阶段零的步骤5已验证模型可正常运行 → 直接确认
- 否则运行 `python3 {script_dir}/ask_{model_name}.py` 验证

**验证检查点：** ✅ NPU驱动正常 ✅ 模型可正常加载和推理 ✅ 输出为自然语言（非乱码）

**CANN ops 内核包验证（⚠️ 易遗漏）：**

`npu-smi info` 只检查驱动，不检查算子库。需额外验证 ops 可用性：
```bash
python3 -c "
import torch, torch_npu; torch.npu.set_device(0)
x = torch.randn(2,2, dtype=torch.float16).to('npu:0')
y = x.to(torch.float32)           # aclnnCast 是否可用
torch.equal(x.cpu(), x.cpu())      # aclnnEqual（模型加载 tie_weights 触发）
z = x * 2.0                        # aclnnMul（基础计算）
print('CANN ops OK')
"
```
任一 op 报 `EZ9999: Op XXX does not has any binary` → 安装对应 Ascend-cann-ops-kernel 包或降级 CANN 版本。

---

### 阶段二：理解验证 ⚠️

> **为什么需要？** 推测的计算逻辑需 Golden 验证确认。

#### 步骤 11：打点采集真实 Tensor 信息 ★

**目标：** 从原始网络采集真实 shape/dtype，构造必须 pass 的测试用例。

**操作：**
1. 定位并插入 print（采集所有外部输入的 shape/dtype）
2. 运行原始网络采集
3. 采集完成后删除打印，恢复代码原状
4. 创建 test_cases.json

**注意事项：**
- ⚠️ **不要限制打印次数**：禁止使用计数器限制，否则会漏掉不同场景
- 应采集所有调用场景，覆盖不同 shape/dtype 组合

**格式参考：** `pypto-op-develop/templates/test_cases-template.json` 和 `references/test-cases-template.md`

**输出物：** test_cases.json → 存放 `models/{model_name}/pto_kernels/xxx/test/test_cases.json`

---

#### 步骤 12：编写 Golden 脚本（场景区分）

**目标：** 编写 PyTorch 参考实现，精度对比基准。

**场景判断：** 检查**被替换逻辑**是否使用 torch_npu 融合算子（只看逻辑本身，不看文件import）。

| 场景 | 判断条件 | 策略 | 输出件 |
|------|---------|------|--------|
| **场景A** | 只使用基础算子 | 直接复制原始代码，无需理解验证 | `xxx_golden.py` |
| **场景B** | 使用融合算子 | 用 torch 重写等价实现，必须验证 | `xxx_golden.py` + `test_xxx_golden_correctness.py` |

**通用要点：** 纯 PyTorch，禁止引入 pypto/torch_npu。导出 `{op}_golden()` 函数，独立 `{op}_golden.py` 文件。

**⚠️ Golden 必须逐行匹配原始代码的操作顺序**，不仅是数学等价。例如 `gamma * x_norm.to(input_dtype)`（FP16 乘）与 `(gamma.to(FP32) * x_norm).to(input_dtype)`（FP32 乘再降精度）会差 ~1e-4，导致 `max_diff≠0` 验证失败。

**推荐 Skill：** `pypto-golden-generate`

---

#### 步骤 13：验证理解正确性（分场景验证）

**场景A：未引用 torch_npu → 无需验证**，Golden = 原始代码，直接进入步骤15。

**场景B：引用 torch_npu ★ 必须验证**

验证流程：
1. 构造测试用例（来自步骤11采集的 shape/dtype）
2. 对比 torch Golden 与 torch_npu：`assert_allclose(output_npu, output_golden, rtol=1e-3, atol=1e-3)` → `[PRECISION_PASS]`
3. 一致后替换 Golden 到整网，验证输出正确

**验证检查点：** ✅ Golden与torch_npu一致(diff < 1e-3) ✅ 整网替换后输出正常 ✅ 无NaN/Inf

---

#### 步骤 14：决策与迭代

- **通过 ✅**：场景A → 步骤14.5；场景B → torch Golden与torch_npu一致 + 整网验证通过 → 步骤14.5
- **失败 ❌**：场景B → 重新理解融合算子语义，修正 Golden

---

#### 步骤 14.5：选择算子开发路线 🔀

> ⚠️⚠️⚠️ **强制阻塞点 — 禁止跳过！**
> 
> 此步骤是 **🔀 用户选择点**，AI **必须在此处停下来询问用户，等待用户明确回复"A"或"B"后才能继续**。不允许 AI 自行猜测、默认假设、或跳过此步直接进入步骤15/15B。
> 
> **违规判断：** 如果 AI 在 Golden 验证完成后没有展示路线选择询问、没有等待用户回复就开始了步骤15（生成 Benchmark Case）或步骤15B（设计方案），即视为违规跳过。

**询问模板（严格按此格式，不可省略表格）：**

```
阶段二（理解验证）已完成。现在进入阶段三（算子开发），有两种开发方式：

| 路线 | 方式 | 特点 |
|------|------|------|
| **A：Benchmark 自动化** | 生成 Benchmark Case → Orchestrator Agent 自动编排 | 全自动 7 阶段工作流（planner→mathematician→architect→designer→coder→verifier→optimizer），适合标准算子 |
| **B：经典手动开发** | design → develop → precision-compare → perf-tune | 逐步手动精细化控制，适合复杂/定制算子 |

请选择 A 或 B。
```

- **用户选 A** → 进入 [阶段三-A](#阶段三-abenchmark-自动化)（步骤15-19）
- **用户选 B** → 进入 [阶段三-B](#阶段三-b经典手动开发)（步骤15B-19B）
- **用户未回复 / 模糊回复** → 重新展示询问模板，继续等待明确选择

---

### 阶段三-A：Benchmark 自动化

> 阶段三-A 走**生成 benchmark case → 调用 Orchestrator Agent** 路线。Orchestrator 是 PyPTO 的 7-Agent 团队（planner→mathematician→architect→designer→coder→verifier→optimizer），由 `python -m benchmark run` 驱动。

#### 步骤 15：生成 Benchmark Case ★

**目标：** 将阶段二的 golden 函数封装为 KernelBench 格式。

**推荐 Skill：** `pypto-testcase-to-benchmark`

**操作流程：**

1. 分析 golden 函数签名和实现，提取输入/输出 shape/dtype、权重tensor、控制参数
2. 按 KernelBench 格式生成 case 文件，放置在 `benchmark/KernelBench/pto_case/<N>_<OpName>.py`

   **强制规则：**
   - 权重必须用 `nn.Parameter()` 注册
   - `forward()` 返回结果（非 in-place）
   - 输入 tensor 进行数值缩放（`/ math.sqrt(M)`）避免 BF16 溢出
   - `get_inputs()` 只返回 tensor，控制参数放 `__init__`
   - 多输出时 `forward()` 返回 `tuple[torch.Tensor, ...]`

3. **Golden 功能覆盖验证（强制）：** 用 3 组不同尺寸输入，1:1 对比 benchmark golden 与原始 golden，确保 max_diff=0

4. **NPU 预检（强制）：**
   ```bash
   python .agents/skills/pypto-testcase-to-benchmark/scripts/validate_case_npu.py \
       benchmark/KernelBench/pto_case/{N}_{OpName}.py --device 0
   ```

**产物：** `benchmark/KernelBench/pto_case/<N>_<OpName>.py`

---

#### 步骤 16：运行 Orchestrator 生成算子代码 ★

**推荐：通过 benchmark 框架运行**

生成 YAML 配置：
```yaml
# /tmp/pypto_{op}_case.yaml
bench_dir: ""
cases: "pto_case={N}_{OpName}"
devices: [0]
concurrency: 1
output:
  root_dir: "{pypto_gym_repo}/benchmark_runs/{op}_case"
pypto:
  timeout_sec: 10800
  pref_round: 3
  skip_pypto_gen: false
verifier:
  mode: "performance"
  verifier_mode: "direct"
  verify_rtol: 1.0e-2
  verify_atol: 2.5e-2
```

```bash
python -m benchmark run --config /tmp/pypto_{op}_case.yaml
```

> **关键：不要加 `--foreground`！** 默认模式会 fork 到后台运行，终端断开不影响。`--foreground` 仅用于本地调试。

**Orchestrator 流程：** planner→SPEC.md → mathematician→golden → architect→DESIGN → designer→interfaces → coder↔verifier↔debugger→impl+test → optimizer→性能调优

**监控方式：**
```bash
# 重新连接 monitor TUI
python -m benchmark monitor {pypto_gym_repo}/benchmark_runs/{op}_case/state

# 实时查看 agent 输出
tail -f {pypto_gym_repo}/benchmark_runs/{op}_case/report/pto_case/{op}/pypto_run.log

# 运行日志
tail -f {pypto_gym_repo}/benchmark_runs/{op}_case/logs/benchmark.out
```

**结果判断：**
- **成功：** `overall_status = "success"` 且产物齐全
- **超时但产物齐全：** `pypto_status = "timeout"` 但 artifacts 齐全 → 产品可用
- **失败（编译/实现错误）：** 检查 pypto_run.log，必要时重试

---

#### 步骤 17：收集产物到算子目录

产物来源：`{pypto_gym_repo}/benchmark/.cache/pypto/custom/pto_case/{op}/`

```bash
mkdir -p {pto_kernels_dir}/{op}/test

cp {pypto_repo}/custom/pto_case/{op}/{op}_impl.py       {pto_kernels_dir}/{op}/
cp {pypto_repo}/custom/pto_case/{op}/{op}_golden.py     {pto_kernels_dir}/{op}/
cp {pypto_repo}/custom/pto_case/{op}/{op}_pypto_impl.py {pto_kernels_dir}/{op}/
cp {pypto_repo}/custom/pto_case/{op}/test_{op}.py       {pto_kernels_dir}/{op}/test/
```

| 文件 | 用途 |
|------|------|
| `{op}_impl.py` | PyPTO kernel 实现 → 步骤22 |
| `{op}_golden.py` | PyTorch 参考实现 |
| `{op}_pypto_impl.py` | ModelNew 桥接类 → 步骤22（可选） |
| `test_{op}.py` | 单算子精度测试 → 步骤18 |

---

#### 步骤 18：单算子测试 ★

```bash
export PTO_TILE_LIB_CODE_PATH=/workspace/project/pto-isa
export TILE_FWK_DEVICE_ID=0
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 {pto_kernels_dir}/{op}/test/test_{op}.py
```

**验证通过标准：** `[PRECISION_PASS]`、无NaN/Inf、golden与impl差异 < 2e-3

---

#### 步骤 19：aclgraph 支持（如需要）

**必须先询问用户：** "是否需要 aclgraph 支持？"

若需要，在 `{op}_impl.py` 或独立 bridge 文件中补充：
1. **torch library 注册：** `pyptolib = torch.library.Library("pypto", "FRAGMENT")` → `pyptolib.define(f"{op}(Tensor ...) -> (Tensor ...)")`
2. **torch infershape：** `@torch.library.impl(pyptolib, f"{op}", "Meta")` → 返回 `torch.empty(...)`
3. **NPU kernel 调度：** `@torch.library.impl(pyptolib, f"{op}", "NPU")` → 调 `{op}_wrapper()`

---

### 阶段三-B：经典手动开发

> 阶段三-B 走**逐步手动精细化控制**路线（design → develop → precision-compare → perf-tune），适合复杂/定制算子或需要深度调试的场景。

#### 步骤 15B：生成设计方案 ★

**推荐 Skill：** `pypto-op-design`

**操作：** 基于阶段二采集的 test_cases.json 和 golden，编写算子设计文档，明确：
- 输入/输出 tensor 的 shape 和 dtype
- PyPTO API 选型（cast、mul、sum、add、rsqrt 等）
- 动态轴处理策略（pypto.loop / pypto.loop_unroll）
- TileShape 配置
- 精度路由（FP32 中间计算、BF16 输入输出）

**⚠️ Kernel dtype 策略（关键决策）：**

模型原生 dtype（HuggingFace 加载后的 `torch_dtype`）决定 PyPTO kernel 的 DT_* 标注。**kernel 标注 dtype 必须与模型原生 dtype 一致**，否则 wrapper 需做 expensive 的 NPU-side cast（且 CANN 环境可能不支持某些 dtype cast）：

| 模型加载 dtype | PyPTO kernel DT_标注 | 说明 |
|---|---|---|
| `float16` | `pypto.DT_FP16` | 最常见，CANN 兼容性最好 |
| `bfloat16` | `pypto.DT_BF16` | 精度更高但需 CANN bf16 ops 支持 |
| `float32` | `pypto.DT_FP32` | 仅推理用 FP32 时 |

**决策规则**：优先匹配模型 dtype，不要假设 BF16。若 CANN 不支持 BF16→FP16 cast（常见于 ops 内核包不完整的环境），kernel 直接用 FP16。

---

#### 步骤 16B：实现 PyPTO Kernel ★

**推荐 Skill：** `pypto-op-develop`

**操作：**

1. 为每类融合算子独立编写 `{op}_impl.py`：
   - 使用 `@pypto.frontend.jit` 装饰 JIT 编译
   - `@allow_in_graph` 修饰 wrapper 函数
   - wrapper 内完成 torch ↔ pypto tensor 转换

2. **Golden 对比验证：** `assert_allclose(output_impl, output_golden, rtol=1e-3, atol=1e-3)` → `[PRECISION_PASS]`

**输出物：**
- `{op}_impl.py` — PyPTO kernel 实现
- `test_{op}.py` — 单算子精度测试
- 测试须覆盖 test_cases.json 的全部场景

---

#### 步骤 17B：精度调试（如需要）★

**推荐 Skill：** `pypto-precision-compare`、`pypto-precision-debug`

当 golden 与 impl 的 diff 超过容差时使用。

```bash
python3 {pto_kernels_dir}/{op}/test_{op}.py
```

**差异等级：**

| 差异 | 处理 |
|------|------|
| < 1e-3 | ✅ PASS，无需调试 |
| 1e-3 ~ 1e-2 | ⚠️ 轻度不匹配，检查数值稳定性（eps、中间dtype、tile对齐） |
| > 1e-2 | ❌ 严重不匹配，使用 pypto-precision-debug 逐层定位 |

**常见问题：**
- FP32 ↔ BF16 转换丢失精度 → 中间计算全程保持 FP32
- TileShape 对齐不当 → 检查 tail axis 32B 对齐
- 数学等价但实现不同 → 对齐 golden 的计算顺序

---

#### 步骤 18B：AI Core 错误定位（如需要）★

**推荐 Skill：** `pypto-aicore-error-locator`、`pypto-host-stacktrace-analyzer`

编译/运行时出现 AI Core Error 时使用。

---

#### 步骤 19B：性能调优 [可选]

**推荐 Skill：** `pypto-op-perf-tune`

通过调整 Tile 配置、合图策略、向量化等手段优化性能。包括：
- TileShape 优化（block 大小调优）
- DM 搬运与计算流水线优化
- 多核并行调度

---

### 阶段四：模型集成

#### 步骤 20：创建算子库目录

创建 pto_kernels 目录结构。目录结构和命名规则见 `references/directory_structure.md`。

---

#### 步骤 21：配置适配层

**关键要点：**
- 开关设计：按算子粒度 `USE_PTO_{OP}`，便于渐进式验证
- 函数命名：与原始算子同名
- 文档注释：包含目标文件、目标类、替换代码片段
- `allow_in_graph`：适配层函数 `{op}_pto` 调用 `@allow_in_graph` 修饰的 `{op}_wrapper`，禁止越级调 JIT kernel

**参考案例：** https://gitcode.com/songle1/glm-net/blob/main/glm_pto_kernels/__init__.py

---

#### 步骤 22：修改模型调用逻辑 + sys.modules注入 ★

**推荐方案：sys.modules注入**

**原理：** Python的 `sys.modules` 是全局模块注册表。脚本预导入算子库→注入→modeling自动获取，无缓存依赖。

**实施步骤：**

**步骤A：脚本注入（在transformers导入前）**
```python
parser.add_argument("--use-pto", action="store_true", help="启用PyPTO算子")

if args.use_pto:
    sys.path.insert(0, args.model_path)
    import {model}_pto_kernels
    sys.modules["{model}_pto_kernels"] = module
    module.USE_PTO_{OP} = True

from transformers import AutoModelForCausalLM   # 之后加载模型
```

**步骤B：modeling获取**
```python
pto_kernels = sys.modules.get("{model}_pto_kernels")

def forward(self, hidden_states):
    if pto_kernels is not None and pto_kernels.USE_PTO_{OP}:
        return pto_kernels.{op}(self, hidden_states)
    # 原始torch实现（fallback）
    ...
```

**关键要点：** 注入位置在 transformers 导入前，必须保留原始 torch 实现作为 fallback。

**⚠️ PTO 注入时序规则（易错）：**

PTO 注入（`USE_PTO_XXX=True`）必须发生在模型加载到 NPU **之后**：
```
# ✅ 正确
model = AutoModelForCausalLM.from_pretrained(...)
model.to('npu')
_pto.USE_PTO_RMS_NORM = True       # ← 注入在 NPU 就绪后

# ❌ 错误（JIT kernel 收到 CPU tensor）
_pto.USE_PTO_RMS_NORM = True       # ← 太早！
model = AutoModelForCausalLM.from_pretrained(...)
```
建模内路由已加 `device.type == "npu"` 守卫，但依赖同步问题仍应按此顺序。

---

#### 步骤 23：aclgraph适配 ★

**目标：** 使用 torch.compile 将入口 model 编译成 aclgraph。

**步骤A：使能开关**
```python
parser.add_argument("--use-acl-graph", action="store_true")

if args.use_acl_graph:
    module.USE_ACL_GRAPH = True
```

**步骤B：图编译配置**
```python
if pto_kernels is not None and pto_kernels.USE_ACL_GRAPH:
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    compiler_config = CompilerConfig()
    compiler_config.experimental_config.frozen_parameter = True
    compiler_config.experimental_config.tiling_schedule_optimize = True
    npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
    self.model = torch.compile(self.model, dynamic=True, fullgraph=True, backend=npu_backend)
```

---

### 阶段五：验证与提交

#### 步骤 24：端到端验证 ★

**目标：** 整网推理验证 + 问题排查。

- 运行整网推理，确认输出正常
- 与原始实现对比

**⚠️ 单算子替换性能预期：**

当只替换单个独立算子（如 RMSNorm）而未做前后融合时，**PTO 模式通常比基线慢 2-3x**。原因：(1) JIT kernel 首次编译；（2）kernel launch + dtype cast 开销 > 原 CANN 融合 kernel；（3）单算子无法抵消调用开销。

**这属于正常现象**，不是 bug。RMSNorm 的收益来自与相邻算子的融合（如 pre-attn RMSNorm + QKV projection），单算子替换仅验证**路由逻辑正确性和精度一致性**。真正的性能提升见后续 round：融合 pre-attn、post-attn 等复合算子。

**问题排查 Skill：** `pypto-aicore-error-locator`、`pypto-host-stacktrace-analyzer`、`pypto-precision-debug`

---

#### 步骤 25：性能采集与分析 [可选]

> **优先级：** 精度验证（必须） > 性能对比（可选）

**方法一：整网计时+显存对比**
```bash
bash {model_weight_dir}/scripts/bench_{model_name}.sh
```
bench脚本自动解析两份 JSON，生成对比表格（推理耗时、吞吐、峰值显存）。

**方法二：msprof kernel级采集**
```bash
bash {model_weight_dir}/scripts/prof_{model_name}.sh
```
输出：op_statistic CSV + Chrome trace + AI Core利用率，脚本自动diff Top15算子耗时。

**方法三：msprof Task Wait深度分析**
```bash
msprof --application="python3 scripts/ask_{model_name}.py --use-acl-graph" --output=PROF/
python3 {pypto_repo}/.agents/skills/pypto-fused-op-integration/scripts/analyze_msprof_taskwait.py PROF_xxx/
```

**方法四：torch_npu profiler + parse_prof 整网 PyPTO kernel 级分析**

适用于同时替换了多个算子、需要逐 kernel 统计 PYPTO 耗时占比的场景。

前置条件：推理脚本已支持 `--profile-dir`（用 `torch_npu.profiler.profile()` 包裹 `model.generate()` 并设置 `analyse_flag=True`），产出 `ASCEND_PROFILER_OUTPUT/trace_view.json`。

```bash
# 采集 PyPTO 模式 trace（一次 generation，不需 warmup/iters 循环）
python3 scripts/ask_{model_name}.py --use-pyto \
    --profile-dir ./prof_pyto --output-length 100 \
    --report-file ./prof_pyto/bench_pyto.json

# parse_prof 解析（过滤 tilefwk / PYPTO 命名的 kernel）
python3 -c "
import os, json
prof = {}
exec(open('/path/to/parse_prof.py').read(), {'os': os, 'json': json, '__builtins__': __builtins__})
result = prof['parse_prof']('./prof_pyto')
print(json.dumps(result, indent=2))
"
```

**输出：**
```json
{
  "prof": {
    "aicore_e2e": 884527.4,        // PYPTO kernel 总耗时 (μs)
    "aicore_e2e_jitter": 2.06,     // 抖动系数 (整网推理 token 间差异大，>1 正常)
    "aicpukernel_gap": 0.0         // AICPU 阻塞间隙
  }
}
```

> **注意：** `aicore_e2e` 是 PYPTO/tilefwk 命名 kernel 的累计耗时，非单次均值。如需单 kernel 耗时分布（avg/min/max），直接解析 `trace_view.json` 按 `name` 字段聚合。

**进一步调优 Skill：** `pypto-op-perf-tune`（Tile配置、合图策略、向量化等）

**记录结果：** 将对比表格追加到 `{model_weight_dir}/scripts/README.md` 末尾。

---

#### 步骤 26：归档到 pypto-gym 仓库 [可选]

**触发：** 询问用户是否归档。

**变量定义：** 见 `references/directory_structure.md`

**文件映射：**

| 来源 (`{model_dir}/`) | 目标 (`{pypto_gym_repo}/`) | 操作 |
|---|---|---|
| `scripts/*` | `modeling/transformers/{model_name}/` | 全量覆盖 |
| `core/` 全部 .py | `src/pypto_gym/transformers/{model_name}/` | 全量覆盖 |
| `{model}_pto_kernels/`（除去 `*/golden*.py` + `*/test/`） | `src/pypto_gym/ops/pypto_tile/{model_name}/` | 顶层覆盖，子目录新建 |
| `{model}_pto_kernels/*/golden*.py` + `*/test/` | `tests/ops/{model_name}/` | 新建 |

**test 文件 import 改造：**

移至 `tests/ops/{model_name}/` 后，impl 不在同级目录，通过 `sys.path` 跨目录引用：
```python
from pathlib import Path; import sys; import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_IMPL = Path(__file__).resolve().parents[3] / "src/pypto_gym/ops/pypto_tile/{model_name}/{op}"
sys.path.insert(0, str(_IMPL))
from {op}_impl import {op}_wrapper
from {op}_golden import {op}_golden
```

**归档验证：** `python3 tests/ops/{model_name}/test_{op}.py` → `[PRECISION_PASS]` + `All tests passed!`

**提交 PR [可选]：** 使用 `pypto-pr-creator` skill。

---

#### 步骤 27：还原重建指南 [可选]

将归档文件 + 下载的权重重建为可运行环境。完整命令见原版 SKILL.md 或按以下核心步骤执行：

```bash
MODEL_DIR=/path/to/weights
mkdir -p $MODEL_DIR/core $MODEL_DIR/{model}_name_pto_kernels $MODEL_DIR/scripts

# 1. 复制归档文件（transformers/ops/scripts）
# 2. 修复 auto_map（从 config.json 动态提取 model_type 和 architectures）
# 3. 运行验证：python3 $MODEL_DIR/scripts/ask_{model_name}.py --prompt "你好" 和 --use-pto
```

---

#### 步骤 28：Git提交 PTO融合变更

```bash
cd {model_weight_dir}
git add {model}_name_pto_kernels/ core/ scripts/ config.json
git commit -m "pto integration: add fused kernel for {model_name}"
```

---

#### 步骤 29：提交与文档

创建 Issue 和 PR。推荐 Skill：`pypto-issue-creator`、`pypto-pr-creator`

---

**遇到问题？** 查阅 `references/troubleshooting.md`

---

**Skill 版本：** v3.6  
**最后更新：** 2026-06-04  
**维护者：** PyPTO Team  
**更新说明：** 
- v3.6: 步骤25新增方法四 — torch_npu profiler + parse_prof 整网 PyPTO kernel 级分析；步骤4补充 --profile-dir 参数说明
- v3.5: 8路并行迁移交叉验证 — 补充 golden 操作顺序要求、FP16 JIT 首编崩溃重试、CANN 多版本 set_env.sh 选择
- v3.4: 三路并行迁移实测 — tokenizer检查、CANN多版本诊断、单op延迟预期
- v3.3: CANN ops 验证、PTO 注入时序规则、kernel dtype 策略
- v3.2: 阶段三算子开发拆分为两条路线