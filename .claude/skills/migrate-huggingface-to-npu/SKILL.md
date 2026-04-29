---
name: migrate-huggingface-to-npu
description: 将大语言模型迁移到华为昇腾NPU环境运行。当用户提到"NPU"、"昇腾"、"Ascend"、"华为AI处理器"、"华为GPU"、或想要在NPU上运行大模型、解决torch/torch-npu版本问题、创建推理脚本、部署HuggingFace模型时使用此skill。也适用于用户询问模型内存占用、离线加载模型、遇到transformers导入错误、auto_map配置问题等场景。即使用户未明确说"迁移"或"NPU"，只要涉及华为昇腾平台的模型部署或推理，也应触发此skill。
---

# 大语言模型迁移到NPU运行指南

## 概述

本skill记录将HuggingFace大语言模型迁移到华为Ascend NPU环境的完整步骤。

**强制要求：** 必须使用真实NPU硬件，最终验证推理脚本成功运行。

**关键要求：**
- **询问HuggingFace模型链接**（如：https://huggingface.co/Qwen/Qwen2-7B）
- 询问模型存放目录（默认：/data/models）
- 强制检查NPU环境可用性
- 验证ask脚本在NPU上成功运行

## 关联Skill获取方式

本skill位于 `pypto-gym` 仓库（`cann/pypto-gym`），与 `pypto-fused-op-integration` 同级。

本skill引用的其他skill不在本仓库中，**均位于 pypto 主仓库**：
```
https://gitcode.com/cann/pypto/tree/master/.agents/skills
```

| 被引用Skill | 用途 | 所在仓库 |
|------------|------|---------|
| `pypto-environment-setup` | NPU环境安装（步骤1） | cann/pypto |

> **获取方法：** 克隆或浏览 `https://gitcode.com/cann/pypto`，skill文件均在 `.agents/skills/` 目录下，每个skill对应一个子目录。

---

## 完整迁移流程（9步）

### 步骤0：获取用户信息

**必须询问：**
1. HuggingFace模型链接 → 提取 `repo_id`（如：Qwen/Qwen2-7B）和 `model_name`（如：Qwen2-7B）
2. 模型存放目录 `user_model_dir`（默认：/data/models）

**自动检测：**
- `pypto_repo`：当前仓库根目录（通过 pwd 或检测 .git 文件，仅用于归档）

**变量定义（后续步骤统一使用）：**
```
model_weight_dir = {user_model_dir}/{model_name}           # 模型权重目录（运行时文件）
script_dir = {model_weight_dir}/scripts                    # 脚本目录
core_dir = {model_weight_dir}/core                         # 代码目录（transformers内置模式）
model_archive_dir = {pypto_repo}/models/experimental/huggingface/{model_name}  # 归档目录（可选）
```

**检测已有模型：**

```bash
ls {model_weight_dir}/*.safetensors {model_weight_dir}/*.bin 2>/dev/null
```

- 存在权重 → 直接使用，跳过步骤 4
- 不存在 → **询问：**「无权重文件。有已下载的目录吗？（给路径则跳过下载，否则自动下载）」

若用户提供目录：`model_weight_dir` / `script_dir` / `core_dir` 同步更新。

### 步骤1：检查NPU环境与内存预估

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

### 步骤2：安装依赖（版本匹配是关键）

**torch和torch-npu版本必须完全一致！**

```bash
pip install torch==2.7.1 torch-npu==2.7.1
pip install transformers accelerate sentencepiece protobuf

# 验证安装
python3 -c "import torch; import torch_npu; print(f'torch: {torch.__version__}'); print(f'NPU可用: {torch.npu.is_available()}')"
```

### 步骤3：项目目录结构

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
    ├── ask_{model_name}.py          # 推理脚本
    ├── bench_{model_name}.sh        # 两段式基准测试脚本
    ├── sample_inputs.txt            # 示例提示词文件（bench使用）
    └── README.md                    # 迁移说明（必须）
```

**trust_remote_code模式：** core/目录不存在，网络结构代码已包含在HuggingFace下载的文件中。

### 步骤4：下载模型

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

### 步骤5：创建ask脚本

**创建目录并生成脚本：**
```bash
mkdir -p {script_dir}

python3 {pypto_repo}/.agents/skills/migrate-huggingface-to-npu/scripts/generate_ask_script.py \
    --model-name "{model_name}" \
    --script-dir "{script_dir}" \
    --default-model-dir "{user_model_dir}"
```

**生成的脚本特性：**
- 默认模型路径：`{user_model_dir}/{model_name}`（可通过 `--model-path` 指定）
- 使用 `local_files_only=True` 离线加载，`trust_remote_code=True`
- 参数：`--prompt`（优先级最高）、`--sentence_file`（从文件读取提示词）、`--output_length`（默认100）、`--use_pypto`（PyPTO占位标记）
- `--prompt` 提供时优先使用，否则 `--sentence_file` 提供时读取文件，均不提供则使用默认 "你好"

### 步骤5.5：生成benchmark脚本

生成两段式基准测试脚本，分别测试baseline和PyPTO模式：

```bash
python3 {pypto_repo}/.agents/skills/migrate-huggingface-to-npu/scripts/generate_bench_script.py \
    --model-name "{model_name}" --script-dir "{script_dir}" --model-dir "{model_weight_dir}"

cp {pypto_repo}/.agents/skills/migrate-huggingface-to-npu/scripts/sample_inputs.txt {script_dir}/
chmod +x {script_dir}/bench_{model_name}.sh
```

**生成的脚本行为：**
- Phase 1：调用 `ask_{model_name}.py` 不加 `--use_pypto`（baseline）
- Phase 2：调用 `ask_{model_name}.py` 加 `--use_pypto`（PyPTO占位模式）
- 两阶段均使用 `sample_inputs.txt` 作为输入，`--output_length 100`

### 步骤6：本地完整代码部署（强制执行）

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
   python3 {pypto_repo}/.agents/skills/migrate-huggingface-to-npu/scripts/fix_imports.py {model_weight_dir}/core/modeling_{model_type}.py
   python3 {pypto_repo}/.agents/skills/migrate-huggingface-to-npu/scripts/fix_imports.py {model_weight_dir}/core/configuration_{model_type}.py
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

### 步骤7：验证脚本运行

```bash
python3 {script_dir}/ask_{model_name}.py --device 15
```

**验证通过标准：**
- 脚本成功加载模型到 NPU
- 输出显示正在使用 NPU 设备
- 生成回复并输出
- 无错误退出

### 步骤8：可选归档到 pypto 仓库

**验证通过后询问用户是否归档便于版本控制。若同意：**

```bash
mkdir -p {model_archive_dir}
cp -r {model_weight_dir}/core {model_archive_dir}/ 2>/dev/null || true
cp -r {model_weight_dir}/scripts {model_archive_dir}/
```

**说明：** 归档仅用于版本控制，transformers 不读取此目录。修改代码需同步更新 `{model_weight_dir}/`。

---

### 步骤 8.5：Git 管理模型目录 [可选]

将代码文件纳入版本控制，排除权重大文件：

```bash
cd {model_weight_dir}

# HuggingFace 仓库已有 .git 则跳过 init
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

git add core/ scripts/ *.md *.sh .gitignore
git commit -m "migrate {model_name} to NPU — add ask script, core code, README"
```

> 后续 `{model}_pto_kernels/` 等 PyPTO 产出的代码变更也通过此 git 仓库管理。

---

### 步骤 9：下一步 — PyPTO 融合算子优化 [可选]

**必须询问：** 是否需要 PyPTO 融合算子整网集成？

模型已可在 NPU 推理。`pypto-fused-op-integration` 复用本 skill 产出
（ask 脚本、core 代码、权重目录、.git），进行算子融合 → 集成 → 端到端验证。

**启动：** `使用 pypto-fused-op-integration，模型路径 {model_weight_dir}`

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
python3 {pypto_repo}/.agents/skills/migrate-huggingface-to-npu/scripts/fix_imports.py {model_weight_dir}/core/modeling_{model_type}.py
```

---

**Skill 版本：** v1.3  
**最后更新：** 2026-04-29  
**维护者：** PyPTO Team  
**更新说明：** 步骤5增强ask脚本（新增--sentence_file/--output_length/--use_pypto），新增步骤5.5生成两段式benchmark脚本