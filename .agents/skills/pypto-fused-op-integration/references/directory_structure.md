# 目录结构参考

## 阶段零：模型部署目录结构

所有文件集中在模型权重目录（默认 `{user_model_dir}/{model_name}`）：

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

## 阶段四：算子库目录结构（pto_kernels）

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

## 归档到 pypto-gym 仓库后的结构

```
src/pypto_gym/ops/pypto_tile/qwen3_1_7b/
  __init__.py              # USE_PTO 开关 + 适配层函数
  rms_norm/
    rms_norm_impl.py        # PyPTO kernel 实现
    rms_norm_pypto_impl.py  # ModelNew 桥接类（可选）
    SPEC.md                 # 算子规格（可选）
tests/ops/qwen3_1_7b/
  rms_norm_golden.py        # PyTorch 参考实现
  test_rms_norm.py          # 单算子精度测试
```

## 变量定义

| 变量 | 定义 | 说明 |
|------|------|------|
| `model_weight_dir` | `{user_model_dir}/{model_name}` | 模型权重目录（运行时文件） |
| `script_dir` | `{model_weight_dir}/scripts` | 脚本目录 |
| `core_dir` | `{model_weight_dir}/core` | 代码目录（transformers内置模式） |
| `pto_kernels_dir` | `{model_weight_dir}/{model}_pto_kernels` | 算子库目录 |

## README.md 必须包含的字段

| 字段 | 说明 |
|------|------|
| HuggingFace | 模型的 repo_id |
| 权重目录 | 模型权重存放的实际路径 |
| 代码来源 | trust_remote_code 或 transformers包 |
| 运行命令 | 执行脚本的具体命令 |

**情况B（transformers内置模式）额外必须包含：**

| 字段 | 说明 |
|------|------|
| transformers版本 | 复制代码时的 transformers 版本号 |
| 代码位置 | 复制后代码的实际路径 |
| 修改内容 | 导入方式修改、auto_map添加等 |