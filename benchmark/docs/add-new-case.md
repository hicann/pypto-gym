# 新增 KernelBench Case

本文档说明如何为 benchmark 增加 PyPTO 自维护的 KernelBench 风格 case，并让
`case_loader.py` 自动生成符合 PyPTO 工作流要求的 `SPEC.md`。

## 放置位置

新增 case 应提交到 PyPTO 维护的 KernelBench fork，并放在 `pto_case` level 下：

```text
KernelBench/pto_case/<N>_<Name>.py
```

约束：

- `pto_case` 是 PyPTO 自维护 case 的 level 名。
- 文件名使用 KernelBench 扁平布局：`<序号>_<CaseName>.py`。
- 序号建议从 `pto_case` 目录未占用编号继续递增。
- PyPTO 仓内不保存这些 case 文件；本仓通过下载后的 KernelBench 数据集读取。

## 必需代码结构

每个 case 必须包含以下 KernelBench 标准入口：

```python
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, *init_args):
        super().__init__()

    def forward(self, *inputs):
        ...


def get_inputs():
    return [...]


def get_init_inputs():
    return [...]
```

约束：

- `Model(*get_init_inputs())(*get_inputs())` 必须能直接运行。
- `get_inputs()` 返回 forward 输入列表。
- `get_init_inputs()` 返回构造参数列表；没有参数时返回 `[]`。
- 输入 tensor 的 shape/dtype 应尽量固定、可复现，便于 loader 生成
  `p0_shapes` 和 `supported_dtypes`。

## 公式和动态轴

新增 case 如果能提供数学语义和动态轴，请在文件顶层添加两个大写全局变量：

```python
FORMULA = "out[b, s, d] = x[b, s, d] + bias[d]"
DYNAMIC_AXIS = ["B", "S"]
```

- `FORMULA`: 数学公式或简洁计算语义，会写入 `SPEC.md` 的
  `### 1.3 数学公式` 小节。
- `DYNAMIC_AXIS`: 动态轴名称列表，会写入 `SPEC.md` front matter 的
  `dynamic_axis` 字段。
- 两者必须是可被 `ast.literal_eval` 解析的常量表达式。

## 推荐模板

```python
#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn


FORMULA = "out[b, s, d] = x[b, s, d] + bias[d]"
DYNAMIC_AXIS = ["B", "S"]


class Model(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return x + bias


def get_inputs():
    batch = 2
    seq_len = 4
    hidden_size = 8
    x = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32)
    bias = torch.randn(hidden_size, dtype=torch.float32)
    return [x, bias]


def get_init_inputs():
    return [8]
```

## 本地校验

benchmark 测试统一通过 pytest 执行：

```bash
python -m pytest benchmark/tests
```

新增 case 的 loader 行为应在 `benchmark/tests` 中补充或更新断言。生成的
`SPEC.md` front matter 应包含 dtype、shape、tolerance 和动态轴信息；正文中
应包含公式小节。

## 运行 benchmark

使用 `pto_case` 跑新增 case 时，先下载包含该目录的 KernelBench fork，
再在 YAML 中设置 `bench_dir`、`cases`、`devices` 和 verifier 配置：

```yaml
bench_dir: "/path/to/KernelBench/KernelBench"
cases: "pto_case=1"
```

然后使用公开 CLI：

```bash
python -m benchmark run --config configs/local_new_case.yaml
```

提交前至少执行 pytest。完整业务验证是否可跑，取决于当前环境是否具备
NPU、CANN、`torch_npu`、opencode 和 LLM 配置。
