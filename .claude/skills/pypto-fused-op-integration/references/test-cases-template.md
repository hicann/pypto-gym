# 测试集采集与格式

本文件说明如何从真实网络采集 tensor 信息，构造必须 pass 的测试用例。

---

## 采集步骤

### 1. 定位打点位置

在模型代码中找到待替换算子的调用位置。

### 2. 插入打印代码

```python
print(f"[DEBUG] input: shape={x.shape}, dtype={x.dtype}")
```

### 3. 运行采集

执行模型推理，记录真实 shape/dtype。**禁止限制打印次数**，应采集所有调用场景。

### 4. 去重并创建 test_cases.json

对采集到的 shape/dtype 组合去重，每种唯一组合作为一个测试用例。

---

## test_cases.json 格式

**统一格式以 `pypto-op-develop/templates/test_cases-template.json` 为准。**

该模板定义了完整的字段规范：`op_name`、`source`、`test_cases[]`（含 `id`、`input`、`output`、`rtol`、`atol` 等）。
生成 test_cases.json 时请参考该模板，字段说明和示例均以该文件为准。

### 多输入算子补充

当算子有多个输入参数（如 RMSNorm 的 `hidden_states` + `weight`）时，`input` 字段使用命名子对象替代平铺的 `shape`/`dtype`。完整示例见 `pypto-op-develop/templates/test_cases-template.json` 中的 `multi_input_example`。

- **单输入算子**：`"input": {"shape": [...], "dtype": "..."}`
- **多输入算子**：`"input": {"param_name": {"shape": [...], "dtype": "..."}, ...}`

test 文件中按 `input` 结构类型判断：含 `shape` 键 → 单输入模式；否则遍历子键 → 多输入模式。

---

## 使用流程

test_cases.json 与 test_{op}.py 配合使用：

```bash
python test_{op}.py              # 遍历所有用例
python test_{op}.py case_001     # 运行单个用例
python test_{op}.py --list       # 列出所有用例
```

---

## 文件位置

```
models/{model_name}/{model}_pto_kernels/{op}/test/
├── test_cases.json     # 真实 tensor 信息
└── test_{op}.py        # 精度测试（遍历读取 JSON）
```

---

## 关键原则

真实用例是必须 pass 的基准，覆盖所有调用场景。
