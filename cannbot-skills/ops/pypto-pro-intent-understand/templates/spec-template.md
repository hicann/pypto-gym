# 算子需求规范

## 机器合同

> 下列 JSON 是公式、公开接口、验收配置的唯一机器事实源。正文只解释语义与证据，不复制字段值。

```json machine-contract
{
  "schema_version": 1,
  "op_name": "{{OP_NAME}}",
  "formula": "{{FORMULA_OR_NUMBERED_STEPS}}",
  "supported_dtypes": [],
  "inputs": [],
  "outputs": [],
  "default_params": {},
  "tolerance": {
    "atol": null,
    "rtol": null
  },
  "dynamic_axes_ranges": {},
  "shape_constraints": [],
  "p0_cases": [],
  "perf_target": null
}
```

填写约束：

- `op_name` 使用 lower_snake_case；`supported_dtypes` 按首次出现顺序列出本 SPEC 公开输入输出实际使用的全部 canonical dtype。同一接口的另一组 dtype 组合拆为独立 class/SPEC。
- `formula` 用一个 JSON 字符串定义每个公开输出；简单算子写可审查的等式，复杂递推写编号伪代码步骤（换行转义为 `\n`），且每个输出都必须是赋值或箭头目标。不要只写自然语言标签。
- `inputs` / `outputs` 按公开签名顺序填写，每项严格包含 `name`、`shape`、`dtype`、`value_range`；rank-0 shape 写 `[]`，闭区间值域写有限数值 `[min, max]`。
- shape 维度只使用正整数，或由符号、整数、`+ - * //` 和括号构成的表达式；每个动态符号须在至少一个输入 shape 中作为独立维度出现，并在 `dynamic_axes_ranges` 中给出正整数闭区间。
- `default_params` 只放公开签名中的标量默认参数，顺序与签名一致；无默认参数时写 `{}`。
- `p0_cases` 至少一项，每项严格包含 `name`、`params`、`input_shapes`、`output_shapes`。所有映射均按合同声明顺序覆盖全量字段；首项 `params` 等于 `default_params`，每个 shape 都是公式代入后的具体整数数组。
- 不适用的可选字段使用空数组、空对象或 `null`，不要保留示例值或另建第二份机器表格。

## 语义说明

### 1. 功能与分类

{{DESCRIPTION_AND_CATEGORY}}

### 2. 公式符号与依据

{{FORMULA_NOTATION_AND_RATIONALE_WITHOUT_REPEATING_MACHINE_FIELD}}

### 3. 算法描述

{{ALGORITHM_OR_NOT_APPLICABLE}}

### 4. 数据流说明

{{DATAFLOW}}

### 5. 接口语义

{{INTERFACE_SEMANTICS_WITHOUT_REPEATING_MACHINE_FIELDS}}

### 6. 功能与可选参数依据

{{FEATURE_AND_OPTIONAL_PARAM_RATIONALE}}

### 7. 精度语义

{{PRECISION_SEMANTICS_WITHOUT_REPEATING_TOLERANCE}}

### 8. 动态 Shape 与约束依据

{{SHAPE_RATIONALE_WITHOUT_REPEATING_MACHINE_FIELDS}}

### 9. 边界条件处理

{{ZERO_EXTREME_NAN_INF_BEHAVIOR}}

### 10. P0 与性能目标依据

{{P0_AND_PERF_RATIONALE_WITHOUT_REPEATING_MACHINE_FIELDS}}

### 11. 参考与来源

{{SOURCES_AND_CONFIDENCE}}

### 12. 自动决策

{{DECISIONS_OR_NOT_APPLICABLE}}

---
*生成时间: {{TIMESTAMP}}*
*确认状态: {{CONFIRMATION_STATUS}}*
