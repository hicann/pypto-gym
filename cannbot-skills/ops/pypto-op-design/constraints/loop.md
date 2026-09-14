# 循环约束

```yaml
- id: C-LOOP-01
  level: should
  rule: "静态循环可用 Python range 展开；迭代较多时评估图膨胀，不能仅因边界静态就排除 pypto.loop。"
  source: "PyPTO docs/zh/api/tensor_api/controlflow/pypto-loop_unroll.md"
  consequence: "过度展开增加编译成本。"

- id: C-LOOP-02
  level: must
  rule: "符号循环边界使用 pypto.loop 等框架循环，不能交给 Python range。"
  source: "PyPTO docs/zh/api/tensor_api/controlflow/pypto-loop.md"
  consequence: "Python 无法把运行时符号转换为整数。"

- id: C-LOOP-03
  level: must_not
  rule: "不将符号循环索引用作 Python 容器下标或 Python 布尔条件。"
  source: "PyPTO docs/zh/api/tensor_api/controlflow/pypto-loop.md"
  consequence: "符号无法转换为 Python 整数或真值。"

- id: C-LOOP-04
  level: must
  rule: "循环内读取循环前尚未提交的计算结果时，核对提交依赖；submit_before_loop=True 用于进入循环前提交已有任务，并不代表每轮迭代之间同步。"
  source: "PyPTO docs/zh/api/tensor_api/controlflow/pypto-loop.md"
  consequence: "依赖处理错误可能导致结果不可用。"

- id: C-LOOP-05
  level: should
  rule: "展开因子从单一候选开始，依据边界和余数处理选择；loop_unroll 解包索引与展开因子两个返回值。"
  source: "PyPTO docs/zh/api/tensor_api/controlflow/pypto-loop_unroll.md"
  consequence: "多种展开方式会增加代码路径，错误解包会影响实现。"

- id: C-LOOP-06
  level: must
  rule: "需要跨迭代保留的状态在合适的外层作用域建立，明确初始化、更新及最终写回；局部临时张量不等于循环携带状态。"
  consequence: "状态重置或作用域错误会破坏结果。"
```

## 循环选择

| 需要 | 表达方式 | 返回值 |
|---|---|---|
| 按运行时边界重复执行 | `pypto.loop` | 符号索引 |
| 按指定因子展开计算 | `pypto.loop_unroll` | `(索引, 展开因子)` |
| 少量、编译期确定的迭代 | Python `range` | 具体整数索引 |

`loop_unroll` 的展开因子在对应代码路径中是编译期值，可用于确定当前处理的 tile 大小。`submit_before_loop` 只指定进入循环前提交已有计算，不保证数据搬运与计算重叠。
