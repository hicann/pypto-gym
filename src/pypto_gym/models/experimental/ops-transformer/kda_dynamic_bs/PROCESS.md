# KDA 实现过程记录（`kda_dynamic_bs`）

## 0. 任务目标

用户目标：

1. 用 PyPTO 实现 KDA 算子
2. `B/S` 两个轴都实现为动态轴
3. 把实现过程完整记录下来

---

## 1. 需求收敛（Stage 1）

### 1.1 已确认事实

- 仓库中已有与递推注意力相关实现，可参考：
  - `models/qwen3_next/gated_delta_rule_impl.py`
  - `examples/02_intermediate/controlflow/others/dynamic.py`
- PyPTO 动态轴能力明确支持 `pypto.DYNAMIC`

### 1.2 关键决策

- 采用 KDA 风格递推核心（state 递推 + token 输出）
- `B/S` 显式标记为 `pypto.DYNAMIC`
- `D` 保持静态编译轴（由 wrapper 构建 kernel 时确定）

---

## 2. API 可行性分析（Stage 2）

### 2.1 重点查阅文档

- `docs/api/operation/pypto-view.md`
- `docs/api/operation/pypto-assemble.md`
- `docs/api/operation/pypto-expand_clone.md`
- `docs/api/operation/pypto-sum.md`
- `docs/api/controlflow/pypto-loop.md`
- `docs/api/config/pypto-set_vec_tile_shapes.md`

### 2.2 关键约束

- `view(shape=...)` 的 `shape` 必须是 `List[int]`，不能放 SymbolicScalar
- `loop` 索引是符号值，只能用于 `offset` 和 loop 内运算
- `sum` 在本场景使用 FP32 路径
- `assemble` 用于显式写回，避免 `output = ...` 的无效赋值

---

## 3. 设计收敛（Stage 3/4）

### 3.1 数据流

- 外层：`B` 动态 loop
- 内层：`S` 动态 loop
- 每步：
  1. `view` 取 `[1,1,D]` token
  2. `reshape` 成 `[D,1]` / `[1,D]`
  3. `expand_clone + mul` 得到外积 `[D,D]`
  4. 更新 `state`
  5. `sum` 归约得到输出 `[1,D]`
  6. `assemble` 写回 `output[b,s,:]`
- 每个 batch 结束后写回 `last_state[b,:,:]`

### 3.2 为什么不用 matmul 版外积

- `k[:,None] @ v[None,:]` 虽然表达直接，但在部分 tile/对齐组合上约束复杂
- v1 选择 `expand_clone + mul` 先保证双动态轴正确性与稳定性

---

## 4. 代码实现（Stage 5）

新增目录：

- `models/experimental/ops-transformer/kda_dynamic_bs/`

新增文件：

- `SPEC.md`
- `API_REPORT.md`
- `DESIGN.md`
- `kda_golden.py`
- `kda_impl.py`
- `test_kda.py`
- `README.md`
- `PROCESS.md`

---

## 5. 验证过程（Stage 5）

### 5.1 已执行

- 对新增 Python 文件做语法级检查（`py_compile`）
- 使用 `kda_golden.py` 的自检入口做纯 torch 正确性检查

### 5.2 待环境验证

- `test_kda.py` 的 NPU 实测需要有效 NPU 环境与 `TILE_FWK_DEVICE_ID`
- 若环境缺失，应在提交结果中明确标注“未完成 NPU 实测”的风险边界

---

## 6. 当前交付状态

- 规格、API 报告、设计文档：已完成
- golden 与 PyPTO 实现：已完成
- 测试入口：已完成
- 过程记录：已完成
