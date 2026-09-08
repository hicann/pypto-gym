# PyPTO Pro API 索引

在 `$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/` 下按功能类别定位，再读取该类 `index.md` 或按 API 名递归查找正文。下表路径均相对此目录，内容随资料版本变化，以缓存中的索引为准。

| 功能类别 | 索引路径 |
|---|---|
| 全部 Pro API | `index.md` |
| SIMD 基础数据结构 | `SIMD-API/basic_data_structures/index.md` |
| SIMD 操作（搬运、矩阵计算、控制流、同步等） | `SIMD-API/index.md` |
| VF 计算（寄存器类型、搬运、计算、归约、掩码等） | `SIMD-API/vf_computation/index.md` |
| SIMT 执行控制 | `SIMT-API/execution/index.md` |
| SIMT 标量计算 | `SIMT-API/scalar_compute/index.md` |
| SIMT 原子操作 | `SIMT-API/atomic/index.md` |
| Python 语法糖 | `Utils-API/python_syntax_sugar/index.md` |
| 调试工具 | `Utils-API/debugging/index.md` |

例如查跨核同步：读取 `SIMD-API/synchronization/index.md`，再读取其中的 `set_cross_core.md`、`wait_cross_core.md` 正文，确认参数与约束。

`docs/api/pro_api/` 与 `docs/pypto_pro/api/` 保留同一份上游 API 原文，检索后者即可。编程范式与用法见 [`doc-index.md`](doc-index.md)。
