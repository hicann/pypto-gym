# 编写与执行模型

先看[最小代码锚点](playbooks/author.md#first-code)，完整 owner 样例提供 imports、生成输入与独立 reference。

1. 声明 typed GM 输入/输出和显式标量参数。Host 分配输出，按签名顺序传参并使用 `OpExec`
   返回值。读改写任务用 `seed_outputs=True` 保留初始化。
2. 带装饰器的 Python AST 变成 Surface IR。Kernel 中 `range` 即使使用字面量边界也是
   runtime 循环，`unroll` 在编写期展开。静态表达式和参数全为静态值的调用执行 host Python。
   请求的公式留在 kernel 内；helper 与控制流查 [authoring API](../../library/docs/api/authoring.md)。
3. 按物理形状分配 local storage。`<<=` 根据源/目标空间选择搬运。沿 GM → local storage →
   VF/SIMT 或 cube 计算 → GM 推进；改变形状前先推导 valid lane 和完整指令 footprint。
4. `auto_sync` 处理受支持的同侧依赖。Slot 复用、跨侧生产/消费需要显式的归属与生命周期计划，
   出现这些特征时再沿 preflight 的对应触发项阅读。
5. 检查 `kernel.ir()`，执行独立比较，再检查 lowered hazard 与归属；按[证据表](common-language.md#evidence)解释结果。

GMList、分组寄存器、标量单元与精确签名查 [API 入口](../../library/docs/api/README.md)。
A2/A3 CCE 与 A5 的适用范围按当前任务验证；源码版本见
[pyproject.toml](../../library/pyproject.toml)，验证范围见[状态](../../library/docs/status.md)。
第一个 cube kernel 可用[完整 matmul 样例](../../library/examples/api/cube_matmul)，遵循其
FP16 16×16 操作数和 FP32 输出契约。写出来的东西在本机和上卡怎么跑，见[一页](references/development-execution.md)。
安装见[快速开始](quickstart.md)，递进任务见[练习](practice.md)。
