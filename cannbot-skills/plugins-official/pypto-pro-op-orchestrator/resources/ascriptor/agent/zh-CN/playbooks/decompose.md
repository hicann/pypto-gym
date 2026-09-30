# 分解一个算法

新上下文先读[共同语言](../common-language.md)，实现或实质修改 kernel 前完成
[preflight](../references/authoring-preflight.md)。

仅在任务允许多个 runtime kernel 时使用。交付一个明确计划、独立 stage reference 和独立完整
reference。单 matmul 中的 tile 循环不是 host stage，也不会自动完成跨 core reduction merge。

1. 写公共数学与[编写契约](../references/authoring-contract.md)，保留原公式计算顺序和 cast。
   ONNX 输入还要记录 opset、IO、真实 initializer、外部数据和动态维度；随机替代权重仅证明
   连通性。本库不承诺 ONNX 转换工具。
2. 定义各 stage ABI：参数顺序、输入/输出 shape、dtype、layout 与 workspace 归属。
   按依赖确定 launch 顺序。每条边记录 producer/consumer、shape、
   dtype、layout、已初始化范围、alias 与最后消费者。验证唯一 producer、完整输出和无环图。
3. 记录每处累加、cast、saturation 和 rounding。相对于原公式的 `plan_tolerance` 与相对于
   stage reference 的 `implementation_tolerance` 分开，各有比较数值与理由。代数相等不能
   证明浮点相等。
4. saved state 记录版本、producer、初始化、含义、layout、dtype、生命周期与 backward
   consumer。backward 单元自带必需 forward-state 生成，不能导入相邻 forward 项目。
5. 使用[计划模板](../../templates/decomposition-plan.md)。独立的完整公式与分阶段公式写在同
   一个 `reference.py` 里，它永不 import ascriptor，确定性的 `make_inputs(case)` 也在其中；
   两种检查都在同一个目录内。[可执行示例](../../templates/decomposition/reference_example.py)
   展示独立完整/分阶段公式、命名 checkpoint 和 DAG 检查。
6. 执行所有 stage 与 reference composition，直接对原公式验证总体预算；检查 DAG，然后在
   scratch 里单独运行 `reference.py`（不装或干脆不 import ascriptor）。它若仍能给出分阶段和
   完整期望值，独立性就是事实而不只是意图。

在 agent checkout、已安装 Torch 的环境执行：

```bash
python templates/decomposition/reference_example.py
```

它输出生成 case 与 stage 数，未执行 DSL kernel，也不声称硬件支持。
交接就是一个 demo 目录，别无他物：`kernel.py`、`reference.py`、`main.py`、`metadata.json`。
`reference.py` 导出 `make_inputs`、`reference`，流水型还导出 `reference_stages`；`main.py` 放
`execute`、case 列表，以及一个 `--stages` 开关——打开后比较每个中间结果，而不只是最终输出。
按这个形态交接，不另造协议。仅 reference 的规划可明确保留 kernel 执行待办。错误契约用证据和版本修改后再继续依赖它的实现。
契约接受后按[分解实现](implement-decomposition.md)逐 leaf 落地并验证 composition。
