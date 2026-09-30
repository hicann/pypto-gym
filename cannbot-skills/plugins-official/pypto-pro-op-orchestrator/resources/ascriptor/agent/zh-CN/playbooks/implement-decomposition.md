# 按已有分解实现

新上下文先读[共同语言](../common-language.md)，实现或实质修改 kernel 前完成
[preflight](../references/authoring-preflight.md)。

按照已接受 stage/DAG 和精度契约实现，不把公式偷偷放进 host reference。先重跑独立完整与
stage reference。失败或 ABI 不完整时，通过[分解](decompose.md)依据证据修正契约，不修改
预期值迎合 kernel。

1. 记录 contract/library revision，拓扑排序。各 stage 明确 typed signature、symbol、输出
   初始化、footprint、workspace/saved state 归属和比较预算。
2. 逐 leaf 实现并验证后再用于组合，覆盖支持的最小、常规、tail 和重复 slot case。按需读
   一个公开样例和[内存/同步事实](../references/memory-and-tails.md)。
3. 在 `main.py` 的 `execute` 里组合：它按顺序 launch 各 stage 并返回命名输出，每个目的张量
   都由调用方分配成 poison 值、以 `seed_outputs=True` 交给 `OpExec`，这样"没被写过的 lane"
   与"被正确写成 0 的 lane"仍然可区分。所需 state preparation 和全部 import 都在该目录内。
   `reference.py` 不调用 simulator；不支持的 device/backend 不能回退执行 reference。
4. 验证 `--stages` 的输出对 `reference_stages`、完整执行对分阶段 reference、最终输出对
   独立原公式，两种误差预算均保留。每个输出检查名称、shape、dtype 和非有限值政策。
   不开 `--stages` 时，最终结果错了只知道错了，不知道是哪一次 launch 造成的。
5. 把整个目录单独复制到 scratch，在那里运行：`python main.py --list`、`python main.py`，
   再 `--launcher pipesim`，最后在有卡的机器上用设备 launcher。先正确后计时。
   emit、compile、functional、pipesim、board 证据分开。

scaffold 是按目录里每个文件各给一份模板：
[`kernel.py.template`](../../templates/kernel.py.template)、
[`reference.py.template`](../../templates/reference.py.template)、
[`main.py.template`](../../templates/main.py.template)、
[`metadata.json.template`](../../templates/metadata.json.template)。四份一起复制，去掉
`.template` 后缀，再填 TODO；填完之前每个函数体都会明确抛 `NotImplementedError`，而且
`kernels/tools/build_index.py` 对目录里多出第五个文件和少一个文件都会报错。
`metadata.json.template` 带齐九个必需键，`device` 与 `topology` 故意留成 TODO：它们取自
`build_index.py --check` 强制的受控词表，要按词表选，不要猜。想看写完之后的样子，读最小的
demo [`examples/axpy`](../../../kernels/ascriptor_kernels/examples/axpy)。
并行获得授权时，为 worker 分配互不重叠的 stage 文件及固定版本；主 agent 负责共享接口、
composition、环境和硬件。worker 返回文件、精确命令/结果、ABI 修改请求和待办，交付消息
不能代替集成验收。
