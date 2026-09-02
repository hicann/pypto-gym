# Dual-AIV mailbox 负向证据

本页只保留旧 `dual-aiv-mailbox.py.tmpl` 中独特的 liveness 与 tail 失败边界。该原型未经
production-shaped 验证，已从 `templates/` 移除；它不是 Stage 5 优化项、实现骨架或可复用 API
范式。

## 已知风险

- 源码、静态检查、数学结果或 set/wait 总数一致，都不能证明 Acc→Vec mailbox 在真实 block
  launch 中可存活。
- 对齐 shape 返回成功不能证明 odd tail 正确；必须同时检查有效行 bit-exact、无效行 sentinel
  未被覆盖，以及所有 launched AIV lane 均参与事件。
- FREE/READY token 穿过 MIX barrier 的生命周期未经证明。无 barrier 的 probe 不能替代包含
  “seed → first wave → MIX barrier → second-wave reuse”的生产拓扑。
- 一个物理核重复执行多个 mailbox task 时，旧记录要求最后一个 READY/FREE/STORE-FREE credit
  排空，并在两种 engine 的最后事件后各执行 local `bar_all()`；该结论不证明 token 可以跨 MIX
  barrier 保存或复用。
- 双 ring 只是未验证的历史设想，不得从本页恢复为 production candidate。

## 失败判据与保留证据

最小 production-shaped probe 必须覆盖 aligned/odd tail、真实 block/task mapping、每核不均匀
task depth 和最大动态深度。确认已进入目标原型执行并排除环境、工具、编译和 launch 前故障后，
任一 timeout、device error、结果不匹配、sentinel overwrite、缺失 tail，或 watchdog 中 AIC/Cube
持续活跃而 AIV 空闲，均直接否定该原型。

失败后保留 CCE、设备计数器、命令、shape、task depth、source SHA 和原始日志；发生 device
error 后关闭该 source SHA，修改协议并使用新 SHA 才能重试。当前算子的新 mailbox 方案登记为
`bottleneck_derived`；只有贡献为可复用预置项时，才以新稳定 ID 加入知识卡或模板 INDEX，并用
独立事实锚点、完整正确性和目标设备性能证据重新评审。
