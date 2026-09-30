# 找到唯一事实所有者

使用本快照的源码索引一起核对 `library/`、`kernels/`、`agent/`。
下列路径相对各自所属目录。作者直接使用仓内源码，无需安装 wheel 或另取源码仓。

| 事实 | 所有者 |
|---|---|
| Public API 名称与签名 | library facade 与相邻声明 |
| Frontend 约束 | library `ascriptor/frontend/rules_*.py`、static-subset RFC |
| IR 验证、语义、provenance | library `ascriptor/ir/`、`ascriptor/passes/`、RFC |
| Backend 生成 | library `ascriptor/backends/` 与 backend RFC |
| 执行/build/board | library `ascriptor/runtime/` |
| 功能/pipe model | library `ascriptor/backends/sim/` |
| 样例合同 | library `docs/rfc/0012-product-contracts.md`（第 3 节） |
| Kernel 算什么、有哪些 case、拿什么比较 | 两个 owner 的目录都一样：`metadata.json`、`main.py`、`reference.py` |
| 某 kernel 在某后端能不能跑 | 没有任何记录。在有卡的机器上用该 launcher 跑那个目录的 `main.py` |
| API 教学 | library `examples/`；attention 带 canonical 来源 |
| Defect | library `docs/defects/` |
| 路线 | agent 相应语言 router/playbook |

指南可以展示从 owner 的命名函数生成的小段流程代码。
[代码锚点](../../docs/code-anchors.json)和 `tools/sync_snippets.py --check` 核对其与所选 library 的一致性。
完整 imports、签名、范围、reference 和可执行样例仍由 owner 维护；指南中的摘录通过工具同步。

运行结果前核验源码 import origin：

```bash
python -c 'import ascriptor; print(ascriptor.__version__); print(ascriptor.__file__)'
```

路径应指向本快照的 `library/ascriptor/`。本地 Python 环境由主 agent 选择。Facade import
只绑定名称，不改变进程全局 target；decorated kernel 保有自己的 device。A2 tensor-vector
不借用 A5 register helper。
公开执行/检查使用 `ascriptor.runtime.OpExec`、`compile_kernel`、kernel `.ir()` 和文档 CLI。
实际输出从 `OpExec` 返回读取，不假定所有 launcher 都修改传入 placeholder。in-place/atomic
契约需要显式 output seeding。`sim` 是功能模拟，`pipesim` 是独立的 Lowered 模拟路线；
两者都是 `OpExec` launcher，其中一个通过不代表另一个通过。

不能打印的 op 报 source-located gap，IR 无 inline text 绕过验证。高级 backend 使用声明的
`Backend/Artifacts/Capabilities/ResourceLimits` 与 `ascriptor.backends` entry point；package
与 IR 版本分开。扩展前读 library 契约，不把 compiler helper 当普通 DSL API 导出。
机器访问仅在 `ASCRIPTOR_MACHINE_SPECS`、`ASCRIPTOR_BOARDS` 选择的 ignored 外部配置。
值不进入指南、元数据、发布日志或 commit message。硬件证据记录 profile/version，不记访问坐标。

<a id="coordination"></a>
## 协作任务

多个 agent 共同执行任务时，环境绑定或重绑、硬件锁管理、分支与共享改动整合
必须仅由主 agent 负责。Worker 必须只写分配的路径，将共享改动交回主 agent，不得再派生 worker。
每个 worker 最多运行一个重型测试进程，除非主 agent 明确调整资源预算。
设备执行必须使用隔离的输出身份。
