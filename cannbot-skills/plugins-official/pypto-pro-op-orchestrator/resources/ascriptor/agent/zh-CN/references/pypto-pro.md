# 目标 PyPTO-Pro

这个后端和 CCE 性质不同的地方，按会咬人的顺序排列。执行入口本身见
[开发期怎么跑](development-execution.md)。

## 交付形态

"一个 PyPTO-Pro kernel" 在本工作区指的是 **Ascriptor DSL 源码，加上它 emit 出来的
`kernel_pypto.py`**（每个 shape 一份），外加 manifest、driver 和板上证据。不是手写
`pypto_pro.language` 的代码。它们在一个交付包里怎么摆，见[交付区](#delivery-area)。

<a id="emit-first"></a>
## 先 emit，再跑

```python
from ascriptor.runtime import compile_kernel
compile_kernel(kernel, backend="pypto_pro", bindings={"B": 2, "T": 9})
```

几秒钟，不传数据，不需要卡。**在 sim 之前做这一步。**

理由：PyPTO 的支持表是按 **opcode** 列的，但很多缺口是 **operand 级**的——同一个 opcode，
换一种 stride 或 dtype 就没有 pto 写法。这类缺口**只有 emit 时才暴露**。真实案例：一个
kernel 的 sim 和 pipesim 都通过了，emit 才报
`dma.gm_to_ub.nd loop_src_stride=[64, 1]: a non-unit INNERMOST stride (64) has no pto spelling`
（即 `gm_to_ub_nd_dma_transpose`），整个数据流方向要重写。先 emit 能把这个损失从"重写"降到
"改设计"。

`PyptoGap` 总是带 op id 和源码位置。

## 按标量取值特化

每个 shape symbol 都要有绑定，否则 emit 报 `scalar parameter D has no binding`。编译期常量
直接写字面量（`GM[f32, ('B', 'T', 'Nqk', 64)]`），只把真正运行时变化的维度留成 symbol。

一次 emit 只对应一组标量取值，所以**每个 case 一份产物**；板上第一次调用时才 JIT 编译。

<a id="核数"></a>
## 核数：漏了是挂起，不是报错

板卡条目必须写 `cube_cores`，填这张卡的 **AIC 核数**。缺了它：

- **未 pin 的 launch**（kernel 和调用都没写 `block_dim`）：emitter 回退到 device profile
  （a5 写的是 32），vec 模式再翻一倍，于是 **64 个 AIV 发到 56 个 AIV 的卡上**。实测：
  未声明的 vec kernel，emit 出来的 manifest 写的就是 `"block_dim": 64`。
- **已 pin 的 launch**：clamp 整个不执行，也不翻倍——`block_dim=8` 就发 `8`，所以 pin 了一个
  超过卡核数的数字，就照着发出去。

`sync_all` 是硬件 barrier，会等 launch 点名的每一个核。等不到的核不会报错，**会挂**——
曾有一个 `simt_atomic_add` 在这里坐了九分钟才被理解。所以 `run_pypto` 现在直接拒绝执行。
Ascend950PR 是 28 AIC / 56 AIV。

## 交付同步模式

新 PyPTO-Pro 交付默认只生成并验收 `sync_mode="auto_mutex"`，即
`@pl.jit(auto_mutex=True)`；仅用户明确要求时才生成 manual。
底层 `OpExec(..., launcher="pypto")`/`compile_kernel` 的历史默认仍是 manual，
所以交付工作流须显式选择 auto_mutex，不能依赖那个默认值。auto_mutex 编译或
真机验证失败时保留证据并报告阻断，不静默改成 manual。
细则见[交付同步政策](../runtime-and-maintenance.md#sync-closeout)。

## 必须和 CCE 对照

emit 成功不证明厂商编译，板上通过也不证明语义对。这个后端出过**能 build、能跑、结果错**的
情况：64-bit 的 `vf.cast` 在 pypto 下读错 lane，而 cce 逐位正确（D-221）。

所以板上出现意外结果时，**先用 cce 跑同一个 case 做对照**再下结论。一个 pypto 失败只有在
cce 通过同一个 case 时才是移植缺陷。

## 板上失败怎么读

`BoardError` 会带上板上 `run.log` 的尾部。两种常见的空消息签名：

| 日志里的 | 原因 |
|---|---|
| `libhccl.so: cannot open shared object file` | CANN 树不完整或已被物主删除 |
| `F7A008 FILE_ERROR … aarch64 toolchain g++` | 同上；pypto 按 canonical path 从 `ASCEND_HOME_PATH` 找交叉工具链 |

完整日志在 `out_dir/board_pypto_run.log`。

<a id="delivery-area"></a>
## 交付区

PyPTO-Pro 任务交出的是一个交付包，**不进画廊**——[补另外三个文件](development-execution.md#admission)
那条路是给 CCE demo 的。开发与原始证据留在目标项目的 `custom/<op>/`；最终可运行包
单独放在 `delivery/<op>/`。在本源码快照工作区开发时，这两个目录均位于任务项目下：

```
delivery/<op>/
├── kernels/         实际使用的 PyPTO-Pro 内核源码；相同实现可覆盖多个 case
├── golden_cpu.py    torch only，绝不 import ascriptor
├── wrapper.py       按 case 分流到 kernels/ 里的一份
├── test.py          唯一入口
├── DESIGN.md
├── REPORT.md
└── testing/ 等       仅当 test.py 的本地依赖确实需要时加入
```

Scriptor 的冻结合同、`scriptor/` 源码、`generated/` 导出、`.scriptor/` 状态与
`reports/` 原始证据仍在 `custom/<op>/`，不复制进交付包。`delivery/<op>/`
必须只靠本目录文件及声明的 PyPTO-Pro、Torch/NPU 运行依赖独立运行；测试辅助模块、
精度比较器等本地 import 必须随包提供，不能隐含依赖 `.opencode`、开发工作区或另一算子目录。
同一最终 DSL 导出的每个 case 必须映射到包内一份字节一致的内核源码；可以给不同
case 复用同一份源码，`kernels/` 中的每份源码至少被一个 case 命中。优先使用
`kernels/<variant>.py` 的精简布局；确需伴随本地依赖时可保留子目录。
`test.py` 使用字面量 `CASES`，逐行对应 SPEC P0 case，包含
`name`、`kernel`、`input_shapes`、`input_dtypes`、`output_shapes`、`output_dtypes`、`params`，
并与最终 `generated/export.json` 逐项一致。`wrapper.py` 实际从交付包的
`kernels/` 加载实现，不能仍从工作区 `generated/` 加载。

| 文件 | 负责什么 | 边界 |
|---|---|---|
| `golden_cpu.py` | `make_inputs(case)`、`reference(inputs)` | 和画廊 `reference.py` 同一条红线：调用被检查对象的 reference 什么也没检查 |
| `wrapper.py` | 按 case 选中 `kernels/` 里的一份 | 每个分支都要有 case 命中 |
| `test.py` | 通过交付包公开 wrapper 逐 case 真机运行，并与独立 CPU golden 比较 | `CASES` 是 SPEC P0 的交付 case 记录；`--output` 写逐 case 机器结果 |
| `DESIGN.md` | 生效的约束和它们逼出的结构 | 不记试错经过 |
| `REPORT.md` | 下面那份最小内容 | 删 `.tmp` 之前必须定稿 |

**四步在同一台机器上。** 在本机设备机器的同一解释器中生成输入、计算 CPU golden、
运行 kernel 并比较。Torch 随机输入跨平台不保证逐位相同，因此不能用其他机器
计算的 golden 对照本机 kernel 结果。每项任务使用独立工作目录。

**分流器要证明自己跑满。** `test.py` 逐 case 打印实际选中的文件，写出每例的
`name/status/kernel` 机器结果，收尾断言 `kernels/` 下每份内核源码至少被命中一次。
测试须在启动前污染输出并确认未写元素不会伪装通过；工作区的 DSL `OpExec` 测试
仍使用 `seed_outputs=True`，最终 PyPTO-Pro 包以等效的输出污染方式验证。
Scriptor 的独立 verifier 在工作区另行检查 DSL `OpExec` 与最终导出 wrapper；
交付包 `test.py` 只测试真正交给用户的 `wrapper.py`，不把开发工具变成运行依赖。

**REPORT.md 至少要有：** 每个 case × 每个 launcher（`sim`/`pipesim`/`pypto`）的结果，含 **cce
对照那一栏**——一个 pypto 失败只有在 cce 通过同一个 case 时才是移植缺陷；
实际选定同步模式的板上实测时延（见[交付同步政策](../runtime-and-maintenance.md#sync-closeout)）；每个
case 的 manifest `block_dim` 与 op counts；源码身份与实际执行的依赖版本；skip 掉的 case 及其理由，
理由要说明改跑什么；结尾的完成清单与剩余清单。
另记最终选定源码和导出产物的身份，逐 case 的 shape、dtype、参数、命中的 `kernels/` 文件及其
验收证据。若工作流先验收、后续调优，旧验收报告保留历史身份，`REPORT.md` 只把最终实际交付
版本写成最终版本；失败或回退的候选须明确标为未采用。一次交付的所有 dtype 都须在同一
`CASES` 和验收范围内，不能靠未经该交付入口验证的旁路目录补齐。
Scriptor 模式的 `REPORT.md` 写入最终 `generated/export.json` 及每个 `scriptor/*.py`
的完整 SHA-256，供收尾检查核对；只有短哈希不足以确定交付身份。
用唯一的 `Final export SHA-256: <64位哈希>` 行和每份源码各一行
`Final source SHA-256 (scriptor/文件.py): <64位哈希>` 明确标记最终版本。
历史候选的哈希可写在后文，但不能占用这些最终字段。

**删 `.tmp` 的顺序。** emit 产物和模型结果可重现，删了重跑即可；**板上结果不可重现**——卡不一定
还拿得到，时延测量本身是一次性的。所以顺序是 **REPORT.md 定稿 → 删 `.tmp` → 交付**，反过来就只
能凭记忆写板上那几个数字。交付包不含 `.tmp/`、`.scriptor/`、`scriptor/`、
`generated/`、`reports/`、`prototype/` 或 `.DS_Store`；工作区保留原始证据。

交付前检查 `REPORT.md` 和最终交接文档是否包含机器地址、凭证或私有路径，
并把检查记录留在任务证据中。
