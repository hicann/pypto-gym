# 开发期怎么跑

写 kernel 时用 `OpExec`。demo 目录的四个文件和画廊的批量 runner 是**入库和批量测试**用的，
不是开发路径——它们存在是为了形态统一维护、防漂移、跑整套回归。开始写代码之前不需要先
对接它们。

## 本地

```python
from ascriptor.runtime import OpExec

out = OpExec(kernel, launcher="sim")(q, k, v, o, B, T)       # 功能
out = OpExec(kernel, launcher="pipesim")(q, k, v, o, B, T)   # event/hazard/deadlock
```

张量按签名顺序在前，显式标量在后。**签名里的每一个张量都要传，输出也不例外**——上面的 `o`
是你自己分配并传进去的缓冲，runtime 不会替你建；漏传会报 `<name> needs N tensor
argument(s), got M`。输出默认被污染、以及 `seed_outputs=True` 的存在，也是同一个原因：
launch 之前缓冲里是什么由你负责。返回的就是 torch tensor，用它做比较。

`pipesim` 会连带检查 event balance、hazard 和 deadlock。`OpExec` 在任一不过时**抛错**，
demo 的 `--launcher pipesim` 也一样，所以这两条路不会给出不同结论。但直接调用
`simulate()`——那是排查用的入口，不是 launcher——是把结果**返回**在 result 上；
不看 `result.hazards` 的运行器会把有 hazard 的 kernel 报成通过。判据相同，由谁抛出不同。

**先 emit，再模拟。** `compile_kernel` 只要几秒、不需要卡，`sim` 接受而后端拒绝的写法会在
这一步暴露，而不是等你基于它写下去之后。目标是 PyPTO-Pro 时这是[那一页的第一条规则](pypto-pro.md#emit-first)，
[编写 playbook](../playbooks/author.md) 也把它放在同一条阶梯的最前面。[runtime](../runtime-and-maintenance.md#hardware-first)
里的硬件优先规则排的是**卡**和模拟器的先后，不影响这一条：emit 不产生任何往返，无论如何都在最前。

<a id="on-hardware"></a>
## 上板

在持有分配板卡的机器上运行脚本，把本快照的 `library/` 加入 `PYTHONPATH`。
同一个 `OpExec` 使用 `launcher="board"`（CCE）或 `launcher="pypto"`
（PyPTO-Pro）直接运行本机卡。输入、独立参考、执行和比较在同一进程完成。

## 六个 launcher

`OpExec` 只认这六个，**每一个都在调用它的那台机器上跑**。没有"远端"那一类。

| launcher | 在哪 | 通过能证明什么 |
|---|---|---|
| `sim` | 本机，进程内 | 受支持的算术与 Surface 语义；不碰后端 |
| `pipesim` | 本机，进程内 | lowered 的 event balance / hazard / deadlock，三项任一不过即经 `OpExec` 抛错 |
| `aclnn` | 本机的卡 | 厂商编译加实际执行（CCE 走 aclnn 自定义算子工程） |
| `cannsim` | 本机 | 厂商模拟器执行，与宿主 pipesim 分开记录 |
| `board` | 本机的卡 | 同一个工程在这台机器上构建并运行 |
| `pypto` | 本机的卡 | 生成的 PyPTO-Pro 源码在这台机器的卡上执行 |

后三个需要这台机器**说明自己是板子**（见下）；在工作机上它们会拒绝，`sim` 和 `pipesim`
不受影响。`compile_kernel(entry, backend=...)` 不是 launcher——它只生成源码，不执行。
它和 `OpExec` 来自同一个模块（`from ascriptor.runtime import compile_kernel`）；包根目录没有导出它。

## 板上要先具备什么

两件事，**配机器时做一次**：

1. **源码导入**：从本快照运行，把 `library/` 加入 `PYTHONPATH`；运行前核对
   `ascriptor.__file__` 与快照根目录的 `sources.json`。
2. **机器说明自己是谁**：工作目录放一份配置，其中描述本机的条目标 `"local": true`；
   环境脚本 `export ASCRIPTOR_BOARDS` 指向它。

条目至少要有 `workspace`、`env_script`、`cube_cores`，以及 `"local": true`。
`cube_cores` 填这张卡的 **AIC 核数**（Ascend950PR 是 28，不是 device profile 里的 32）——
漏了它 `pypto` 会拒绝执行，理由见 [PyPTO-Pro 专题](pypto-pro.md#核数)。

`ASCRIPTOR_BOARD` 在一台机器有多个条目（比如每卡一个）时指定用哪个。

**两份配置是两个文件，别名也各不相同。** 工作机那份列出它能连到的每一台盒子；盒子那份只
描述它自己，常常只有一个条目。你在这边用的别名在那边并不存在，所以不要把它带过去——读盒子
自己的配置，或者让 `ascriptor doctor` 告诉你它撞上的是哪一种拒绝。

**要把分配给你的别名对应到一台机器，先问 doctor 配置在哪，再读那个条目。** 没有页面写死这个
文件名，因为它由 `ASCRIPTOR_BOARDS` 决定；`doctor` 的 `boards` 那一行会打印解析后的路径，
条目里有主机、工作目录和卡号。查法就这一条，没有另外的注册表。

**板上的工作目录是共享的。** 两个会话挑中同一个显而易见的目录名时，会在"拷过去"和"跑起来"
之间互相覆盖，失败看上去像是你自己的代码。给任务目录起一个撞不上的名字——任务名加时间戳
——并把所有输出都放在它下面。

## 先问 doctor

```sh
PYTHONPATH=<快照根目录>/library python -m ascriptor.cli doctor
```

一次答完：用的哪个解释器、**实际 import 的** ascriptor 在哪、版本多少、torch/numpy/pypto_pro
在不在、这台机器算不算板子、卡的 AIC/AIV 数。在工作机上它会明说 `board` 和 `pypto` 会被拒绝，
而 `sim` 和 `pipesim` 不受影响。

<a id="admission"></a>
## 什么时候才去补另外三个文件

当这个 kernel 要**进画廊**时。那时脚本才变成恰好四个文件的目录——`kernel.py`、
`reference.py`、`main.py`、`metadata.json`——并按[编写 playbook](../playbooks/author.md)
的证据阶梯走。在此之前，`OpExec` 加一个自己写的比较就够了。

进画廊的判据是在 kernels checkout 跑 `python tools/run_all.py` 通过：它让每个 demo 的
`main.py` 在自己的进程、自己的工作目录里跑，因为读者就是这么跑它的。在那里不过的 demo
就是坏的，`metadata.json` 写什么都不算。

目标是 PyPTO-Pro 的任务不走这条路：它交的是一个交付包，见[交付区](pypto-pro.md#delivery-area)。
