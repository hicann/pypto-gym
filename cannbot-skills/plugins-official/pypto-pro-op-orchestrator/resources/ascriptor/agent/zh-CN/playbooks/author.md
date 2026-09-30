# 编写单个 kernel

若已读[浓缩起点](../../context/kernel-authoring.zh-CN.md)，它已概括本页的
单 kernel 路径、共同语言与通用 preflight 检查表；无需仅为重复而重读本页。否则读一次
[共同语言](../common-language.md)。按当前任务完成[preflight](../references/authoring-preflight.md)，
只展开命中的小节。

<a id="first-code"></a>
## 从一条完整数据路径开始

下面是 [axpb](../../../library/examples/api/axpb) 的原始 `axpb` 函数：签名 → 分配 → DMA → VF 计算 → 写回。
`axpb_vf` 就在旁边的 `kernel.py` 里完成 `2*x+y`；独立 reference 是 `reference.py`，运行命令在
`main.py`——它既是入口也是比较。
这是固定 `(1,64)` FP32、单 vector core 的起步锚点；扩展形状前重新填写 footprint 和精度契约。

<!-- code-anchor:author:start -->
```python
@kernel(mode="vec", block_dim=1)
def axpb(x: GM[f32, (1, 64)], y: GM[f32, (1, 64)], o: GM[f32, (1, 64)]):
    ub_x = Tensor(DT.float, [1, 64], Position.UB, name="ub_x")
    ub_y = Tensor(DT.float, [1, 64], Position.UB, name="ub_y")
    ub_o = Tensor(DT.float, [1, 64], Position.UB, name="ub_o")
    with auto_sync():
        ub_x <<= x
        ub_y <<= y
        axpb_vf(ub_x, ub_y, ub_o)
        o <<= ub_o
    return o
```
<!-- code-anchor:author:end -->

## 实现流程

1. 从公式与独立 reference 写[任务契约](../references/authoring-contract.md)：cast、括号、alias、shape、launch 数及允许的 host 工作。
2. 从 [API 样例](../../../library/examples/api/README.md)选 primitive，或从 [kernel 画廊](../../../kernels/README.md)选完整算法；画廊生成的[索引](../../../kernels/index.json)逐条列出每个 demo 的 formula、device、topology、tags 与 case 数。
   签名查 [API 入口](../../../library/docs/api/README.md)。
   两个 owner 的目录都不声明任何范围：它教什么、什么时候不要照抄写在 `metadata.json` 里，能不能跑要在你需要答案的那台机器上跑它的 `main.py`。实测只在 library receipt 里，由 release 记录划范围，不由目录划。
3. 推导物理存储、初始化与核间输出归属。普通尾块先读[短检查段](../references/memory-and-tails.md#vector-tail)；
   规约、cast、view、slot 再按 preflight 展开；[跨侧交接](../references/cross-side-handoff.md)有它自己的调用序列。重复混合图使用[流水方法](../references/pipeline-model.md)。
4. 实现最小完整候选，检查 `kernel.ir()`，用 `OpExec` 的实际返回值做独立比较。
   怎么调用、脚本怎么上卡、`ascriptor doctor` 回答什么，见[开发期怎么跑](../references/development-execution.md)。
   按[硬件优先](../runtime-and-maintenance.md#hardware-first)先上板验证完整工作负载，出问题后再缩小 shape/核数诊断；一次往返贵到无法迭代时，那里有专门的分支，[哪一步在哪台机器上跑](../runtime-and-maintenance.md#where-each-step-runs)也在那里定。
   PyPTO-Pro 交付默认显式使用 auto_mutex；manual 只在用户明确要求时生成。一个 launch 的任务将公式留在 kernel 内；host 负责生成输入、分配、dispatch 和比较。
5. 检查完整输出、有效尾块、最后一行/slot、初始化与同核复用，再检查 lowered hazard/event。
   若任务要求重叠，使用同核实际 stage/item 计算区间；若目标是时延，按[成本](../references/roofline.md)选改动。
6. 证明成果可以独立成立：在各 checkout 之外的 scratch 目录里按约定依赖执行，并报告[对应证据层级](../common-language.md#evidence)。
   还没入库时，成果就是你的脚本——`OpExec` 加你自己写的比较。要[进画廊](../references/development-execution.md#admission)才整理成四文件 demo 目录；library API 样例也是同样四个文件；两者都拷贝即可转移，没有导出步骤。
   两者只要还从原来的树里 import 东西，就没做完。保留失败与有意义的负对照，按用户约定的目标和停止条件推进。

使用 PyPTO-Pro 后端时，按[交付同步政策](../runtime-and-maintenance.md#sync-closeout)核对源码装饰器、真机结果和最终选定模式。

以下是固定小 shape、单核的配方。它不是上卡的前置条件：你自己的脚本上卡的方式一样，见
[开发期怎么跑](../references/development-execution.md)。画廊 demo 在它自己的目录里走同样三个阶段
（`python main.py`、`python main.py --launcher pipesim`、`python main.py --launcher board`），
在那里每一阶段证明的东西也一样：一个阶段能证明什么，和完整工作负载的第一次 launch 花在
哪里，是两个问题；上面的代价分支生效时就按这个顺序走。

emit 放在最前：几秒钟、不需要卡，sim 接受而后端拒绝的写法在你基于它继续写之前就会暴露——对 PyPTO-Pro
这是[那一页的第一条规则](../references/pypto-pro.md#emit-first)。
样例没有 `emit` 子命令，emit 用 library CLI，点名 kernel 和后端：

```bash
ascriptor compile examples/api/axpb/kernel.py::axpb --backend cce -o tmp/agent-author/emit
```

然后在 library checkout 中跑两个模型。`main.py` 自己生成输入、算独立 reference 并逐 case 比较，所以没有单独的 `reference` 步骤：

<!-- checked-command: library -->
```bash
python examples/api/axpb/main.py --launcher sim
python examples/api/axpb/main.py --launcher pipesim
```

功能结果应完整精确，pipe 检查应无 imbalance/hazard/deadlock；详见
[实际执行配方](../runtime-and-maintenance.md)。失败进入[调试](debug.md)，不确定 primitive 用生成输入的小 probe 定位。
Attention 工作另按[专题](../references/attention-authoring.md)连接数学、布局和精确 canonical case。
