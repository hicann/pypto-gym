# 运行源码样例

把本快照的 library、agent 和 kernels 目录作为同一源码身份使用，任务开始前
校验快照索引。数值模拟需要 Torch/NumPy；设备执行还需要分配的板卡、CANN 和
PyPTO-Pro。无需安装 Ascriptor wheel。

从快照根目录运行，并将源码 library 加入 `PYTHONPATH`：

```bash
PYTHONPATH=library python library/examples/api/axpb/main.py
PYTHONPATH=library python library/examples/api/axpb/main.py --launcher pipesim
PYTHONPATH=library python -m ascriptor.cli compile library/examples/api/axpb/kernel.py::axpb --backend cce -o tmp/emit
```

独立公式为一行 64 个 float32 lane 上的 `2*x+y`。`main.py` 既是入口也是比较：它生成输入，
用 `reference.py` 算出该表达式，先在功能模型、再通过 pipe 模型逐 lane 精确比较；`--list` 列出它的
三个 case。emit 是单独一步，只报告产物文件名，不执行 NPU；同一个 kernel 切换 `--backend`
就生成支持的 PTO ISA 或 PyPTO Pro 源码。

本入门样例使用受支持的 A5 facade。decorator 各自绑定设备；导入另一个 facade 不会改变
已有 kernel 的目标。签名使用 `GM[dtype, dimensions]` 和共享字符串 symbol，固定 shape 不要求
额外标量。调用按签名顺序传 tensor，再传显式标量，并使用实际 `OpExec` 返回值。无需运行时猜测
shape 绑定，也不使用旧执行 flag。`<<=` 按内存空间选择搬运；同侧 autosync 不替代跨侧所有权协议。

先读[编写模型](concepts.md)，再选择一条[任务路线](ROUTER.md)。A5 尾部与重复 tile
场景在 [kernel 画廊](../../kernels/README.md)里打开一个 A5 demo，在该目录执行
`python main.py --list` 查看它的 case，并保留其 `metadata.json` 与 `reference.py` 写明的
所有权与精度边界。

primitive 家族从
[API 样例目录](../../library/examples/api/README.md)选择，它生成的[索引](../../library/examples/api/index.json)
逐条列出每个目录的 surface、device、topology、tags 与 case。目录不记录任何结果：今天能做到
什么就是跑它的结果，实测只在 library receipt 里。
