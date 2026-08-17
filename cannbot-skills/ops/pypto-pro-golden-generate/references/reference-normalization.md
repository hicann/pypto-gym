# 既有参考规范化

当用户提供 PyTorch 或 NumPy 参考实现时使用本流程。先确认参考源路径并保持原文件只读，再按 [../SKILL.md](../SKILL.md) 生成 NPU golden 骨架，把经等价核对的数学逻辑填入骨架保留的 TODO。若参考源正是任一目标交付文件，候选骨架必须生成到临时目录；等价验证通过并取得覆盖确认后才替换目标，不能先覆盖 oracle。目标是得到语义等价、可验证且便于后续实现对照的 golden；不在此阶段预测 Module 数量、划分 Module 边界或设计 kernel。

## 1. 选择参考源

按以下顺序选择可执行且语义最完整的参考：

1. PyTorch forward/backward 参考；
2. NumPy 参考；
3. 没有可用参考时，才根据 SPEC 数学公式自行实现。

在新写数学逻辑前，先搜索仓内已有实现和 PyPTO-Pro API 文档：

```bash
rg -n "<operator name>" examples/ custom/ models/ "$PYPTO_DEVKIT_DIR/docs/pypto_pro/api/"
```

记录最终参考文件、入口函数和选择理由。多个来源冲突时，以 SPEC 为合同并保留冲突证据；不要静默拼接不同语义。

## 2. 审计并规范化

逐项检查：

- 隐式多轴 broadcast、隐藏的 reshape/transpose/layout 变化；
- 不透明库调用和难以核对的复合调用；
- host-side 便利逻辑、复杂控制流和高维操作；
- 未显式说明的 dtype 提升、降精度或 accumulator 精度；
- shape、索引范围和 tensor 间依赖未写清的中间值。

对不明确的操作读取对应 PyPTO-Pro API 文档。规范化时：

- 保留数学语义，不机械保留源代码语法；
- 显式写出关键 shape、dtype 转换、broadcast 轴和 layout 变化；
- 将多步隐式操作拆为可单独核对的中间 tensor，并使用有意义的名称；
- 使用 `torch.transpose(t, dim0, dim1)` 或 `tensor.transpose(dim0, dim1)`，不使用 `.T` / `.t()` 隐藏转置轴；
- 所有 matmul 输入先转 FP32 累加；
- 不因后续 kernel 可能采用某种结构而改变 golden 的数学结果。

默认采用一次性处理完整输入的 full computation。只有 SPEC 本身定义分块、窗口、递归状态或 partial accumulation，且需要验证 tile 边界语义时，才增加 tiled 版本。若同时保留 full 与 tiled，两者必须在 dtype 对应容差内等价。

## 3. 在 golden 头部记录数学清单

在规范化后的 `{op}_golden.py` 文件级 docstring 或紧随其后的注释中维护一份简短清单，使数学步骤和 shape 变化可追溯：

```text
Golden operation inventory:
1. matmul(q, k^T): [B,H,T,K] x [B,H,K,T] -> [B,H,T,T]
2. softmax(scores, dim=-1): [B,H,T,T] -> [B,H,T,T]
3. matmul(probs, v): [B,H,T,T] x [B,H,T,K] -> [B,H,T,K]
```

清单只描述当前 golden 的数学操作和 shape/dtype 变化，不包含未来 Module、PyPTO-Pro 实现行号或状态字段。CPU golden 保持同一数学步骤。

## 4. 证明与原始参考等价

保留原始参考作为独立 oracle，使用相同输入分别执行原始与规范化实现。至少覆盖：

- 固定 random seed；
- 小 shape、代表性/P0 shape 和边界 shape；
- SPEC 支持的 dtype；
- 输出 shape、NaN/Inf；
- dtype-aware 的 `assert_allclose`；
- full/tiled 两种实现同时存在时的逐 case 等价性。

输入具有索引、状态或多 tensor 依赖时，复用同一组合法输入，不为两侧分别随机生成。任何不匹配都必须先定位并修复；不得通过改 SPEC、放宽到无意义容差或跳过 case 继续。

## 5. Freeze 并进入标准交付

将规范化逻辑填入已生成骨架并完成等价验证后，在 golden 文件头部记录参考源、验证命令和 `frozen` 状态。此后把规范化实现作为唯一 golden 来源；除非有可复现证据证明其语义错误，否则不要改动。

随后返回 [../SKILL.md](../SKILL.md) 的 NPU/CPU 生成与直接执行流程，交付并验证两份 golden。Freeze 不替代这两份文件各自的 exit-code 门禁。
