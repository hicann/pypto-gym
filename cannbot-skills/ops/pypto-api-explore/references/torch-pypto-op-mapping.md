# Torch ↔ Pypto 算子对标手册

## 对标映射表

> **参考实现说明**：每个算子链接到 `examples/` 下的 kernel 参考文档，包含 `@pypto.frontend.jit` 装饰的 kernel 函数代码。无 loop kernel 的算子在文档中标注原因。

### A 类 - 一一对应（52 个）

Torch 算子有直接对应的 Pypto API，可以 1:1 替换。

#### 形状变换（7 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 1 | `cat` | `concat` | [cat.md](../examples/cat.md) |
| 2 | `unsqueeze` | `unsqueeze` | [unsqueeze.md](../examples/unsqueeze.md) |
| 3 | `view` | `view` | [view.md](../examples/view.md) |
| 4 | `reshape` | `reshape` | [reshape.md](../examples/reshape.md) |
| 5 | `transpose` | `transpose` | [transpose.md](../examples/transpose.md) |
| 6 | `expand` | `expand_clone` | [expand.md](../examples/expand.md) |
| 7 | `permute` | `permute` | [permute.md](../examples/permute.md) |

#### 数学运算（10 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 8 | `sum` | `sum` | [sum.md](../examples/sum.md) |
| 9 | `pow` | `pow` | [pow.md](../examples/pow.md) |
| 10 | `rsqrt` | `rsqrt` | [rsqrt.md](../examples/rsqrt.md) |
| 11 | `exp` | `exp` | [exp.md](../examples/exp.md) |
| 12 | `log` | `log` | [log.md](../examples/log.md) |
| 13 | `cos` | `cos` | [cos.md](../examples/cos.md) |
| 14 | `sin` | `sin` | [sin.md](../examples/sin.md) |
| 15 | `sqrt` | `sqrt` | [sqrt.md](../examples/sqrt.md) |
| 16 | `div` | `div` | [div.md](../examples/div.md) |
| 17 | `mul_` | `mul` | [mul_.md](../examples/mul_.md) |

#### 归约操作（3 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 18 | `cumsum` | `cumsum` | [cumsum.md](../examples/cumsum.md) |
| 19 | `max` | `amax` | [max.md](../examples/max.md) |
| 20 | `prod` | `prod` | [prod.md](../examples/prod.md) |

#### 矩阵运算（1 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 21 | `matmul` | `matmul` | [matmul.md](../examples/matmul.md) |

#### 激活函数（3 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 22 | `softmax` | `softmax` | [softmax.md](../examples/softmax.md) |
| 23 | `sigmoid` | `sigmoid` | [sigmoid.md](../examples/sigmoid.md) |
| 24 | `tanh` | `tanh` | [tanh.md](../examples/tanh.md) |

#### 索引操作（5 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 25 | `gather` | `gather` | [gather.md](../examples/gather.md) |
| 26 | `index_select` | `index_select` | [index_select.md](../examples/index_select.md) |
| 27 | `scatter` | `scatter` | [scatter.md](../examples/scatter.md) |
| 28 | `scatter_` | `scatter_` | [scatter_.md](../examples/scatter_.md) |
| 29 | `argmax` | `argmax` | [argmax.md](../examples/argmax.md) |

#### 比较操作（5 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 30 | `where` | `where` | [where.md](../examples/where.md) |
| 31 | `clamp` | `clip` | [clip.md](../examples/clip.md) |
| 32 | `gt` | `gt` | [gt.md](../examples/gt.md) |
| 33 | `eq` | `eq` | [eq.md](../examples/eq.md) |
| 34 | `logical_not` | `logical_not` | [logical_not.md](../examples/logical_not.md) |

#### 特殊操作（6 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 35 | `topk` | `topk` | [topk.md](../examples/topk.md) |
| 36 | `one_hot` | `one_hot` | [one_hot.md](../examples/one_hot.md) |
| 37 | `argsort` | `argsort` | [argsort.md](../examples/argsort.md) |
| 38 | `triu` | `triu` | [triu.md](../examples/triu.md) |
| 39 | `tril` | `tril` | [tril.md](../examples/tril.md) |
| 40 | `sort` | `sort32` + `mrgsort` | [sort.md](../examples/sort.md) |

#### 张量创建（2 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 41 | `arange` | `arange` | [arange.md](../examples/arange.md) |
| 42 | `pad` | `pad` | [pad.md](../examples/pad.md) |

#### 类型转换（2 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 43 | `to`(dtype) | `cast` | [to.md](../examples/to.md) |
| 44 | `type_as` | `cast` | [type_as.md](../examples/type_as.md) |

#### 其他（8 个）

| # | Torch 算子 | Pypto API | 参考实现 |
|---|-----------|-----------|---------|
| 45 | `abs` | `abs` | [abs.md](../examples/abs.md) |
| 46 | `add` | `add` | [add.md](../examples/add.md) |
| 47 | `sub` | `sub` | [sub.md](../examples/sub.md) |
| 48 | `mul` | `mul` | [mul.md](../examples/mul.md) |
| 49 | `neg` | `neg` | [neg.md](../examples/neg.md) |
| 50 | `relu` | `relu` | [relu.md](../examples/relu.md) |
| 51 | `maximum` | `maximum` | [maximum.md](../examples/maximum.md) |
| 52 | `minimum` | `minimum` | [minimum.md](../examples/minimum.md) |


---

### B 类 - 组合对标（40 个）

Torch 算子需要通过多个 Pypto 算子组合实现。

#### 形状变换（11 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 53 | `flatten` | `reshape` | [flatten.md](../examples/flatten.md) |
| 54 | `squeeze` | `reshape` / `view` | [squeeze.md](../examples/squeeze.md) |
| 55 | `t` | `transpose` | [t.md](../examples/t.md) |
| 56 | `expand_as` | `expand_clone` | [expand_as.md](../examples/expand_as.md) |
| 57 | `movedim` | `permute` | [movedim.md](../examples/movedim.md) |
| 58 | `split` | `view` (切片) | [split.md](../examples/split.md) |
| 59 | `chunk` | `view` (切片) | [chunk.md](../examples/chunk.md) |
| 60 | `unbind` | `view` + `reshape` | [unbind.md](../examples/unbind.md) |
| 61 | `stack` | `unsqueeze` + `concat` | [stack.md](../examples/stack.md) |
| 62 | `repeat` | `unsqueeze` + `expand_clone` + `reshape` | [repeat.md](../examples/repeat.md) |
| 63 | `repeat_interleave` | `unsqueeze` + `expand_clone` + `reshape` | [repeat_interleave.md](../examples/repeat_interleave.md) |

#### 数学运算（1 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 64 | `outer` | `unsqueeze` + `mul` | [outer.md](../examples/outer.md) |

#### 归约操作（4 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 65 | `mean` | `sum` + `div` | [mean.md](../examples/mean.md) |
| 66 | `norm` | `mul` + `sum` + `sqrt` | [norm.md](../examples/norm.md) |
| 67 | `all` | `where` + `sum` + `gt` | [all.md](../examples/all.md) |
| 68 | `any` | `where` + `sum` + `gt` | [any.md](../examples/any.md) |

#### 矩阵运算（5 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 69 | `nn.Linear` | `matmul` + `add` | [linear.md](../examples/linear.md) |
| 70 | `linear` | `matmul` + `add` | [linear.md](../examples/linear.md) |
| 71 | `nn.Conv2d` | `conv` + TileL1/L0 | [conv2d.md](../examples/conv2d.md) |
| 72 | `nn.Conv3d` | `conv` + TileL1/L0 | [conv3d.md](../examples/conv3d.md) |
| 73 | `scaled_dot_product_attention` | `matmul` + `amax` + `sub` + `exp` + `sum` + `div` | [attention.md](../examples/attention.md) |

#### 激活函数（4 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 74 | `silu` | `sigmoid` + `mul` | [silu.md](../examples/silu.md) |
| 75 | `softplus` | `exp` + `add` + `log` | [softplus.md](../examples/softplus.md) |
| 76 | `nn.GELU` | `mul` + `tanh` + `add` | [gelu_tanh.md](../examples/gelu_tanh.md) |
| 77 | `glu` | `view` + `sigmoid` + `mul` | [glu.md](../examples/glu.md) |

#### 索引/查表（1 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 78 | `nn.Embedding` | `gather` + padding | [embedding.md](../examples/embedding.md) |

#### 归一化（1 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 79 | `nn.LayerNorm` | `sum` + `div` + `sub` + `mul` + `sqrt` | [layer_norm.md](../examples/layer_norm.md) |

#### 索引操作（3 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 80 | `masked_fill` | `gt` + `where` | [masked_fill.md](../examples/masked_fill.md) |
| 81 | `masked_fill_` | `gt` + `where` | [masked_fill_inplace.md](../examples/masked_fill_inplace.md) |
| 82 | `masked_scatter` | `where` + `cumsum` + `cast` + `gather` | [masked_scatter.md](../examples/masked_scatter.md) |

#### 特殊操作（3 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 83 | `roll` | `index_select` + `concat` | [roll.md](../examples/roll.md) |
| 84 | `diff` | `view` + `sub` | [diff.md](../examples/diff.md) |
| 85 | `bincount` | `one_hot` + `sum` | [bincount.md](../examples/bincount.md) |

#### 张量创建（2 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 86 | `eye` | `arange` + `one_hot` + `cast` | [eye.md](../examples/eye.md) |
| 87 | `linspace` | `arange` + `mul` | [linspace.md](../examples/linspace.md) |

#### 其他（5 个）

| # | Torch 算子 | Pypto 组合方案 | 参考实现 |
|---|-----------|---------------|---------|
| 88 | `reciprocal` | `rsqrt` + `mul` | [reciprocal.md](../examples/reciprocal.md) |
| 89 | `index_add_` | `add` (近似) | [index_add_.md](../examples/index_add_.md) |
| 90 | `rope` | `view` + `neg` + `concat` + `mul` + `add` | [rope.md](../examples/rope.md) |
| 91 | `l2_norm` | `mul` + `sum` + `rsqrt` | [l2_norm.md](../examples/l2_norm.md) |
| 92 | `moe_routing` | `matmul` + `sigmoid` + `topk` + `sum` + `div` | [moe_routing.md](../examples/moe_routing.md) |

---
