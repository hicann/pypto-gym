# examples 占位符约定

本目录下每个 `<op>.md` 是 kernel **参考骨架**，用于展示"哪些轴 loop、哪些轴整块"的切分模式，**不保证逐个编译通过**（无 NPU 环境无法验证）。所有骨架共用以下占位符：

| 占位符 | 含义 |
|--------|------|
| `sl` | 输入 shape 列表，如 `[B, S, D]` |
| `ol` | 输出 shape 列表 |
| `pypto_dtype` | 元素 dtype，如 `pypto.DT_FP32` |
| `batch` | 被 loop 的外层轴长度（通常 `sl[0]`） |
| `inner` | 单次迭代处理的内层 shape，如 `sl[1:]` |
| `inner_out` | 单次迭代的输出内层 shape（末轴长度可能变化，如 glu/diff/topk） |
| `zeros_in` / `zeros_out` | 与内层维数等长的 0 偏移列表，用于 `view`/`assemble` 定位 |
| `tile` | `set_vec_tile_shapes` 的 tile 形状 |
| `half` | last-dim 折半长度（rope/glu 等用） |
| `k` | topk 的取数 k（topk） |
| `scale` | attention 缩放系数（attention） |
| `rep` | 复制次数（repeat / repeat_interleave） |
| `idx_inner` / `idx_zeros` | 索引张量单迭代内层 shape / 0 偏移（gather/index_* 等） |
| `gather_dim` | 被查表/被选轴（gather/index_select，代码中以形参 `dim` 体现） |
| `out_interleaved` | sort 交错输出（值+索引）shape（sort） |
| `num_tiles` / `tile_len` | 生成类沿输出轴切 tile 的分块数 / 每块长度（arange/linspace 等） |

## 最小可运行 setup 模板

把占位符替换为具体值后即可 jit：

```python
import pypto

B, D = 8, 128
sl, ol = [B, D], [B, 1]
pypto_dtype = pypto.DT_FP32
batch, inner, inner_out = B, [D], [1]
zeros_in = zeros_out = [0]
```

## Note 约定

- 有 loop 的骨架：Note 说明"loop 了哪些轴 / 哪些轴整块 / 原因"。
- 无 batch-row loop 的骨架（如 cube、sort、生成/索引类）：Note 说明改用的切分方式（cube tiling / 输出轴 loop / 整 tile），**不得再使用 "not applicable"**。
- 涉及 `sigmoid` 的骨架：`sigmoid` 仅支持 FP32，非 FP32 需先 `cast`。
