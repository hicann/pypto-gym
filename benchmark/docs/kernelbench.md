# KernelBench 数据集

benchmark 默认使用仓内内置的完整 KernelBench case 集，不再下载外部
KernelBench 仓库。

- 内置路径: `benchmark/KernelBench/`
- 期望布局: `KernelBench/<level>/{N}_{name}.py`
- 来源: `https://github.com/zwx2238/KernelBench` 的 `pypto-supported-21fbe` 分支
- 导入 commit: `5bb8dda`
- 编号策略: 保留上游原始目录和原始编号

## 内置规模

| Level | Case 数量 |
| --- | --- |
| `level1` | 60 |
| `level2` | 35 |
| `level3` | 22 |
| `level4` | 20 |
| `pto_case` | 44 |

合计 181 个 case。

## YAML 配置

`bench_dir` 需要指向包含 level 子目录的 `KernelBench/` 目录。留空时自动使用
内置路径 `benchmark/KernelBench/`。

相关字段：

| 字段 | 说明 |
| --- | --- |
| `bench_dir` | 数据集根目录；留空时使用内置 `benchmark/KernelBench/` |
| `cases` | 必须写明 level，支持完整 stem、序号、闭区间；`level=` 表示选择该 level 全部 case |
| `limit` | 使用 `level=` 选择整个 level 时的截断数量 |

选择全部内置 case：

```yaml
cases: "level1=;level2=;level3=;level4=;pto_case="
```

单 case 示例 ReLU 使用上游原始编号：

```yaml
cases: "level1=19_ReLU"
```

## 新增或替换数据集

默认运行不需要下载 KernelBench。若本地实验需要使用自定义 case 集，可通过
`bench_dir` 指向另一个符合相同布局的目录，并在 `cases` 中显式选择对应 level
和编号。新增仓内或外部 case 的格式要求见 `docs/add-new-case.md`。
