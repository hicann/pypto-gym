# 内置 KernelBench 数据集

本目录包含随 `pypto-gym` 仓库内置的完整 KernelBench case 集。

- 来源: `https://github.com/zwx2238/KernelBench`
- 来源分支: `pypto-supported-21fbe`
- 本次导入 commit: `5bb8dda`
- License: 见 `LICENSE`

case 文件保留上游原始目录和原始编号，不再按 `pypto.yaml` 重新编号。

| Level | Case 数量 |
| --- | --- |
| `level1` | 60 |
| `level2` | 35 |
| `level3` | 22 |
| `level4` | 20 |
| `pto_case` | 44 |

选择全部内置 case 可使用空 selector：

```yaml
cases: "level1=;level2=;level3=;level4=;pto_case="
```
