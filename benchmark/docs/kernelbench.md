# KernelBench 数据集

benchmark 默认使用 PyPTO 维护的 KernelBench fork，内容为 PyTorch 原版 case
的支持子集。

- 来源: <https://github.com/zwx2238/KernelBench>
- 分支: `pypto-supported-21fbe`
- 期望布局: `KernelBench/<level>/{N}_{name}.py`
- PyPTO 自维护 case 使用 `KernelBench/pto_case/{N}_{name}.py`

## 下载

唯一保留的下载脚本是：

```bash
bash benchmark/scripts/download_kernelbench.sh
```

默认下载到 `benchmark/.cache/KernelBench/`，运行时 `bench_dir` 留空会使用
`.cache/KernelBench/KernelBench/`。

如果需要自定义下载位置，仍然使用同一个脚本：

```bash
KERNELBENCH_DIR=/data/KernelBench bash benchmark/scripts/download_kernelbench.sh
```

不要新增其他下载脚本；升级数据集分支时，同步修改
`benchmark/scripts/download_kernelbench.sh` 顶部的 `KERNELBENCH_BRANCH`
常量和相关文档。

## YAML 配置

`bench_dir` 需要指向包含 `level1` / `level2` / `level3` / `pto_case` 子目录的
`KernelBench/` 目录。常见错误是把路径指到外层缓存目录，导致运行时找不到
level 目录。

相关字段：

| 字段 | 说明 |
| --- | --- |
| `bench_dir` | 数据集根目录；留空时使用默认缓存位置 |
| `cases` | 必须写明 level，支持完整 stem、序号、闭区间 |
| `limit` | 使用 `level=` 选择整个 level 时的截断数量 |

## 新增仓内 case

仓内自维护 case 的格式要求见 `docs/add-new-case.md`。新增后应通过 pytest
覆盖 loader 行为，再使用公开 benchmark CLI 进行业务验证。
