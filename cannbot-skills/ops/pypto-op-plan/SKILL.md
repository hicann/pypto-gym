---
name: pypto-op-plan
description: Requirement planning — structurally-similar example search and feasibility setup.
---

# PyPTO Complex Kernel — Planning

> 资料获取统一使用 skill `pypto-docs-search`：按需搜索算子 API 文档、参考实现与 golden 等文件/目录/内容。

## Requirements and environment (optional)

**If requirements are unstructured:** structure them into `SPEC.md` (task summary lives in SPEC.md, not MEMORY.md).

**SPEC.md front matter 的 `supported_dtypes`** 仅收录 P0 输入/输出 tensor 的 dtype 集合，必须与 `REQUIRE.md` 声明一致；权重 buffer 与中间计算 dtype 不写入。模板与生成指引见 skill `pypto-intent-understand`（`templates/spec-template.md`）。

**If environment issues arise:** confirm with the caller whether to ignore them and continue planning.

## Find structurally similar examples

Search for existing kernels with similar structure —— 用 `pypto-docs-search` 搜索 "<kernel type>" 的 算子参考实现与 golden 用法；另可扫描当前工作树 `grep -rn "<kernel type>" custom/`。

Note: golden/用法 is an **API-usage reference only**, not the production implementation standard — its simplified forms (e.g. `pypto.Tensor([])`) may violate lint / gates. When it conflicts with lint, lint takes priority; do NOT cite such a sample to declare lint a false positive.


