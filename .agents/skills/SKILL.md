---
name: pypto-static-check-repire
description: Git 仓库 Python 代码静态规范问题修复，遵从门禁检测规则，依据 Excel 报表定位并修复不规范问题，同步记录修改日志。
---

# 技能：代码静态规范问题批量修复（Git 仓库 + 门禁检测适配）

## 技能名称
Git 仓库 Python 代码静态规范问题修复（遵从门禁检测规则 + 日志溯源）

## 技能适用场景
当前工作目录为 `pypto-gym` Git 仓库，门禁静态代码检测不通过，依据导出 Excel 报表 `CANN_pypto-gym_全量导出（未解决）_20260629112325.xlsx` 定位并修复代码不规范问题，严格按过滤规则执行，同步记录修改日志。

## 前置依赖
1. 工作目录：`pypto-gym` 根目录（Git 本地仓库）
2. 问题清单文件：`CANN_pypto-gym_全量导出（已忽略）_20260612124040.xlsx`
3. 日志文件：仓库根目录 `static.md`（用于记录所有修改行为）
4. 工具：Excel 查看工具、代码编辑器、Git（仅环境校验，本次不做提交）

## 核心执行规则（强制遵守）
1. 依据 Excel 表头 `文件路径`、`文件名称`、`行号` 定位代码位置，对照 `问题描述`、`编码规范` 判断修复动作；
2. 判定无需修改的场景，直接跳过该条问题，不做任何代码变更；
3. 若目标文件路径属于 `src/pypto_gym/transformers/` 目录，直接跳过，不修改；
4. 问题类型为 `超大函数[PYTHON]`、`超大深度函数[PYTHON]`、`超大圈复杂度[PYTHON]` 三者其一，且问题文件为 kernel 函数相关文件，直接跳过，不修改；
5. 所有代码修改、跳过原因、文件位置，均逐条记录至 `static.md`，用于问题溯源。

## 执行步骤

### 步骤 1：环境与文件校验
1. 确认当前终端 / 工作目录为 `pypto-gym` Git 仓库根目录；
2. 检查 Excel 问题报表 `CANN_pypto-gym_全量导出（未解决）_20260629112325.xlsx` 存在并可正常打开；
3. 检查仓库根目录 `static.md`，文件不存在则新建空文件，存在则在文件末尾追加日志（不覆盖原有内容）。

### 步骤 2：逐行解析 Excel 问题清单
1. 打开 Excel 文件，按行遍历每条检测结果，提取关键字段：
   - `文件路径`、`文件名称`、`问题行号`
   - `问题类型`、`问题描述`、`标准编码规范`
2. 如果找不到对应代码路径, 可直接按文件名搜索。
3. 针对单条问题，依次执行规则过滤判断：
   - **判断 1**：结合行号 + 代码上下文，判断该行无需修改,如果确认无问题 → 跳过，记录「无需修改」；
   - **判断 2**：文件路径匹配 `src/pypto_gym/transformers/` → 跳过，记录「目录豁免，不修改」；
   - **判断 3**：问题属于超大函数/超大深度函数/超大圈复杂度 且文件为 kernel 函数文件 → 跳过，记录「kernel 函数 + 复杂度问题，豁免不修改」；kernel函数说明:带有@pypto.frontend.jit装饰器的为kernel函数,如:
@pypto.frontend.jit(
  pass_options={},
  runtime_options={
      "stitch_function_max_num": 512,
      "device_sched_mode": 3,
  },
)
def compressor_ratio_4_kernel(
    x: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    kv_state_total: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    score_state_total: pypto.Tensor([pypto.STATIC, pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    kv_block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    score_block_table: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_INT32),
    sin: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    cos: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    wkv: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    wgate: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_BF16),
    ape: pypto.Tensor([pypto.STATIC, pypto.STATIC], pypto.DT_FP32),
    weight: pypto.Tensor([pypto.STATIC], pypto.DT_FP32),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    kv_state_out: pypto.Tensor([...], pypto.DT_FP32),
    score_state_out: pypto.Tensor([...], pypto.DT_FP32),
    start_pos_dy: pypto.Tensor([...], pypto.DT_INT32),
    ratio, 
    rope_head_dim
    )
    除kernel以外的函数可以修改超大函数/超大深度函数/超大圈复杂度问题
   - 以上判断均不命中 → 进入代码修复环节。

### 步骤 3：代码修复操作（仅非豁免问题执行）
1. 根据「文件路径 + 文件名称 + 行号」，在仓库中打开对应代码文件；
2. 对照 Excel 中编码规范，修正当前行 / 对应代码的不规范问题；
3. 保存文件，检查语法无报错、无新增问题。

### 步骤 4：日志写入（必做，每条问题都记录）
在 `static.md` 追加单条日志，统一格式：

```plaintext
## 静态检测修复记录 - 日期时间
- 文件路径：{Excel内文件完整路径}
- 文件名称：{文件名}
- 问题行号：{行号}
- 原始问题：{问题描述}
- 处理结果：【跳过/已修复】
- 处理原因：{豁免原因 / 修复简述}