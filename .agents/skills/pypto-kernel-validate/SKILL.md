---
name: pypto-kernel-validate
description: 校验一个声称由 PyPTO 开发的算子产物 — 反作弊 (脚本机械检测 + LLM 语义审阅) 加精度与性能验证, 输出统一 JSON 报告. 调用方: pypto-gym benchmark verifier.
---

# PyPTO Kernel Validate Skill

> 给定一个算子产物目录, 判定该实现是否 (1) 真正使用 PyPTO 而非作弊, (2) 精度通过, (3) 性能符合要求.
> 整套流程以 **JSON 报告** 结束, 不在对话中堆叠人类总结.

## 适用场景

任意 agent 工作流生成 PyPTO 算子之后, 在交付前调用本 skill 做最终把关. 典型调用方:

- KernelBench 桥接层 (`pypto-gym/benchmark/`) 在 verify 阶段.
- 外部团队的算子 agent, 把产物丢给本 skill 做合规校验.

## 输入约定 (由调用方在 prompt 中传入)

| 字段 | 必需 | 说明 |
|---|---|---|
| `op_name` | 是 | 算子名 (与核心实现文件 `<op>_impl.py` 前缀一致) |
| `op_dir` | 是 | 算子产物目录绝对路径 |
| `task_desc_file` | 是 | KernelBench 风格 task_desc.py 绝对路径 (含 `Model` / `get_inputs` / `get_init_inputs`) |
| `output_dir` | 是 | 报告输出目录, skill 会写出 `skill_report.json` |
| `mode` | 否 | `correctness` (默认) / `performance` / `full` |
| `device_id` | 否 | NPU 卡号, 默认 0 |
| `arch` | 否 | 默认 `ascend910b4` |
| `log_dir` | 否 | KernelVerifier verify/profile 临时工作目录根; 缺省使用 verifier 默认值 |
| `verify_timeout` | 否 | 单次验证子进程超时, 秒, 默认 300 |
| `verify_rtol` | 否 | verify 精度比较 rtol; 缺省使用 verifier 默认值 |
| `verify_atol` | 否 | verify 精度比较 atol; 缺省使用 verifier 默认值 |
| `keep_artifacts` | 否 | 是否保留 verifier 的 verify/profile 临时工作目录; 默认 `false` 自动清理 |

## 工作流 (必须严格按以下 4 步执行, 不得跳步)

在进入 Step 1 前, 先做一次设备准备. 这部分判断与重试由你这个 validator agent 自己完成, 不依赖 PyPTO 仓内脚本:

- 若 prompt 传入了 `device_id`, 优先使用这张卡.
- 若需要判断设备可用性, 使用系统工具或 `torch_npu` 在当前 pypto-gym benchmark 环境中探测, 不调用 PyPTO 源码仓里的 helper 脚本.
- 仅对“潜在卡问题”做最多 3 次重试; 由你根据日志与现象自行判断是否属于卡问题.
- 下列情况不按卡问题重试: 编译报错、correctness 结果是精度差异、performance 只是测速结果不够好.
- 下列情况按潜在卡问题处理并允许重试: runtime error, 或 performance 阶段 `base/gen` 任一未采集到.
- 如果切卡或重试过, 必须在最终 `skill_report.json` 的 `final_reasoning` 中说明尝试了哪些 device、重试了几次、依据是什么.

### Step 1: 脚本机械检测 (cheat_detector)

调用统一 verifier CLI 的 `cheat-check` 子命令:

```bash
python -m benchmark.verifier cheat-check \
  "<op_dir>" \
  --op-name "<op_name>" \
  --json-out "<output_dir>/cheat_check_script.json"
```

读 `cheat_check_script.json`, 关注 `verdict` 与 `checks[*]`:

- `verdict == "cheat"` —— 机械层命中了强规则, **按 warning 处理, 不得阻塞 Step 2 / Step 3**。因为脚本规则存在误报可能, 你必须继续完成语义审阅和精度/性能验证, 再综合裁定.
- `verdict == "suspicious"` —— 软警告, 你要在 Step 2 重点核查这些项.
- `verdict == "pass"` —— 机械层 OK, Step 2 可正常进行.

机械检测覆盖范围 (你不必在 Step 2 重复同样的工作, 但要交叉印证):

- `import_pypto`: 是否存在 `import pypto` / `from pypto`.
- `has_jit`: 是否至少存在一个 `@pypto.frontend.jit` 装饰器或 `pypto.frontend.jit(...)` 调用.
- `forbidden_text_patterns`: 是否出现 testing-only / fallback / workaround / TODO use pypto / bypass pypto 等可疑文本.

### Step 2: 语义审阅 (你必须亲自完成 — 这是本 skill 的核心价值)

机械检测能抓最显眼的作弊形态, 但很多隐性作弊依赖**语义理解**, 必须由你逐行读源码后判断.

**强制要求**: 用 `read` 工具打开 `<op_dir>/<op>_impl.py` 并至少完整读完一遍. 这是核心实现文件, 不得跳过.

如果仓内还存在其它与评测相关的包装/辅助文件, 可以作为补充材料阅读, 但不要把这类非标准产物当成必读前置条件.

然后**逐项**对照下表做判定. 对每项给出 PASS / FAIL / SUSPICIOUS, 并指明依据 (引用具体行号或代码片段).

| # | 隐性作弊形态 | 识别要点 |
|---|---|---|
| S1 | **空壳 jit kernel** —— `@pypto.frontend.jit` 装饰器装在一个 pass-through 函数上, 函数体只 `return torch.xxx(x)` 或直接 `return x` | 看被装饰函数的 body, 是否真有 `pypto.dsl.*` / pypto 算子调用; 全 `torch.*` 即作弊 |
| S2 | **forward 双路径** —— `forward` 中 `if cond: return pypto_path(...) else: return torch_path(...)` 评测时走 pypto, 默认走 torch | 找 forward 中的 if/else / mode flag / 环境变量分支 |
| S3 | **try/except fallback** —— 用 try 包 jit 调用, 失败直接 fallback 到 torch native | 找 try/except 包 jit 调用的代码块 |
| S4 | **绕过 jit 的 pre/post-process** —— 把核心计算放在 jit 前后的 numpy/torch 代码里, jit 只做无关紧要的一小步 | 看 forward / wrapper 在 jit 调用前后还做了哪些计算; 只允许 to-device / reshape / dtype cast / 简单 broadcast |
| S5 | **多 kernel 拆分** —— 只能由 runtime profile 判定。若 `verify_run.json.performance.cheat_multi_kernel == true`, 则判 `fail`; 否则不得仅凭 AST 中多个 jit 定义、dtype 路由或 wrapper 分支判作弊 | 以 Step 3 的 `performance.cheat_multi_kernel` 为唯一真相源; 对 dtype-specialized 多个 jit 定义但单次 forward 只产生一个 profile kernel 的情况应判 `pass` 或 `suspicious`, 不能判 `fail` |
| S6 | **mock kernel** —— jit 函数只写 `pass` / `return None` / 返回 dummy tensor; 实际计算在 wrapper 里 torch 完成 | 看 jit 函数体是否真有意义的运算 |
| S8 | **tile config 关闭核心算子** —— 通过 PyPTO config 把核心算子改成 noop / identity, 让评测路径不实际算 | 看 wrapper / __init__ / forward 是否对 pypto 全局配置做特殊修改 |
| S9 | **可疑注释** —— 代码注释提示 "for testing", "use pypto later", "fallback", "workaround", "demo only" 等 | 通读注释; 与 Step 1 的 `forbidden_text_patterns` 交叉印证 |

把审阅结论组织成 `cheat_check_semantic` 字段, 至少包括:

```json
{
  "verdict": "pass | suspicious | cheat",
  "items": [
    {"id": "S1", "status": "pass", "evidence": "..."},
    {"id": "S2", "status": "fail", "evidence": "<op>_impl.py L45-L50: forward 中存在 USE_PYPTO env flag 分支..."},
    ...
  ],
  "reasoning": "<总结性论述, 200-500 字>"
}
```

> **底线原则**: 拿不准的项一律标 `suspicious` 而非 `pass`, 把不确定性写在 `evidence` / `reasoning` 里. 宁可让调用方多核一遍, 不要错放作弊产物过关.

### Step 3: 精度 + 性能验证 (调统一 verifier CLI)

无论 Step 1 是否报 `cheat`, 都必须继续执行 Step 3. 调:

```bash
python -m benchmark.verifier verify \
  "<op_dir>" \
  --op-name "<op_name>" \
  --task-desc "<task_desc_file>" \
  --mode "<mode>" \
  --arch "<arch>" \
  --device-id <device_id> \
  --verify-timeout <verify_timeout> \
  --json-out "<output_dir>/verify_run.json"
```

若 prompt 中提供了 `log_dir` 且值不是 `default`, 必须继续追加:

```bash
  --log-dir "<log_dir>" \
```

若 prompt 中提供了 `verify_rtol` / `verify_atol` 且值不是 `default`, 必须继续追加:

```bash
  --verify-rtol <verify_rtol> \
  --verify-atol <verify_atol> \
```

若 prompt 中提供 `keep_artifacts=true`, 必须继续追加:

```bash
  --keep-artifacts
```

注意:

- 该 CLI 内部会再跑一次 `cheat_detector`, 但当前仅作为 warning 信号, 不应阻塞 correctness/performance.
- 若 `mode` 为 `performance` 或 `full`, 同时检查 `verify_run.json.performance.cheat_multi_kernel`: 该字段为 true 表示运行时 swimlane 抓到了多 kernel, 你必须把综合 verdict 升级为 `cheat`.
- **multi-kernel 单一真相源**: 不得根据源码中 `@pypto.frontend.jit` 定义数量、`pypto.frontend.jit(...)` 调用数量、dtype 路由分支数量来判定 multi-kernel 作弊。只有 `performance.cheat_multi_kernel == true` 才能触发 S5 `fail` / `FAIL_CHEAT`。如果未运行 performance/profile, S5 最多标 `suspicious` 并说明缺少 runtime profile 证据, 不得标 `fail`。
- CLI 退出码: `0` 通过, 非零失败. 但**不要依赖**退出码做判定, 一律以 JSON 内的 `verdict_machine` / `correctness.status` / `performance.status` 为准.
- **runtime/aicore retry 优先于 final_verdict**: 如果 `correctness.status != "passed"` 的日志显示 runtime error、aicore error、ACL stream synchronize failed、507xxx 运行时错误, 不得直接套用 `FAIL_CORRECTNESS`; 必须先按前置设备准备规则重试/切卡。只有在重试预算耗尽后仍复现, 或你能基于日志明确排除卡问题时, 才能进入 Step 4 做最终裁定。

### Step 4: 综合报告 (写入 skill_report.json)

把三方结论合并写到 `<output_dir>/skill_report.json`. **必须** 用 `write` 工具落盘, 不要只在 chat 里输出.

写入后必须立即执行 JSON 合法性门禁:

```bash
python -m json.tool "<output_dir>/skill_report.json" >/dev/null
```

若该命令失败, 说明 `skill_report.json` 不是合法 JSON。必须重写该文件并重新执行
`python -m json.tool`, 直到命令通过。JSON 校验失败时不得输出
`skill_report.json written: ...` 成功行, 也不得把本 skill 判为完成。

报告结构:

```json
{
  "op_name": "...",
  "op_dir": "...",
  "task_desc_file": "...",
  "mode": "...",
  "skill_version": "1.0",
  "cheat_check_script": { /* Step 1 cheat_check_script.json 全文 */ },
  "cheat_check_semantic": {
    "verdict": "pass | suspicious | cheat",
    "items": [...],
    "reasoning": "..."
  },
  "correctness": { /* verify_run.json.correctness, status 之外补 reasoning */ },
  "performance": { /* verify_run.json.performance, 同上 */ },
  "final_verdict": "PASS | FAIL_CHEAT | FAIL_CORRECTNESS | FAIL_PERFORMANCE | BASELINE_FAILED | ERROR",
  "failure_category": "<见下表; `final_verdict == \"PASS\"` 时必须为 \"\" (空字符串) >",
  "final_reasoning": "<60-200 字的综合判定论述, 引用以上字段做依据>"
}
```

`final_verdict` 决定规则 (按优先级从上到下):

> 以下规则只在 Step 3 的 runtime/aicore retry 处理完成后适用；未完成应继续重试, 不应写最终报告。

1. `cheat_check_semantic.verdict == "cheat"` 或 `performance.cheat_multi_kernel == true` → `FAIL_CHEAT`
2. `correctness.status != "passed"` 且 Step 3 日志明确显示 **KernelBench/PyTorch baseline 或 framework model 自身无法执行**（例如 torch_npu/aclnn 报参数不支持、baseline 在调用生成实现前失败、PyPTO 实现未被实际测试到）→ `BASELINE_FAILED`
3. `correctness.status != "passed"` 且属于数值差异、输出不匹配、shape 不匹配、或 runtime/aicore 在完成规定重试后仍复现 → `FAIL_CORRECTNESS`
4. mode 含性能且 `performance.status == "failed"` 或 `"error"` → `FAIL_PERFORMANCE`
5. `cheat_check_script.verdict == "cheat"` 或 `cheat_check_script` / `cheat_check_semantic` 任一为 `suspicious` → 仍 `PASS`, 但 `final_reasoning` 必须明确列出未消的 warning / suspicious 项, 让调用方自行评估
6. 其余 → `PASS`
7. 任何 step 因技术性原因失败 (CLI 异常, 文件不可读, etc.) → `ERROR`

`failure_category` 赋值规则 (LLM 必须产出该字段; **不修改、不替换** 上文 `final_verdict` 判定规则, 仅在已有 `final_verdict` 与证据基础上按本表归类):

- **PASS**: `failure_category` **必须**为 `""` (空字符串), 不得省略该字段.
- **非 PASS**: 从下表 **自上而下** 扫描, **命中第一条**即为此字段唯一取值; 各类别 **互斥**. 若多条看似同时成立, 选 **最能概括根因** 且 **编号更小** 的那一类.
- 取值必须是下表 **英文常量** 之一 (全大写 + 下划线), 不得自造别名.

| 优先级 | `failure_category` | 触发条件 (摘要) |
|---:|---|---|
| 1 | `CHEAT_MULTI_KERNEL` | `performance.cheat_multi_kernel == true` (与 Step 3 / `final_verdict` 规则 1 中 multi-kernel 分支一致) |
| 2 | `CHEAT_SEMANTIC` | `cheat_check_semantic.verdict == "cheat"` 且不满足优先级 1 |
| 3 | `BASELINE_FAILED` | 与 `final_verdict` 规则 2 同义: baseline / framework 在测到实现前即失败或不支持 |
| 4 | `CORRECTNESS_RUNTIME` | `correctness.status != "passed"`, 日志为 runtime / aicore / ACL / 507xxx 等, 且已按设备准备规则重试耗尽后仍失败 (与优先级 3 `BASELINE_FAILED` 区分: baseline 未测到实现; 本项为 **被测实现路径** 上运行失败) |
| 5 | `CORRECTNESS_SHAPE_OR_IO` | correctness 失败且根因为 **shape / dtype 契约 / 输出结构** 等非纯浮点比对问题 |
| 6 | `CORRECTNESS_NUMERICAL` | correctness 失败且根因为 **数值差异、atol/rtol 不达标、张量近似比较失败** |
| 7 | `PERFORMANCE_FAILED` | 与 `final_verdict` 规则 4 同义: mode 含性能且 `performance.status` 为 `failed` 或 `error` |
| 8 | `ERROR_INPUT_OR_ARTIFACT` | 输入契约破坏或产物缺失导致无法按 workflow 执行 (如 `op_dir` 缺源、缺 `task_desc_file` 等), 或规则表「`op_dir` 缺源文件」类 **ERROR** |
| 9 | `ERROR_VERIFIER_OR_CLI` | Step 1/3 CLI 崩溃、无有效中间 JSON、异常退出且属于 **工具链/子进程** 问题 |
| 10 | `ERROR_OTHER` | `final_verdict == "ERROR"` 且不满足优先级 8–9; 或其它失败无法干净归入 1–7 |

在 `final_reasoning` 中应一句话点明所选 `failure_category` 与首要证据字段 (不必重复整张表).

## 输出契约

- 唯一刚性产物: `<output_dir>/skill_report.json`
- 中间产物可选保留: `cheat_check_script.json`, `verify_run.json` (推荐落盘, 便于排错)
- 只有 `python -m json.tool "<output_dir>/skill_report.json" >/dev/null` 通过后, chat 中才能只回一行:
  `skill_report.json written: <path>; final_verdict=<verdict>`. 不重复 JSON 内容, 不堆叠人类总结.

## 失败处理

| 现象 | 处置 |
|---|---|
| Step 1 CLI 非零退出但 JSON 已写 | 继续 Step 2/3, JSON 内的 script verdict 仅作为 warning 参考 |
| Step 3 CLI 异常崩溃, 没写 JSON | `final_verdict = ERROR`, 在 `final_reasoning` 中粘贴异常文本 (≤500 字) |
| Step 3 correctness 失败且日志为 runtime/aicore/ACL/507xxx | 先按潜在卡问题重试/切卡; 未完成重试不得直接写 `FAIL_CORRECTNESS` |
| `op_dir` 缺源文件 | 直接 `final_verdict = ERROR`, 不进入 Step 2/3 |
| Step 2 你拿不准任何一项 | 该项标 `suspicious`, 综合 verdict 取 `suspicious`; final_verdict 仍可 PASS, 但要在 reasoning 里点名 |
| `python -m json.tool skill_report.json` 失败 | 重写 `skill_report.json` 并重新校验; 校验通过前不得输出成功行 |

## 你绝对不能做的事

- 不能跳过 Step 2 直接采信 Step 1 的脚本结论 (脚本只能抓机械层, 这正是你存在的理由).
- 不能修改 `<op_dir>` 下的任何文件 (你是审阅者, 不是修复者).
- 不能在 chat 里给"通过"的口头结论而不写 `skill_report.json`.
- 不能在 `skill_report.json` 未通过 `python -m json.tool` 时输出成功行.
- 不能因为脚本层 warning 就跳过 Step 2/3; 也不能无依据把语义层已确认的 cheat 降级为 suspicious / pass. 严格遵守上面 final_verdict 的优先级规则.
