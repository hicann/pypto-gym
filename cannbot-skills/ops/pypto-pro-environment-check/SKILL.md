---
name: pypto-pro-environment-check
description: PyPTO-Pro 环境检测与反馈技能。当 PyPTO-Pro 工作流（Stage 1–4）任意阶段遇到疑似环境问题（软件/硬件）时统一加载：torch_npu / pypto_pro 导入失败、npu-smi 无响应、NPU 设备不可见或不可用、CANN 未配置、kernel 编译/运行异常疑似环境所致等。采用「事实验证优先、脚本诊断兜底」两步法，只检测与反馈、不修复环境。触发词：环境检查、环境问题、环境验证、environment check。
---

# PyPTO-Pro 环境检测（pypto-pro-environment-check）

PyPTO-Pro 工作流专用的**环境体检**技能。职责边界：

- **只检测 + 反馈，不修复**。符合 PyPTO-Pro 工作流「环境已由用户预配完毕，报错反馈不自行修改」约定。
- 发现问题时给出**分类建议**（该找哪个 skill / 该查什么），但不自动执行安装或修复。
- 与 PyPTO 环境的安装/修复步骤职责不同：本 skill 面向 Pro 工作流、纯检测。若检测判定需要安装/修复，**明确告知**调用者当前缺失的具体组件，并引导查阅 CANN、torch_npu、PyPTO/PyPTO-Pro 官方安装文档，本 skill 不代劳。

---

## 两步法总览

```
┌─ Step 1: 事实验证（fact check）──────────────────────────┐
│  直接跑一个真实 kernel 端到端测试。                        │
│  跑通且 PASS → 环境可用，直接反馈「可用」，结束。          │
│  （最强证据：能真跑通 = 全链路 torch_npu + pypto_pro +    │
│    NPU 设备 + CANN 都正常，无需逐项探测）                  │
└──────────────────────────────┬───────────────────────────┘
                               │ 未通过（失败 / 报错 / 超时）
                               ▼
┌─ Step 2: 脚本诊断（diagnose）────────────────────────────┐
│  运行 env_preflight.py 分层探测，定位具体坏在哪一层，      │
│  按 [1]设备 / [2]架构 / [3]运行时 / [4]CANN 分类反馈，     │
│  并给出每类问题的指向性建议（不自动修）。                  │
└──────────────────────────────────────────────────────────┘
```

**核心原则**：Step 1 通过就不跑 Step 2。事实胜于探测。

---

## Step 1：事实验证

在**当前环境**下直接运行仓库内的真实 kernel 端到端测试，运行时不要做任何环境配置，严格按照以下命令执行（**限时 5 分钟**）：

```bash
# 默认在 0 号卡验证；若需指定某张卡（如疑似该卡环境异常），先 export：
#   export TILE_FWK_DEVICE_ID=<id>
timeout 300 python scripts/test_matmul_8k_example.py
```

该测试的行为（`scripts/test_matmul_8k_example.py`）：
- `import pypto_pro.language as pl` + `@pl.jit(auto_mutex=True)` 编译一个真实 matmul kernel
- 在调用者反馈「有问题」的卡上运行 M=N=8192, K=128 的 matmul（通过 `TILE_FWK_DEVICE_ID` 指定该卡）
- 与 `torch.matmul` golden 对比，`torch.testing.assert_close`（rtol/atol=1e-2）
- 通过后打印 `Correctness PASS`

### PASS 判定（两个条件同时满足）

| 条件 | 判定方式 |
|------|---------|
| 进程正常退出 | exit code == 0 |
| 精度通过 | stdout / 日志中出现 `Correctness PASS` |

### 结果分支

- **PASS** → 环境全链路可用。**直接反馈「环境可用」，不再执行 Step 2**。
- **未通过**（非 0 退出 / 无 `Correctness PASS` / 报错 / 超时）→ 进入 Step 2 定位根因。

> ⚠️ 运行超时保护：该测试正常应在数分钟内完成。**限时 5 分钟（`timeout 300`）**；若 5 分钟内无 `Correctness PASS`（疑似设备 hang 或编译卡住），中断并按「未通过」进入 Step 2。

---

## Step 2：脚本诊断

事实验证未通过时，运行分层诊断脚本定位问题层：

```bash
python scripts/env_preflight.py
```

- 人类可读摘要输出到 stdout。
- 末尾一行 `PREFLIGHT_JSON: {...}` 为机器可读结果，反馈时以此为准。

脚本分四层检测（分层后端 + 只检测不修复）：

| 层 | 检测内容 | 后端 |
|----|---------|------|
| **[1] 设备** | NPU 枚举 / 健康 / 可用数 | 主：npu-smi；回退：torch_npu（npu-smi 为 stub/hang 时） |
| **[2] 架构** | dav-* 架构（如 dav-2201 / dav-3510=a5） | libascend_hal.so |
| **[3] 运行时** | torch / torch_npu / pypto_pro.language / pl.jit 可导入性 | Python import |
| **[4] CANN** | ASCEND_HOME_PATH / toolkit / 版本 / OPP | 文件系统 + 环境变量 |

### PREFLIGHT_JSON 关键字段

| 字段 | 含义 |
|------|------|
| `passed` | 是否无阻断错误（error_count == 0） |
| `errors` | 阻断项列表（真正「没法干活」的问题） |
| `warnings` | 非阻断项（如 npu-smi 无响应但 torch_npu 可用） |
| `devices` | 枚举到的设备（含 health） |
| `npu_usable` | [1] 层唯一裁决：是否枚举到可用卡 |
| `npu_arch` | 架构串 |
| `torch_npu_ok` / `pypto_pro_ok` | 运行时库可导入性 |
| `cann_version` | CANN 版本 |

---

## 分类反馈与建议（检测到问题时，不自动修）

根据 PREFLIGHT_JSON 中各层结论，向调用者给出指向性建议：

| 问题层 | 典型 error / warning | 分类建议（引导，不自动执行） |
|--------|---------------------|------------------------------|
| [1] 设备 | `未检测到任何 NPU 设备` | 双后端都枚举不到卡：检查 `npu-smi info -m` 是否 hang / 是否被 `ASCEND_RT_VISIBLE_DEVICES` 屏蔽 / 设备是否进入 bad state（可能需重置或换卡） |
| [1] 设备 | `npu-smi 无响应` (warning) | npu-smi 疑似 stub 或 hang，但 torch_npu 已确认有卡 → 通常可继续；若同时无卡则按上一条处理 |
| [2] 架构 | `架构探测未成功` (warning) | 一般不阻断；确认 CANN driver（libascend_hal.so）是否就位 |
| [3] 运行时 | `torch_npu 未安装` | 引导用户查阅 [torch_npu 官方安装文档](https://gitcode.com/cann/pytorch/releases)（本 skill 不代装） |
| [3] 运行时 | `torch_npu 已安装但初始化失败` | 多为版本不配套或环境变量缺失；查 CANN 与 torch_npu 版本配套表，按官方文档重装匹配版本 |
| [3] 运行时 | `import pypto_pro.language 失败` | pypto_pro 未安装 / 未编译；引导查阅 PyPTO/PyPTO-Pro 官方安装文档编译安装 |
| [4] CANN | `无法定位 CANN Toolkit` | 未 `source set_env.sh` 或 `ASCEND_HOME_PATH` 未设置；提示用户 source 后重试 |

> 反馈时务必区分「事实、推断、建议」：如实报告 error/warning 原文，标注哪层通过哪层失败，给出建议但不声称已修复。

---

## 使用约定

1. **优先事实验证**：先跑 Step 1，PASS 即结束，不做多余探测。
2. **只检测不修复**：任何检测结果都只反馈，不执行安装/修复/环境变量修改。
3. **需要修复时如实反馈**：判定需安装/修复时，如实列出缺失组件与诊断证据，引导查阅 CANN、torch_npu、PyPTO/PyPTO-Pro 官方安装文档。
4. **脚本资产**：`scripts/env_preflight.py`（聚合入口）依赖同目录 `scripts/_npu_info.py`（npu-smi 封装）与 `scripts/get_npu_arch.py`（架构探测），三者须同目录。
5. **设备号指定**：Step 1 事实验证通过 `TILE_FWK_DEVICE_ID` 环境变量指定在哪张卡上跑，与 PyPTO-Pro 框架惯例一致（见 `conftest.py` 单卡模式注入、`scripts/test_matmul_8k_example.py` 标准用法）。默认 0 号卡；若调用者已知某卡疑似异常，应先 `export TILE_FWK_DEVICE_ID=<id>` 再跑 Step 1，做到「哪张卡出问题就在哪张卡上测」。该变量由调用者设置，skill 本身不修改环境变量（遵循约定 2）。
