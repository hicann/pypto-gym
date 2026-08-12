---
name: pypto-pro-environment-check
description: PyPTO-Pro 环境检测与反馈技能。当 PyPTO-Pro 工作流（Stage 1–4）任意阶段遇到疑似环境问题（软件/硬件）时统一加载：torch_npu / pypto_pro 导入失败、npu-smi 无响应、NPU 设备不可见或不可用、CANN 未配置、kernel 编译/运行超时疑似设备 hang 等。采用「事实验证优先、脚本诊断兜底」两步法 + 设备 hang 三段式评定，用于区分环境故障与算子故障；只执行有界检测并返回结构化证据，不安装依赖、不修改环境。触发词：环境检查、环境问题、环境验证、environment check、设备 hang、卡死、超时。
---

# PyPTO-Pro 环境检测（pypto-pro-environment-check）

PyPTO-Pro 工作流专用的**环境体检**技能，是所有环境问题（软件缺失 / 硬件故障 / 设备 hang）的**统一入口**。

## 职责边界

- **只检测 + 反馈，不修复**。符合 PyPTO-Pro 工作流「环境已由用户预配完毕，报错反馈不自行修改」约定。
- 发现问题时给出**分类建议**（缺什么组件 / 该换卡还是停机），但不自动执行安装或修复。
- 若检测判定需要安装/修复，**明确告知**调用者当前缺失的具体组件，并引导查阅 CANN、torch_npu、PyPTO/PyPTO-Pro 官方安装文档，本 skill 不代劳。

## 调用者上下文

加载本 skill 的 agent（coder / verifier）通常是因为**自己跑的算子脚本报错或超时**，需要判断是算子代码问题还是环境问题。本 skill 的核心价值：用一个**独立于算子代码**的官方 VF smoke 做最小验证——

- **VF smoke PASS** → 环境全链路正常，原报错属算子代码问题，agent 回 debug 循环修代码。
- **VF smoke FAIL / 超时** → 确认环境问题，按下方流程定位根因（软件缺失 or 设备 hang）。

> 这样能避免 agent 把"自己的算子编译失败"误判为"环境 hang"，也能避免把"设备真 hang"误判为"算子代码 bug"。

---

## 检测流程总览

```
agent 跑自己的算子报错/超时
        │
        ▼
  Step 1: 事实验证 ── 跑官方 VF smoke（独立于算子代码）
        │
   PASS? ─是→ 环境可用，原报错属算子代码 → 回 debug 循环
        │否
        ├─ 报错指向软件缺失（导入失败/CANN 未配置）→ Step 2 脚本诊断
        └─ 超时/无响应/疑似卡死 → 设备 hang 评定（三段式）
                                       │
                              HANG_CONFIRMED 且需换卡
                                       │
                                       ▼
                              Step 2 脚本诊断（拿可用卡列表）
```

**核心原则**：Step 1 通过就不跑 Step 2。事实胜于探测。

---

## smoke 脚本

环境检测的事实验证基于 devkit 缓存内的官方 VF smoke：

```
$PYPTO_DEVKIT_DIR/pro_ops/vf_api/test_softmax_tile_group_vf.py
```

该脚本编译一个真实的 softmax VF kernel（`@pl.jit(auto_mutex=True)` + `@pl.vector_function`），跑 6 组动态 shape case，与 `torch.softmax` golden 对比（`rtol/atol=1e-3`），通过后每个 case 打印 `PASS`，末尾打印 `Test completed!`。

**设备号设置**：该脚本通过 `TILE_FWK_DEVICE_ID` 环境变量选卡（默认 `0`）。指定非 0 卡时先 `export TILE_FWK_DEVICE_ID=<id>` 再运行：

```bash
export TILE_FWK_DEVICE_ID=<id>
timeout 600 python $PYPTO_DEVKIT_DIR/pro_ops/vf_api/test_softmax_tile_group_vf.py
```

> `export TILE_FWK_DEVICE_ID` 是选卡运行参数，不属于全局硬性规则禁止的「环境配置命令」（conda/CANN/pip 等）。默认 0 号卡时无需 export。

**超时设为 600s**：softmax VF smoke 含 6 组 case 的 JIT 编译，复杂融合算子编译可达数分钟，超时过短会把"编译慢"误判为 hang。

---

## Step 1：事实验证

在**当前环境**下直接运行官方 VF smoke，运行时不要做任何环境配置：

```bash
timeout 600 python $PYPTO_DEVKIT_DIR/pro_ops/vf_api/test_softmax_tile_group_vf.py
```

> 若需在疑似异常的卡上验证，先 `export TILE_FWK_DEVICE_ID=<id>` 再运行。

### PASS 判定（两个条件同时满足）

| 条件 | 判定方式 |
|------|---------|
| 进程正常退出 | exit code == 0 |
| 精度通过 | stdout 中出现 `PASS`（每 case 一行）且末尾含 `Test completed!` |

### 结果分支

- **PASS** → 环境全链路可用。**直接反馈「环境可用」**，不跑 Step 2，不进 hang 评定。原报错属算子代码问题。
- **未通过**（非 0 退出 / 无 `PASS` / 报错 / 超时）→ 按报错性质分流：
  - **报错明确指向软件缺失**（torch_npu / pypto_pro 导入失败、CANN 未配置等）→ 进入 **Step 2** 定位根因。
  - **运行超时 / 无响应 / 疑似卡死** → 进入下方**「设备 hang 评定」**三段式。

---

## 设备 hang 评定（三段式）

当运行超时或无响应怀疑设备 hang 时，**不得立即判 hang**，必须按以下三段式评定。

**核心原则**：
- **不得用自写临时超短超时测试（如 `set_device+randn+add+synchronize`、15s/30s 级超时）判定 hang**——编译耗时 ≠ hang（复杂融合算子 JIT 编译可达数分钟），超短超时会把"编译慢 / 瞬时卡顿"误判为"永久 hang"。
- 暂时性卡顿优先等待恢复，不轻易放弃；穷尽重试后才认定真设备故障。

**段 1：放宽超时重跑（排除"编译慢 / 瞬时卡顿"）**
以放宽超时（≥ 600s）重跑 Step 1 的 VF smoke：
- PASS → 设备正常，原超时是编译耗时 / 瞬时卡顿所致，回 debug 循环继续开发。
- 仍 FAIL / 超时 → 进入段 2。

**段 2：官方 smoke 评定**
运行 Step 1 的 VF smoke（给足超时 ≥ 600s）：
```bash
timeout 600 python $PYPTO_DEVKIT_DIR/pro_ops/vf_api/test_softmax_tile_group_vf.py
```
- smoke PASS → 卡功能正常，原报错属算子代码问题（非设备），回 debug 循环，并放宽算子运行超时。
- smoke FAIL / 超时 → 进入段 3（可能为暂时性 hang）。

**段 3：暂时性 hang 重试（等待恢复 + 5 次重试）**
判定为"可能暂时性 hang"，执行等待 + 重试循环：
- 等待 5 分钟（让设备从瞬时坏态恢复）。
- 重新运行段 2 的 smoke 脚本。
- 重复，最多 5 次：
  - **任一次 smoke PASS** → 设备已恢复（`TRANSIENT_RECOVERED`），可继续原卡开发。
  - **5 次全部 FAIL** → 认定真设备故障（`HANG_CONFIRMED`），进入段 4。

**段 4：真设备故障处置**
HANG_CONFIRMED 后，**跑一次 Step 2 的 `env_preflight.py`** 拿可用卡列表（`devices[]` 含每张卡的 `npu_id` + `health`），供编排器挑健康卡换卡：
- 有其他健康卡 → 反馈编排器换卡重派（`TILE_FWK_DEVICE_ID` 指定健康卡），原卡标记不可用。
- 无其他可用卡 → 停机向用户反馈，附完整评定证据。

## 设备故障伪装成精度失败（无超时路径）

上面的三段式由**超时 / 无响应**触发。但坏卡还有一条**不触发任何超时**的表现形式，
且更危险：**运行正常结束，干净地报出 `TOTAL 0/N passed`**。

这在现场发生过：一次完整评测跑完，代码生成阶段成功（数十个 kernel、模块正常 parse），随后全部 case "失败"。按"以 `passed` 为准"的规则，
这读起来就是灾难性回归，其预先登记的响应是 **revert 两个 commit**——而那两个 commit
毫无问题，坏的是卡。

**判别要点：坏卡在比对*之前*就失败了，坏 kernel 是比对*失败*。** 按下列证据判断，
任何一条命中即高度怀疑设备故障：

| 证据 | 设备故障 | kernel 缺陷 |
|---|---|---|
| 异常类型 | `RuntimeError`，来自 `npuSynchronizeDevice` / `copy_between_host_and_device_opapi` 等同步/拷贝入口 | 报出 MARE/MERE 与阈值对比，不抛异常 |
| 波及范围 | N/N 全失败，**包括本次改动根本没碰的 case** | 选择性失败，与改动相关 |
| core id | 报出的 core id **超出该 SKU 的核数**（如 56 核的卡报 core 56–63） | 在合法范围内 |
| 跨进程 | 相隔数分钟的两个独立进程，**逐核 dump 逐字节相同**（只有 serial 号变化）——这是被闩锁的错误被重复读出，不是发生了两次故障 | 两次故障不会产生相同寄存器状态 |
| control | **control 工作负载本身失败**（纯 `torch.sqrt`，在任何自定义 kernel 启动之前） | control 正常 |
| `npu-smi` | `Alarm`；或 Util 恒为 100% 且占着 HBM 但**没有任何进程在跑**；降温后仍 `Alarm` 说明是卡死不是过热 | 健康 |

**处置**：不进 debug 循环，不回退代码。按 `env_error` 上报，附上述证据，并在报告中
**明确声明该次运行未产生任何精度结论**（"未测量"，而非"测量为 0/N"）。

**总规则**：`TOTAL n/N passed` 只有在**精度比对确实执行过**时才是精度结论。
处置任何"全量失败"之前，先要求能证明比对执行过的**正面证据**——先读 `npu-smi`
（只读、一次往返），再归因于代码。

---

### 评定结论输出格式

反馈环境问题时必须附此结论，编排器据证据决策：

```
device_assessment: HANG_CONFIRMED / TRANSIENT_RECOVERED / DEVICE_HEALTHY / FAULT_NO_TIMEOUT
smoke_script: <脚本路径>
smoke_result: PASS / FAIL / TIMEOUT (attempt <n>/5)
evidence: <smoke 输出原文 + npu-smi 状态>
recommendation: 换卡(device_id=X) / 继续原卡 / 停机反馈用户
```

---

## Step 2：脚本诊断

事实验证未通过且非设备 hang（报错明确指向软件缺失），**或 hang 评定确认 HANG_CONFIRMED 后需拿可用卡列表**时，运行分层诊断脚本：

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
| `devices` | 枚举到的设备列表（每项含 `npu_id` / `chip_name` / `health`）——供换卡时挑健康卡 |
| `npu_usable` | [1] 层唯一裁决：是否枚举到可用卡 |
| `npu_arch` | 架构串 |
| `torch_npu_ok` / `pypto_pro_ok` | 运行时库可导入性 |
| `cann_version` | CANN 版本 |

### 分类反馈与建议（检测到问题时，不自动修）

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

1. **优先事实验证**：先跑 Step 1 VF smoke，PASS 即结束，不做多余探测。
2. **只检测不修复**：任何检测结果都只反馈，不执行安装/修复/环境变量修改。
3. **需要修复时如实反馈**：判定需安装/修复时，如实列出缺失组件与诊断证据，引导查阅 CANN、torch_npu、PyPTO/PyPTO-Pro 官方安装文档。
4. **脚本资产**：`scripts/env_preflight.py`（聚合入口）依赖同目录 `scripts/_npu_info.py`（npu-smi 封装）与 `scripts/get_npu_arch.py`（架构探测），三者须同目录。
