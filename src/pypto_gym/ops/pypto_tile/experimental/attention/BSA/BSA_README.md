# BSA (Block Sparse Attention) PyPTO 实现说明文档

## 1. 概述

BSA（Block Sparse Attention）是面向华为昇腾 NPU 的块稀疏注意力算子的 PyPTO 实现。采用 **Mask2Idx 风格紧凑张量**策略：在 Python wrapper 层预计算每个 Q 块对应的有效 KV 块集合，构建紧凑的 K/V 张量，使 kernel 内层循环仅迭代 `maxSelectNum` 而非 `numKB`。无效迭代使用 `valid_mask=0.0` 将 S 设为大负数，使 exp 下溢为零，对在线 softmax 产生中性贡献。

**已实现功能**：
- 前向传播（含在线 softmax + LSE）
- 反向传播（重计算策略，dQ / dK/dV 双 kernel 拆分）
- 块稀疏掩码支持（dense / sparse 双路径）
- GQA（Grouped Query Attention）兼容
- BNSD 布局：`[batch, headNum, seqLen, headDim]`
- FP16 数据类型（Q/K/V/O/dO/dQ/dK/dV = FP16, softmaxLse = FP32）
- head_dim = 128（固定）

**严格遵循 BSA.md 规范**：
- 数据类型：Q/K/V/O/dO/dQ/dK/dV = FLOAT16，softmaxLse = FLOAT32
- softmaxLse 输出形状：`[B, H_q, Sq]`
- head_dim = 128（固定）
- GQA：Hq >= Hkv，Hq % Hkv == 0，groupSize = Hq / Hkv
- dV 无 scale；dQ 和 dK 乘以 scaleValue (1/sqrt(d))
- 前向容差：atol=0.0001, rtol=0.0078125
- 反向容差：atol=0.0001, rtol=0.0078125

---

## 2. 文件结构与职责

```
models/experimental/attention/BSA/
├── common/                        # 共享模块
│   ├── bsa_common.py              # 共享配置（BSAConfig）、golden/impl 辅助函数、紧凑张量构建器、掩码生成
│   └── bsa_test_utils.py          # 共享测试基础设施（环境检查、性能追踪、输入生成、比较辅助）
│
├── FWD/                           # 前向算子
│   ├── bsa_fwd_impl.py            # 前向 NPU kernel 实现（sparse + dense 双路径）
│   └── docs/
│       ├── SPEC.md                # 算子需求规格
│       ├── API_REPORT.md          # PyTorch → PyPTO API 映射报告
│       ├── DESIGN.md              # 算子设计文档（计算图、tiling、loop、精度、性能）
│       └── README.md              # 前向算子说明（语义、公式、I/O 规格）
│
├── BWD/                           # 反向算子
│   ├── bsa_bwd_impl.py            # 反向 NPU kernel 实现（dQ + dK/dV 双 kernel）
│   └── docs/
│       ├── SPEC.md                # 算子需求规格
│       ├── API_REPORT.md          # PyTorch → PyPTO API 映射报告
│       ├── DESIGN.md              # 算子设计文档（重计算策略、kernel 拆分、累积器设计）
│       └── README.md              # 反向算子说明（语义、公式、I/O 规格）
│
├── TEST/                          # 测试脚本与 golden 参考基准
│   ├── bsa_fwd_golden.py          # 前向 CPU 精度参考基准（纯 PyTorch）
│   ├── bsa_bwd_golden.py          # 反向 CPU 精度参考基准（纯 PyTorch）
│   ├── test_bsa_fwd.py            # 前向测试套件（10 个用例）
│   └── test_bsa_bwd.py            # 反向测试套件（9 个用例）
│
├── output/                        # 性能数据输出（泳道图、perfetto trace）
├── BSA_README.md                  # 本文档（总体说明）
└── PROMPT.txt                     # 开发需求描述
```

> **说明**：本目录不使用 `__init__.py`。测试脚本通过 `_resolve_bsa_root()` 定位 BSA 根目录，
> 再将 `common/`、`FWD/`、`BWD/` 分别加入 `sys.path`，以直接模块名（如 `from bsa_common import ...`）导入。
> Golden 文件与测试脚本同在 `TEST/` 下，Python 自动将脚本目录加入 `sys.path[0]`，无需额外配置。

### 文件职责

| 文件 | 职责 | 关键导出 |
|------|------|---------|
| `common/bsa_common.py` | 共享配置、golden/impl 辅助函数、紧凑张量构建、掩码生成 | `BSAConfig`, `DEFAULT_CONFIG`, `generate_block_sparse_mask()`, `_build_sparse_kv()`, `_make_jit_opts()` |
| `common/bsa_test_utils.py` | 环境检查、性能数据采集、输入生成、比较辅助 | `_check_env()`, `gen_inputs()`, `_compare_tensors()`, `_compare_grads()` |
| `TEST/bsa_fwd_golden.py` | 前向 CPU 精度参考基准（纯 PyTorch） | `bsa_forward_golden()` |
| `FWD/bsa_fwd_impl.py` | 前向 NPU kernel 实现（sparse + dense 双路径） | `block_sparse_attention_forward()` |
| `TEST/bsa_bwd_golden.py` | 反向 CPU 精度参考基准（纯 PyTorch） | `bsa_backward_golden()` |
| `BWD/bsa_bwd_impl.py` | 反向 NPU kernel 实现（dQ + dK/dV 双 kernel 拆分） | `block_sparse_attention_backward()` |
| `TEST/test_bsa_fwd.py` | 前向自动化测试套件 | `main()` |
| `TEST/test_bsa_bwd.py` | 反向自动化测试套件 | `main()` |

### 依赖关系

```
common/bsa_common         ← (无内部依赖)
common/bsa_test_utils     ← bsa_common

FWD/bsa_fwd_impl          ← bsa_common

BWD/bsa_bwd_impl          ← bsa_common（独立于 fwd_impl，无交叉依赖）

TEST/bsa_fwd_golden       ← bsa_common
TEST/bsa_bwd_golden       ← bsa_common

TEST/test_bsa_fwd         ← bsa_test_utils, bsa_fwd_golden, bsa_fwd_impl
TEST/test_bsa_bwd         ← bsa_test_utils, bsa_fwd_golden, bsa_fwd_impl,
                             bsa_bwd_golden, bsa_bwd_impl
```

---

## 3. 架构设计概览

> 本节为高层概览。详细的设计文档请参阅 `FWD/docs/DESIGN.md` 和 `BWD/docs/DESIGN.md`。

### 3.1 整体调用链

```
用户代码
  │
  ├─ block_sparse_attention_forward(Q, K, V, mask, ...)
  │    ├─ Python wrapper: pad, reshape 4D→2D, allocate outputs
  │    ├─ _is_dense_mask() → dense / sparse 路径选择
  │    ├─ dense: _get_dense_fwd_kernel(...) → 内层遍历 numKB
  │    ├─ sparse: _build_sparse_kv() → _get_sparse_fwd_kernel(...) → 内层遍历 maxSel
  │    └─ 裁剪填充, reshape output
  │
  └─ block_sparse_attention_backward(dO, Q, K, V, O, lse, mask, ...)
       ├─ Python wrapper: pad, reshape, flatten LSE
       ├─ dQ kernel: _build_sparse_kv() → _get_dq_kernel(...)
       ├─ dK/dV kernel: _build_sparse_q_dkdv() → _get_dk_dv_kernel(...)
       └─ 裁剪填充, reshape outputs
```

### 3.2 核心设计策略

| 策略 | 说明 |
|------|------|
| **Mask2Idx 紧凑张量** | wrapper 层预收集有效 KV 块索引，构建紧凑 K/V 张量，kernel 仅迭代有效块 |
| **Online Softmax** | 逐 KV 块迭代维护 m/l/o 累积器，保证分块与全量数学等价 |
| **Dense/Sparse 双路径** | dense 掩码时走无 mask 开销的独立 kernel |
| **反向双 Kernel 拆分** | dQ 和 dK/dV 独立 kernel，规避 `pypto.assemble` 覆写语义问题 |
| **Kernel 工厂 + 缓存** | 因不支持 `pypto.DYNAMIC`，每种 shape 编译独立 kernel 并缓存 |
| **重计算策略** | 反向不存储 S/P，仅保存 O 和 LSE，反向时重计算 P = exp(scale*S - LSE) |

### 3.3 数据布局与分块

- **布局**: BNSD `[B, H, S, 128]`，kernel 内部 reshape 为 2D `[B*H*S, 128]`
- **分块**: block_shape_x=256（Q 块），block_shape_y=512（KV 块）
- **TileShape**: vec `(128, 128)`，cube `(128, 128, 128)`

---

## 4. PyPTO API 约束与规避方案

> 详细的 API 映射请参阅 `FWD/docs/API_REPORT.md` 和 `BWD/docs/API_REPORT.md`。

| # | 限制 | 规避方案 |
|---|------|---------|
| 1 | 不支持 `pypto.DYNAMIC` | 工厂函数 + 字典缓存，每种 shape 编译一个 kernel |
| 2 | 不支持 `pypto.cond()` | 裸 `if pypto.is_loop_begin(v):` |
| 3 | `SymbolicScalar ** -0.5` 不支持 | kernel 外计算为 Python float，闭包捕获 |
| 4 | `pypto.where` 仅支持单轴广播 | 预展开 mask 到 `[BLOCK, BLOCK]` |
| 5 | `pypto.where(mask, tensor, scalar)` CCE 失败 | 使用 `S*mask + (1-mask)*neg` 算术替代 |
| 6 | `pypto.cast(BOOL, FP32)` kernel 内数据损坏 | wrapper 中预转换 `mask.float()` |
| 7 | `pypto.assemble` + `reshape` 冲突 | 输出 tensor 传 3D，避免 kernel 内 reshape |
| 8 | `pypto.sum(FP16)` 返回 FP16 | cast 到 FP32 后再 sum |
| 9 | `pypto.assemble` 是覆写非累加 | 拆分 kernel，局部累积后一次性写出 |
| 10 | `and` + `is_loop_begin` 触发 `ValueError` | 嵌套 if 代替 and |
| 11 | Cube tile `[256, 128]` 不支持 | 必须使用 `[128, 128]` |
| 12 | `pypto.view` 2D 偏移对 LSE 不正确 | 反向时 flatten LSE 到 `[B*Hq*Sq, 1]`，用 1D 偏移 |
| 13 | wrapper 数据与 kernel 存在竞争 | `torch.npu.synchronize()` 确保 wrapper 数据写入完成 |

---

## 5. 精度评测结果

### 5.1 测试环境

- **硬件**: Ascend 910B3, 22 AICore
- **CANN**: 9.0.0
- **数据类型**: FP16 (Q/K/V/O/dO/dQ/dK/dV), FP32 (softmaxLse)
- **容差**: `atol=0.0001, rtol=0.0078125`

### 5.2 前向精度（10 个测试用例，9 执行 + 1 跳过，全部 PASSED）

| 测试 | 配置 | 稀疏率 | O max_diff | LSE max_diff | 结果 |
|------|------|--------|------------|-------------|------|
| Basic Sparse50 | B=1, Hq=4, Hkv=2, Sq=256, Skv=256 | 50% | 0.000122 | 0.000001 | ✅ |
| GQA group4 | B=1, Hq=8, Hkv=2, Sq=256, Skv=512 | 40% | 0.000061 | 0.000001 | ✅ |
| GQA Hq32 Hkv8 | B=1, Hq=32, Hkv=8, Sq=256, Skv=256 | 50% | 0.000122 | 0.000001 | ✅ |
| Long Seq S1024 | B=1, Hq=8, Hkv=1, Sq=1024, Skv=1024 | 30% | 0.000061 | 0.000001 | ✅ |
| Sparse30% | B=1, Hq=4, Hkv=4, Sq=512, Skv=512 | 30% | 0.000061 | 0.000001 | ✅ |
| Sparse70% | B=1, Hq=4, Hkv=4, Sq=512, Skv=512 | 70% | 0.000061 | 0.000001 | ✅ |
| Dense 100% | B=1, Hq=4, Hkv=4, Sq=256, Skv=256 | 100% | 0.000122 | 0.000001 | ✅ |
| Batch2 | B=2, Hq=4, Hkv=2, Sq=256, Skv=512 | 50% | 0.000061 | 0.000001 | ✅ |
| NonAligned | — | — | — | — | ⏭️ SKIP |
| Long Seq S2048 | B=1, Hq=4, Hkv=2, Sq=2048, Skv=2048 | 30% | 0.000061 | 0.000001 | ✅ |

### 5.3 反向精度（9 个测试用例，8 执行 + 1 跳过，全部 PASSED）

| 测试 | 配置 | 稀疏率 | dQ max_diff | dK max_diff | dV max_diff | 结果 |
|------|------|--------|------------|------------|------------|------|
| BWD Basic | B=1, Hq=4, Hkv=2, Sq=256, Skv=256 | 50% | 0.000031 | 0.000061 | 0.000122 | ✅ |
| BWD MHA Dense | B=1, Hq=4, Hkv=4, Sq=256, Skv=256 | 100% | 0.000031 | 0.000031 | 0.000122 | ✅ |
| BWD GQA | B=1, Hq=8, Hkv=2, Sq=256, Skv=512 | 50% | 0.000031 | 0.000031 | 0.000122 | ✅ |
| BWD Long Seq | B=1, Hq=8, Hkv=1, Sq=1024, Skv=1024 | 30% | 0.000031 | 0.000061 | 0.000244 | ✅ |
| NonAligned | — | — | — | — | — | ⏭️ SKIP |
| BWD sparse0.3 | B=1, Hq=4, Hkv=4, Sq=256, Skv=256 | 30% | 0.000031 | 0.000031 | 0.000122 | ✅ |
| BWD sparse0.7 | B=1, Hq=4, Hkv=4, Sq=256, Skv=256 | 70% | 0.000031 | 0.000031 | 0.000122 | ✅ |
| BWD medium | B=1, Hq=4, Hkv=4, Sq=512, Skv=512 | 70% | 0.000031 | 0.000031 | 0.000061 | ✅ |
| BWD long | B=1, Hq=4, Hkv=4, Sq=1024, Skv=1024 | 70% | 0.000031 | 0.000031 | 0.000061 | ✅ |

### 5.4 非对齐序列（已知限制）

当 Sq 或 Skv 不是 BLOCK_SIZE 的整数倍时：
- **零填充方案**：将 Q/K/V 填充到块对齐大小
- **误差来源**：填充的零值行在 softmax 中获得非零权重
- **根治方案**：需要在 kernel 中实现边界块的 valid_shape 裁剪（未实现）
- **当前状态**：test_09_non_aligned 和 test_15_bwd_non_aligned 已跳过

---

## 6. 运行方式

### 环境准备

```bash
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
export TILE_FWK_DEVICE_ID=0
export PYTHONPATH=/mnt/workspace/gitCode/cann/pypto/python:$PYTHONPATH
```

### 运行测试

测试脚本内置三级路径解析（`_resolve_bsa_root()`），可从任意目录运行：

1. **`BSA_ROOT` 环境变量**（显式指定）
2. **自动检测**：从脚本位置向上搜索 `BSA_README.md` + `common/bsa_common.py`
3. **仓库相对路径**：从 CWD 向上搜索 `pypto_6304/models/experimental/attention/BSA`

```bash
# 方式一：从 BSA/ 目录运行（output/ 写入 BSA/output/）
cd models/experimental/attention/BSA/
python TEST/test_bsa_fwd.py
python TEST/test_bsa_bwd.py

# 方式二：从任意目录运行（设置 BSA_ROOT）
BSA_ROOT=/path/to/BSA python /any/where/test_bsa_fwd.py

# 方式三：将 TEST/ 拷贝到任意位置（自动回退到仓库相对路径搜索）
cp -r TEST/ /tmp/bsa_test && cd /path/to/pypto_6304/..
python /tmp/bsa_test/test_bsa_fwd.py
```

### 注意事项

1. **确保无残留进程占用 NPU**：运行前检查 `ps aux | grep python | grep test_bsa`，如有残留进程需 `kill -9` 清理
2. **首次运行较慢**：每种 shape 组合需要 JIT 编译，后续运行命中缓存会快很多
3. **性能数据**：测试会自动采集泳道图数据到 `output/` 目录

---

## 7. 性能数据

### 7.1 测试环境

- **硬件**: Ascend 910B3, 22 AICore
- **CANN**: 9.0.0, bisheng 15.0.5
- **配置**: block_shape_x=256, block_shape_y=512, head_dim=128

### 7.2 前向性能（MHA Dense, B=1, Hq=Hkv=4）

| Shape | Task Time | AICore Time | AICore Util |
|-------|-----------|-------------|-------------|
| 256×256 | ~75 us | ~1.3 ms | ~37% |
| 512×512 | ~92 us | ~2.6 ms | ~43% |
| 1024×1024 | ~180 us | ~6.5 ms | ~60% |
| 2048×2048 | ~820 us | ~26.9 ms | ~55% |

### 7.3 反向性能（MHA Dense, B=1, Hq=Hkv=4）

| Shape | dQ Task Time | dK/dV Task Time | dQ AICore Util | dK/dV AICore Util |
|-------|-------------|-----------------|----------------|-------------------|
| 256×256 | ~79 us | ~85 us | ~38% | ~36% |
| 512×512 | ~100 us | ~112 us | ~44% | ~43% |
| 1024×1024 | ~183 us | ~200 us | ~60% | ~61% |

### 7.4 性能优化措施

| 优化项 | 说明 |
|--------|------|
| **Dense 路径分离** | 独立 dense kernel，省去 mask 加载和 apply 开销 |
| **Sub-block 分割** | Dense dQ 将 256 行 Q-block 分为 2×128 行子块，增加外层并行度 |
| **Per-kernel L1 reuse** | Dense dQ 保持 L1=64，其余 kernel 降至 L1=16 |
| **Kernel 工厂缓存** | 首次编译后缓存，避免重复编译 |

### 7.5 已知性能瓶颈

1. **小 shape 并行度不足**: 256×256 dQ 仅有 8 外层任务（sub-block 后），22 核利用率 ~42%
2. **累积模式限制**: dK/dV 需跨 Q 块累积，无法简单增加外层并行
3. **Cube tile 限制**: `[128,128]` 是 CANN 9.0.0 支持的最大安全配置
4. **vec_nbuffer 不可用**: 与内部 VEC_NBUFFER_MODE 冲突，无法启用 vector 子图合并
5. **Dynamic Shape 受限**: CANN 9.0.0 不支持 PyPTO 动态维度，每种 shape 需独立编译
