# KDA Dynamic B/S 算子

## 概述

本目录提供一个 KDA（Kimi Delta Attention）风格前向算子实现，重点是：

- `B`（batch）动态轴
- `S`（sequence）动态轴
- `D` 静态编译轴

实现以递推状态矩阵为核心，按 token 顺序更新状态并输出当前步结果。

## 数学定义

对每个 `b,t`：

1. `Outer_{b,t} = k_{b,t}[:, None] * v_{b,t}[None, :]`
2. `S_{b,t} = S_{b,t-1} * alpha_{b,t}[:, None] + Outer_{b,t} * beta_{b,t}[:, None]`
3. `y_{b,t} = Σ_i ( q_{b,t}[i] * S_{b,t}[i, :] )`
4. 初始状态：`S_{b,-1} = 0`

## 目录结构

- `SPEC.md`：需求规格
- `API_REPORT.md`：API 探索与约束映射
- `DESIGN.md`：实现设计
- `kda_golden.py`：纯 PyTorch 参考实现
- `kda_impl.py`：PyPTO kernel 与 wrapper
- `test_kda.py`：精度测试入口
- `PROCESS.md`：实现过程记录

## 运行方式

### 1. 运行 golden 自检

```bash
python3 kda_golden.py
```

### 2. 运行 PyPTO 精度测试（NPU）

```bash
export TILE_FWK_DEVICE_ID=0
python3 test_kda.py
```

指定单用例：

```bash
python3 test_kda.py kda::case_mid
```

查看用例：

```bash
python3 test_kda.py --list
```

## 验证入口

- 测试脚本会输出：
  - `[PRECISION_PASS]`：全部通过
  - `[PRECISION_FAIL] ...`：精度失败
  - `Runtime error: ...`：功能或环境失败

## 已知限制

- 当前实现仅支持 `FP32`
- 当前版本不支持外部传入 `initial_state`
- v1 为正确性优先实现，未做性能调优
