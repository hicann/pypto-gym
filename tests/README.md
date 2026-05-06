# PyPTO-Gym Tests

本目录是 pypto-gym 的**测试目录**，与 `src/pypto_gym/ops/pypto_tile/` 下的算子实现一一对应：

| Kernel 实现 | 对应测试 |
|---|---|
| [src/pypto_gym/ops/pypto_tile/arctic/sum_lstm.py](../src/pypto_gym/ops/pypto_tile/arctic/sum_lstm.py) | [ops/arctic/test_sum_lstm.py](ops/arctic/test_sum_lstm.py) |
| [src/pypto_gym/ops/pypto_tile/qat/qat_impl.py](../src/pypto_gym/ops/pypto_tile/qat/qat_impl.py) | [ops/qat/test_qat.py](ops/qat/test_qat.py) |
| [src/pypto_gym/ops/pypto_tile/glm_v4_5/glm_*_impl.py](../src/pypto_gym/ops/pypto_tile/glm_v4_5/) | [ops/glm_v4_5/test_glm_*.py](ops/glm_v4_5/) |
| [src/pypto_gym/ops/pypto_tile/deepseek_v32_exp/*_impl.py](../src/pypto_gym/ops/pypto_tile/deepseek_v32_exp/) | [ops/deepseek_v32_exp/test_*.py](ops/deepseek_v32_exp/) |
| [src/pypto_gym/ops/pypto_tile/qwen3_1_7b/qwen3_*.py](../src/pypto_gym/ops/pypto_tile/qwen3_1_7b/) | [ops/qwen3_1_7b/test_*.py](ops/qwen3_1_7b/) |
| [src/pypto_gym/ops/pypto_tile/qwen3_next/gated_delta_rule_impl.py](../src/pypto_gym/ops/pypto_tile/qwen3_next/gated_delta_rule_impl.py) | [ops/qwen3_next/test_gated_delta_rule.py](ops/qwen3_next/test_gated_delta_rule.py) |

测试文件通过绝对包路径引用 kernel 实现；`tests/ops/qwen3_1_7b/` 因历史原因仍使用 sibling import，依赖本目录下的 `conftest.py` 注入 `sys.path`。

## 运行方式

```bash
# 根目录运行全部测试（排除 experimental/）
pytest

# 运行指定模型
pytest tests/ops/arctic -v

# 运行单个用例文件
pytest tests/ops/glm_v4_5/test_glm_gate.py -v

# 指定 NPU 设备
pytest tests/ops/arctic --device 0

# 运行 experimental 目录下的算子（需显式指定路径）
pytest src/pypto_gym/ops/experimental/matmul -v
```

## 标记（markers）

- `@pytest.mark.soc("950", "910")` — 指定 SoC 版本。未标注等同于 `"910"`。
- `@pytest.mark.world_size(N)` — 指定所需 NPU 卡数，默认 1。

SoC 筛选、耗时估计重排序等调度逻辑定义在顶层的 [conftest.py](../conftest.py)。

## 添加新测试

参考 [README 的 "添加新算子" 小节](../README.md#-添加新算子)。
