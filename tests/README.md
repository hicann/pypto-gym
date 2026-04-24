# PyPTO-Gym Tests

这个目录是 pypto-gym 的**测试入口与共享工具**目录。

## 测试在哪里

每个模型/算子的测试用例就在它自己的目录下，**impl 与 test 同目录**，例如：

- [src/pypto_gym/ops/arctic/test_sum_lstm.py](../src/pypto_gym/ops/arctic/test_sum_lstm.py)
- [src/pypto_gym/ops/qat/test_qat.py](../src/pypto_gym/ops/qat/test_qat.py)
- [src/pypto_gym/ops/glm_v4_5/glm_attention.py](../src/pypto_gym/ops/glm_v4_5/glm_attention.py) — 该文件内含 `test_ifa()` 等用例
- [src/pypto_gym/ops/qwen3_next/qwen3_next_gated_delta_rule.py](../src/pypto_gym/ops/qwen3_next/qwen3_next_gated_delta_rule.py)
- [src/pypto_gym/ops/deepseek_v32_exp/deepseekv32_*.py](../src/pypto_gym/ops/deepseek_v32_exp/)

这种 "impl + test 同目录" 的风格沿袭自 pypto 主仓，保持文件间的兄弟 import 关系，避免无谓的路径重写。

## 运行方式

```bash
# 根目录运行全部测试（排除 experimental/）
pytest

# 运行指定模型
pytest src/pypto_gym/ops/arctic -v

# 指定 NPU 设备
pytest src/pypto_gym/ops/arctic --device 0

# 运行 experimental 目录下的算子（需显式指定路径）
pytest src/pypto_gym/ops/experimental/matmul -v
```

## 标记（markers）

- `@pytest.mark.soc("950", "910")` — 指定 SoC 版本。未标注等同于 `"910"`。
- `@pytest.mark.world_size(N)` — 指定所需 NPU 卡数，默认 1。

SoC 筛选、耗时估计重排序等调度逻辑定义在顶层的 [conftest.py](../conftest.py)。

## 添加新测试

参考 [README 的 "添加新算子" 小节](../README.md#-添加新算子)。
