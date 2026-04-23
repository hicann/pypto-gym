# PyPTO-Gym 文档

本目录用于沉淀 pypto-gym 自身的设计、使用与算子规格类文档。

## 现有文档

暂未规划专属文档，主要信息分布在如下位置：

- 仓库总览与快速上手：[../README.md](../README.md)
- 测试入口与运行规则：[../tests/README.md](../tests/README.md)
- 各算子/模型说明：每个目录下的 `README.md`，例如：
  - [../src/pypto_gym/models/arctic/README.md](../src/pypto_gym/models/arctic/README.md)
  - [../src/pypto_gym/models/deepseek_v32_exp/README.md](../src/pypto_gym/models/deepseek_v32_exp/README.md)
  - [../src/pypto_gym/models/glm_v4_5/README.md](../src/pypto_gym/models/glm_v4_5/README.md)
  - [../src/pypto_gym/models/qat/README.md](../src/pypto_gym/models/qat/README.md)
  - [../src/pypto_gym/models/qwen3_next/README.md](../src/pypto_gym/models/qwen3_next/README.md)

## 计划补充

- `design/` —— 算子库整体定位与与 PyPTO 主仓的边界划分
- `tutorials/` —— 从 PyPTO 框架到典型融合算子的渐进示例
- `benchmarks/` —— 各算子在不同 SoC、不同 shape 下的性能基线
- `contributing.md` —— 提交新算子的流程、测试规约、性能门槛
