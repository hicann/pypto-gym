# llm

存放基于 `pypto_gym/ops` 算子库搭建的 LLM 模型定义。

每个子目录聚焦一个模型家族（如 `deepseek`、`glm`、`qwen3_next` 等），内部放置：
- 模型结构定义（`modeling_*.py`）
- 配置文件
- 对应 README

模型的执行脚本（推理入口、benchmark 脚本、Dockerfile、样例输入等）放在仓库根目录的 `modeling/` 下，按模型家族组织。
