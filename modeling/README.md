# modeling

存放 `src/pypto_gym/llm/` 下各模型的**执行脚本**，对标 TileGym 仓根目录 `modeling/` 的布局。

每个子目录对应一个模型家族，内部放置：
- 推理入口（`infer.py`）
- benchmark 脚本（`bench_*.sh`）
- Dockerfile / 构建脚本
- 样例输入（`sample_inputs/`）
- 对应 README

模型结构定义本身位于 `src/pypto_gym/llm/<model>/`。
