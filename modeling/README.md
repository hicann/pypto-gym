# modeling

存放推理入口、benchmark 脚本、Dockerfile 和样例输入，对标 TileGym 仓 `modeling/` 目录布局。

## 目录结构

```
modeling/
└── transformers/
    ├── infer.py                 # 统一推理/benchmark 入口
    ├── bench_qwen3_1_7b.sh      # Qwen3-1.7B benchmark (基准 vs PyPTO)
    ├── sample_inputs/           # 样例 prompt 文件
    ├── Dockerfile               # Ascend NPU 容器
    ├── build_docker.sh          # 容器构建脚本
    └── README.md                # 使用文档
```

见 [modeling/transformers/README.md](transformers/README.md)。

## PyPTO 内核与模型定义

模型定义（`modeling_qwen3.py` / `configuration_qwen3.py`）和 PyPTO 内核适配器（`qwen3_pto_kernels/`）现已迁入：

```
src/pypto_gym/transformers/
└── qwen3_1_7b/
    ├── __init__.py
    ├── modeling_qwen3.py
    ├── configuration_qwen3.py
    └── qwen3_pto_kernels/
        ├── __init__.py
        └── k3_post_attn.py
```

单算子实现位于 `src/pypto_gym/ops/qwen3_1_7b/`。

测试文件（golden 参考实现、网络形状回归）迁入 `tests/qwen3_1_7b/`。
