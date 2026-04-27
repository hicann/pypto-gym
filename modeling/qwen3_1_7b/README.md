# Qwen3-1.7B Modeling

基于 pypto-gym `ops/qwen3_1_7b/` 算子库的 Qwen3-1.7B 端到端样例与模型级测试。

## 目录结构

```
modeling/qwen3_1_7b/
├── core/                       # 修改过的 transformers modeling/configuration
│   ├── modeling_qwen3.py
│   └── configuration_qwen3.py
├── qwen3_pto_kernels/          # 整网 adapter（torch ↔ pypto，padding/cwd 隔离）
│   ├── __init__.py             # adapter 入口；从 src/pypto_gym/ops/qwen3_1_7b/ 取算子
│   └── k3_post_attn.py         # K3 整网兼容版本（ops K3 接口不兼容）
├── scripts/                    # 端到端推理入口
│   ├── ask_Qwen3-1.7B_pto.py   # PyPTO 融合算子版
│   ├── ask_Qwen3-1.7B.py       # baseline (torch eager)
│   └── README.md
├── qwen3_layer_golden.py       # torch 参考实现 (单 DecoderLayer prefill+decode)
├── test_network_shapes.py      # 网络真实形状下回归 K1/K2/K3
└── README.md
```

## 配套算子位置

| 角色 | 路径 |
|------|------|
| 算子实现 | `src/pypto_gym/ops/qwen3_1_7b/` |
| 整网 adapter（torch ↔ pypto，padding/cwd 隔离） | `modeling/qwen3_1_7b/qwen3_pto_kernels/` |
| 算子精度测试 (pytest) | `tests/qwen3_1_7b/` |

## 端到端推理

详见 [scripts/README.md](scripts/README.md)。

```bash
conda activate pypto2
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PTO_TILE_LIB_CODE_PATH=/data/z00885570/pypto-master/pto-isa

cd modeling/qwen3_1_7b/scripts
python ask_Qwen3-1.7B_pto.py --device 0 --prompt "你好" --max-new 16
```

带 `--no-pto` 切换 baseline、`--pto-attn` 启用 decode attention 融合。

## 模型级正确性脚本

| 文件 | 说明 |
|---|---|
| `qwen3_layer_golden.py` | torch 参考实现，已与 transformers 原生 `Qwen3DecoderLayer` 对照通过 |
| `test_network_shapes.py` | 在网络真实形状 (B=1, S=1/4/16/128/512) 下回归 K1/K2/K3 |
