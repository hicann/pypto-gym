# Qwen3-1.7B 端到端推理脚本

基于 pypto-gym 算子库，把 Qwen3-1.7B `Qwen3DecoderLayer.forward` monkey-patch 为 PyPTO 融合算子的实现。

## 文件清单

| 文件 | 作用 |
|------|------|
| `ask_Qwen3-1.7B_pto.py` | 整网入口（使用 PyPTO 融合算子） |
| `ask_Qwen3-1.7B.py` | baseline 入口（torch eager） |

## 运行环境

| 组件 | 版本 |
|------|------|
| conda env | `pypto2` |
| torch / torch_npu | 2.8.0 / 2.8.0.post2 |
| transformers | **4.57.x**（5.x 不兼容；4.51 也可，但建议 ≥ 4.57） |
| NPU | Ascend910 (64GB HBM) |
| PyPTO | 见仓根 README "环境准备" |

## 运行命令

```bash
conda activate pypto2
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PTO_TILE_LIB_CODE_PATH=/data/z00885570/pypto-master/pto-isa

cd /data/z00885570/pypto-master/pypto-gym/modeling/qwen3_1_7b/scripts
python ask_Qwen3-1.7B_pto.py --device 0 --prompt "你好" --max-new 16
```

## 命令参数

- `--device N` — NPU 卡号（默认 0）
- `--prompt "..."` — 自定义 prompt（默认 "你好"）
- `--max-new N` — 生成 token 数（默认 64）
- `--no-pto` — 关闭 PyPTO patch，用 torch eager（baseline）
- `--pto-attn` — decode 阶段也用 PyPTO 融合 attention（默认走 torch eager attention，更稳）
- `--model-path PATH` — 模型权重路径，默认 `/data/z00885570/models/Qwen3-1.7B`
- `--step` — 手动 decode 循环 + 每 token 计时

## 模型权重 / 修改后的 modeling 文件

- 权重：`/data/z00885570/models/Qwen3-1.7B/*.safetensors`（不进仓）
- modeling/configuration（已修过 `from ...` 相对导入）：
  - `../core/modeling_qwen3.py`
  - `../core/configuration_qwen3.py`
- 仓外路径下的 `config.json` 通过 `auto_map` 指向 `core/`

## PyPTO 算子来源

整网走的算子在 `../qwen3_pto_kernels/` adapter 里（与本脚本目录同级）。adapter 自身只包含 K3 兼容版本，其余算子从 `src/pypto_gym/ops/qwen3_1_7b/` 取：

| 算子 | 取自 | 备注 |
|------|------|------|
| `qwen3_pre_attn_fused` | `src/pypto_gym/ops/qwen3_1_7b/qwen3_pre_attn_fused.py` | 与原仓内副本逐字一致 |
| `qwen3_decode_attn` | `src/pypto_gym/ops/qwen3_1_7b/qwen3_decode_attn.py` | 与原仓内副本逐字一致 |
| `qwen3_pre_qkv_iter1a` (K1) | `src/pypto_gym/ops/qwen3_1_7b/qwen3_iter1a_kernel.py` | fallback 路径，整网走 fused 时不调用 |
| `qwen3_qk_rope_q/k` (K2) | `src/pypto_gym/ops/qwen3_1_7b/qwen3_k2_qk_rope.py` | fallback 路径 |
| `qwen3_post_attn_k3` (K3) | adapter 包内 `k3_post_attn.py` | **不能用 ops 那一份** — ops 版 K3 接口要求 Wgate/Wup/Wdown 已转置成 `[H, INT_SIZE]`，整网传入的是 `Linear.weight` 原始 layout `[INT_SIZE, H]` |

## 端到端验证

```
[pto] Patched 28 Qwen3DecoderLayer instances.
[pto] generated 16 tokens in 10.29s
============================================================
你好:我需要一个关于"人工智能在医疗领域的应用"主题的演讲稿
============================================================
```
