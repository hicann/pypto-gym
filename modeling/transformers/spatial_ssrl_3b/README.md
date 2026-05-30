# Spatial-SSRL-3B 迁移说明

## 基本信息

| 项目 | 值 |
|------|------|
| HuggingFace | [internlm/Spatial-SSRL-3B](https://huggingface.co/internlm/Spatial-SSRL-3B) |
| 权重目录 | /data/h00520348/optimize525/models/spatial_ssrl_3b |
| 代码来源 | transformers包内置 + 本地定制 (core/) |
| transformers版本 | 4.55.4 |
| 运行命令 | `python3 scripts/ask_spatial_ssrl_3b.py --device 0` |

## 目录结构

```
spatial_ssrl_3b/
├── config.json                          # 模型配置（含auto_map）
├── model-00001-of-00002.safetensors     # 模型权重分片1
├── model-00002-of-00002.safetensors     # 模型权重分片2
├── model.safetensors.index.json         # 权重索引
├── tokenizer.json / tokenizer_config.json
├── vocab.json / merges.txt
├── generation_config.json
├── preprocessor_config.json
├── video_preprocessor_config.json
├── chat_template.jinja
├── core/                                # 定制化模型代码
│   ├── modeling_qwen2_5_vl.py
│   └── configuration_qwen2_5_vl.py
├── spatial_ssrl_3b_pto_kernels/         # PyPTO融合算子
│   ├── rms_norm/
│   └── rope/
└── scripts/
    ├── ask_spatial_ssrl_3b.py           # 推理脚本
    ├── bench_spatial_ssrl_3b.sh         # 性能测试脚本
    ├── prof_spatial_ssrl_3b.sh          # 性能采集脚本
    └── README.md                         # 详细使用说明
```

## 使用方法

### 基本推理
```bash
python3 scripts/ask_spatial_ssrl_3b.py --device 0 --prompt "你好"
```

### PyPTO 模式推理（融合算子加速）
```bash
python3 scripts/ask_spatial_ssrl_3b.py --device 0 --use_pto --prompt "你好"
```

### PyPTO + aclgraph 模式
```bash
python3 scripts/ask_spatial_ssrl_3b.py --device 0 --use_pto --use_partial_aclgraph --prompt "你好"
```

### 自定义模型路径
```bash
python3 scripts/ask_spatial_ssrl_3b.py --device 0 --model-path /custom/path --prompt "你好"
```

### 从文件读取提示词
```bash
python3 scripts/ask_spatial_ssrl_3b.py --device 0 --sentence_file prompts.txt
```

## PyPTO 算子集成

| 算子 | 位置 | 状态 |
|------|------|------|
| RMS Norm | spatial_ssrl_3b_pto_kernels/rms_norm/ | ✅ 已集成 |
| RoPE | spatial_ssrl_3b_pto_kernels/rope/ | ✅ 已集成 |

启用方式：`--use_pto` 参数会同时启用 `USE_PTO_RMS_NORM` 和 `USE_PTO_ROPE`

## 性能指标

| 指标 | Baseline | PyPTO (RMS Norm + RoPE) | 提升 |
|------|----------|-------------------------|------|
| 推理耗时 | 2.062s | 1.505s | 27% |
| 吞吐量 | 14.5 tokens/s | 19.9 tokens/s | **37%** |
| 峰值显存 | 7208.2MB | 7208.2MB | 持平 |

测试条件：Prompt "你好，介绍一下华为昇腾NPU"，Output 30 tokens，NPU设备

## 注意事项

- 本模型为多模态视觉语言模型（Qwen2.5-VL架构，model_type=qwen2_5_vl）
- 纯文本推理可通过 processor.apply_chat_template 实现
- 模型参数量约3B，float16精度下单卡64GB HBM可运行
- auto_map 配置指向本地 core/ 目录下的定制化实现

## Citation

```bibtex
@article{liu2025spatial,
  title={Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning},
  author={Liu, Yuhong and Zhang, Beichen and Zang, Yuhang and Cao, Yuhang and Xing, Long and Dong, Xiaoyi and Duan, Haodong and Lin, Dahua and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2510.27606},
  year={2025}
}
```