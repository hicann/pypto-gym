# 官方指定算子样例清单

> **用途**：本文件是整个 PyPTO-Pro 工作流的**统一官方样例索引来源**。
> - **orchestrator** 资源缓存准备时，按本清单清理 `$PYPTO_DEVKIT_DIR/pro_ops/`，只保留清单内文件。
> - **material-explore** 生成 `PRO_MATERIAL_INDEX.md` §B 时，直接复制本清单内容。
> - **所有 skill / agent** 提到"官方指定算子样例""§B"时，最终都指向本文件。
>
> **维护规则**：增删样例时**只改本文件**，无需改动其他文件。路径用 `pro_ops/` 开头的相对路径（相对 `$PYPTO_DEVKIT_DIR`）。
>
> **注意**：部分 skill（如 `pypto-pro-op-develop`）在 SKILL.md 中按具体文件名引用了某些样例（如 `test_matmul_perf_asw_4k_dn_move_offset_dynamic.py`、`test_fa_with_mask.py`）。删除或重命名清单中的样例前，请全局搜索确认无具体引用残留。

| # | 算子名称 | 缓存相对路径 | 类型 | 描述 |
|---|---------|-------------|------|------|
| 1 | add | `pro_ops/element_wise/test_add.py` | elementwise | |
| 2 | matmul_8k_example | `pro_ops/matmul/test_matmul_8K_example.py` | matmul 入门 | |
| 3 | matmul_perf_asw_4k | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset.py` | matmul 性能 | |
| 4 | matmul_perf_asw_8k_k128 | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset.py` | matmul 性能 | |
| 5 | matmul_perf_asw_4k_dynamic | `pro_ops/matmul/test_matmul_perf_asw_4k_dn_move_offset_dynamic.py` | matmul 性能（动态轴） | |
| 6 | matmul_perf_asw_8k_k128_dynamic | `pro_ops/matmul/test_matmul_perf_asw_8k_k128_dn_move_offset_dynamic.py` | matmul 性能（动态轴） | |
| 7 | fa_perf_tkv_preload | `pro_ops/fa/test_fa_perf_tkv_preload_dn_vf_bufid_dynrank.py` | FlashAttention 生产级 | |
| 8 | fa_tilingkey_attn_mask | `pro_ops/fa/test_fa_tilingkey_attn_mask.py` | FlashAttention 教学 | |
| 9 | fa_with_mask | `pro_ops/fa/test_fa_with_mask.py` | FlashAttention 性能（mask + NBuffer + auto_mutex） | |
| 10 | flex_attention | `pro_ops/fa/test_flex_attention.py` | FlexAttention 性能 | |
| 11 | quant_lightning_indexer_vf | `pro_ops/lightning_indexer/test_quant_lightning_indexer_vf.py` | VF TopK | |
| 12 | layernorm_tile_group_vf | `pro_ops/vf_api/test_layernorm_tile_group_vf.py` | VF LayerNorm | |
| 13 | softmax_tile_group_vf | `pro_ops/vf_api/test_softmax_tile_group_vf.py` | VF Softmax | |
