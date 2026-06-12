---
schema_version: 2
op_name: qkv_rms_norm_rope_cache
dynamic_axes: [T]
---
# DESIGN

## 1. 当前实现概览

当前 PyPTO 代码只保留当前网络用例需要的 INT8 quant kernel：

```text
qkv_rms_norm_rope_cache_quant_kernel
```

wrapper 校验 `k_cache/v_cache` dtype：

```text
torch.int8 -> INT8 quant cache 分支
其他 dtype -> TypeError
```

两条网络用例只使用 INT8 quant cache 分支。

## 2. 动态/静态轴设计

JIT 签名只把 token 轴设为动态：

```text
qkv:       [DYNAMIC, STATIC]
cos/sin:   [DYNAMIC, STATIC]
index:     [DYNAMIC]
q_out_out: [DYNAMIC, STATIC]
```

cache、scale、head/dim 相关维度保持 static。这样可以让 TP4/TP1 两组 head/cache shape 走同一份代码逻辑，但分别静态编译。

## 3. INT8 分支计算图

```text
qkv [T, Nqkv*D]
  ├─ Q view [T,Nq*D] -> reshape [T,Nq,D]
  │      -> RMSNorm(q_gamma) -> RoPE(cos/sin)
  │      -> reshape [T,Nq*D] -> q_out
  ├─ K view [T,Nk*D] -> reshape [T,Nk,D]
  │      -> RMSNorm(k_gamma) -> RoPE(cos/sin)
  │      -> quant_int8(k_scale)
  │      -> PA_NZ INT8 cache write
  └─ V view [T,Nv*D] -> reshape [T,Nv,D]
         -> quant_int8(v_scale)
         -> PA_NZ INT8 cache write
```

量化公式：

```text
quant = cast_int8_saturate(cast_int32_rint(x / scale))
```

当前实现函数：

```text
_compute_quant
_quant_int8
_scatter_pa_nz_int8_contiguous_page0
```

## 4. API 映射

| 逻辑 | 实现 API | 说明 |
| --- | --- | --- |
| SplitVD | `pypto.view` | 按静态 width 切出 Q/K/V |
| reshape | `pypto.reshape` | 2D/3D 转换和 cache 视图转换 |
| RMSNorm | `cast/sum/sqrt/div/mul` | FP32 中间计算 |
| RoPE | `view/concat/mul/add` | half-and-half rotate |
| Q 输出 | `move` 或 `assemble` | 按 token 写回 Q 输出 |
| INT8 quant | `pypto.cast` + saturation | round 后饱和到 int8 |
| INT8 cache | reshape + `move` | 当前网络用例 page0 连续写入快路径 |

## 5. 当前网络 case 的分支

### MTP2-TP4

```text
qkv=[48,2304]
qkv_size=[16,3,18,128]
head_nums=[16,1,1]
k/v_cache=[11898,4,128,32] int8
k/v_scale=[1,128]
```

Q/K/V 大小：

```text
Q = 16*128 = 2048
K = 1*128 = 128
V = 1*128 = 128
```

### MTP2-TP1

```text
qkv=[12,9216]
qkv_size=[4,3,72,128]
head_nums=[64,4,4]
k/v_cache=[11898,16,128,32] int8
k/v_scale=[4,128]
```

Q/K/V 大小：

```text
Q = 64*128 = 8192
K = 4*128 = 512
V = 4*128 = 512
```

## 6. Tiling 与 loop

当前实现使用两套 tile config，由 wrapper 根据 `head_nums[0]` 分支选择：

```text
TP4_TILE_CONFIG:
  q_group_heads=16
  q_vec_token_tile=4
  kv_vec_token_tile=2
  stitch_function_max_num=128
  device_sched_mode=3

TP1_TILE_CONFIG:
  q_group_heads=16
  q_vec_token_tile=4
  kv_vec_token_tile=4
  stitch_function_max_num=128
  device_sched_mode=3
```

Q 路径按 `q_group_heads=16` 分组处理；K/V 路径按当前静态 `B*S` 批量处理后写 cache。RMSNorm/RoPE 使用 `set_vec_tile_shapes(tile_t, num_heads, D)`。

历史尝试中，更大 tile 或更通用 scatter 写法出现过编译耗时、性能回退或语义限制，因此当前保留网络 case 的快路径。

## 7. 功能限制

- INT8 cache 写入当前假设 `index=torch.arange(T)`，且 `T <= block_size`。
- 当前 INT8 fast path 只写 page0 连续区域；不等价于任意 `index` 的通用 PA_NZ scatter。
- `k_offset/v_offset` 非 `None` 未实现。
- `is_output_qkv=True` 未实现。
- `cache_mode` 只支持 `PA_NZ`。

## 8. 验证

当前两条网络 case 已通过 NPU precision：

```text
PASS mtp2_tp4_network_quant
PASS mtp2_tp1_network_quant
[PRECISION_PASS]
```

已记录的直接 Python benchmark：

```text
mtp2_tp4_network_quant avg_ms=0.342387 repeat=30 warmup=3 tokens=48
mtp2_tp1_network_quant avg_ms=0.684205 repeat=30 warmup=3 tokens=12
```

已记录的泳道图：

```text
mtp2_tp4_network_quant AICore E2E 98.04 us, utilization 37.23%
mtp2_tp1_network_quant AICore E2E 52.66 us, utilization 39.00%
```

## 2026-06-11 整改同步

- 新增 `qkv_rms_norm_rope_cache_quant_kernel_generic`，用于超出 TP4/TP1 快路径 head 规模的功能兜底。
- Q 继续按 `q_group_heads=16` 分组；K/V fallback 按 `kv_group_heads=2` 分组。
- K/V fallback cache 写入采用 flatten + `pypto.scatter` + assemble，支持非连续 `index`。
- Wrapper 分流保持旧性能 case 走 TP4/TP1 fast path，`Nq>64` 或 `Nk/Nv>4` 走 generic fallback。
