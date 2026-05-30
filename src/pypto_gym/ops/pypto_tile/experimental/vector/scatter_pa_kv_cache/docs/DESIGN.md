---
schema_version: "2.1"
op_name: scatter_pa_kv_cache
status: final
last_updated: "2026-05-13"

compute_kind: vector
dtypes: ["bf16"]
dynamic_axes: ["num_tokens", "num_blocks"]
precision: { rtol: 0.0078125, atol: 0.0001 }
---

# scatter_pa_kv_cache 设计方案

## 1. 计算图与精度路由

### 1.1 API 调用序列

| 步骤 | 操作 | PyPTO API | 输入 dtype | 输出 dtype | 输出 shape | 备注 |
|------|------|-----------|------------|------------|-----------|------|
| 1 | reshape cache to 2D | `pypto.reshape` | BF16 | BF16 | [num_blocks * block_size, num_heads * head_size] | inplace=True，合并 num_heads 维度 |
| 2 | reshape src to 2D | `pypto.reshape` | BF16 | BF16 | [num_tokens, num_heads * head_size] | inplace=True |
| 3 | reshape index to 2D | `pypto.reshape` | INT32 | INT32 | [num_tokens, 1] | 满足 scatter_update 的 index shape 要求 |
| 4 | scatter update key | `pypto.scatter_update` | BF16 | BF16 | [num_blocks * block_size, num_heads * head_size] | 原地更新 key_cache，dim=-2 |
| 5 | scatter update value | `pypto.scatter_update` | BF16 | BF16 | [num_blocks * block_size, num_heads * head_size] | 原地更新 value_cache，dim=-2 |
| 6 | write back cache | `.move()` 或 `[:]=` | BF16 | BF16 | [num_blocks, block_size, num_heads, head_size] | 将 2D 更新结果写回原始 4D cache |

### 1.2 精度路由

```text
输入(BF16) → reshape(BF16) → scatter_update(BF16) → 输出(BF16)
```

**说明**：
- scatter_update 支持 BF16 dtype，无需类型转换
- 整个计算链保持 BF16 dtype，无精度损失
- 输入输出 dtype 一致，符合算子规格要求

### 1.3 替代方案（已排除）

| 替代方案 | 排除原因 |
|---------|---------|
| 使用 4D scatter_update | key_cache 第三维为 num_heads（=2），scatter_update 要求第三维必须为 1，不符合约束 |
| 使用 `pypto.scatter_` | scatter_ 是通用版本，scatter_update 是专为 PA KV cache 优化的版本，性能更好 |
| 使用 `pypto.index_put_` | 参数更复杂，且不支持 PA KV cache 的特殊格式 |
| 不 reshape，直接 loop 每个 token | 效率低，无法利用 scatter_update 的批量更新能力 |

### 1.4 关键决策依据

**决策 1：使用 2D scatter_update**

- **依据**：API 文档显示 scatter_update 4D 格式要求 `[blockNum, blockSize, 1, d]`，第三维必须为 1
- **现状**：SPEC 中 key_cache 为 `[num_blocks, block_size, num_heads, head_size]`，num_heads = 2
- **结论**：必须 reshape 为 2D 格式 `[num_blocks * block_size, num_heads * head_size]`

**决策 2：使用 `.move()` 方法写回**

- **依据**：参考实现 `glm_attention_fusion.py` Line 461-462 使用 `.move()`
- **优势**：自动处理 shape 转换，将 2D tensor 数据写回原始 4D tensor
- **结论**：使用 `.move()` 进行原地更新

---

## 2. 数据规格

### 2.1 Kernel 函数签名

```python
@pypto.frontend.jit(runtime_options={"run_mode": "npu"})
def scatter_pa_kv_cache_kernel(
    key: pypto.Tensor([pypto.DYNAMIC, num_heads, head_size], pypto.DT_BF16),
    key_cache: pypto.Tensor([num_blocks, block_size, num_heads, head_size], pypto.DT_BF16),
    slot_mapping: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),
    value: pypto.Tensor([pypto.DYNAMIC, num_heads, head_size], pypto.DT_BF16),
    value_cache: pypto.Tensor([num_blocks, block_size, num_heads, head_size], pypto.DT_BF16),
):
```

**说明**：
- `num_tokens` 标为 `pypto.DYNAMIC`（运行时确定）
- `num_blocks` 在 kernel 内使用静态值（但支持动态输入）
- `num_heads`, `head_size`, `block_size` 为编译期常量（典型配置：2, 256, 128）

### 2.2 动态轴分析

| 维度名 | 是否动态 | 取值范围 / 常量 | 标注方式 |
|--------|---------|-----------------|---------|
| num_tokens | ✓ 是 | [1, 16384] | `pypto.DYNAMIC` |
| num_blocks | ✓ 是 | [100, 10000] | 编译期静态，但输入 tensor shape 动态 |
| num_heads | ✗ 否 | 2（常量） | 不标 DYNAMIC |
| head_size | ✗ 否 | 256（常量） | 不标 DYNAMIC |
| block_size | ✗ 否 | 128（常量） | 不标 DYNAMIC |

**动态轴处理策略**：
- `num_tokens` 动态 → 使用 `pypto.loop` 遍历
- `num_blocks` 动态 → 影响 cache 总大小，但 scatter_update 不需要 loop num_blocks
- Loop 只针对 `num_tokens`，按 tile 分批更新

### 2.3 值类型分析（避免 SymbolicScalar 误用）

| 变量 | 来源 | 类型 | 注意事项 |
|------|------|------|---------|
| `num_tokens` | `key.shape[0]`（动态轴） | SymbolicScalar | 不可用于 Python `if/range`，必须用 `pypto.loop` |
| `tile_tokens` | 字面量 | int | 常规使用，tile 大小常量 |
| `offset` | `loop_idx * tile_tokens` | SymbolicScalar | 只用于 `view` 的 offset 参数，不可索引 list |
| `actual_tokens` | `(num_tokens - offset).min(tile_tokens)` | SymbolicScalar | 使用 `.min()` 方法，不能用 Python `min()` |
| `num_heads * head_size` | `key.shape[1] * key.shape[2]` | int | 编译期常量，可常规计算 |
| `block_size` | `key_cache.shape[1]` | int | 编译期常量 |

---

## 3. Tiling 策略

### 3.1 箱子类型

**Vector**（scatter_update 是索引更新操作，不涉及矩阵乘法）

### 3.2 Tiling 推导

#### 同时驻留 UB 的 Tensor

| Tensor | 用途 | shape | dtype | 大小估算（tile） |
|--------|------|-------|-------|-----------------|
| index_tile | slot_mapping tile | [tile_tokens, 1] | INT32 | tile_tokens × 1 × 4B |
| key_tile | key data tile | [tile_tokens, num_heads * head_size] | BF16 | tile_tokens × 512 × 2B |
| value_tile | value data tile | [tile_tokens, num_heads * head_size] | BF16 | tile_tokens × 512 × 2B |

**总 UB 占用**：
```
UB_total = (index_tile + key_tile + value_tile)
         = tile_tokens × (4 + 1024 + 1024) B
         = tile_tokens × 2052 B
```

#### 推导步骤

1. **尾轴对齐**：BF16 → 16 元素对齐
   - num_heads * head_size = 512 > 16，满足对齐要求

2. **UB 预算**：假设 UB 容量约 128KB
   - 最大 tile_tokens = 128KB / 2052B ≈ 62
   - 考虑其他 buffer 和安全余量，取 tile_tokens = 8 ~ 16

3. **展开约束**：`(num_tokens / tile_tokens) × tensor_count ≤ 18000`
   - 最大 num_tokens = 16384
   - tile_tokens = 8 → 16384 / 8 × 3 = 6144 < 18000 ✓
   - tile_tokens = 16 → 16384 / 16 × 3 = 3072 < 18000 ✓

4. **scatter_update 约束**：
   - TileShape 维度数 = src 维度数（2D）
   - 尾轴 d（=512）不可切分

#### 最终 tile 配置

```python
# 推荐配置
tile_tokens = 8  # 每次 scatter 更新 8 个 token

pypto.set_vec_tile_shapes(tile_tokens, num_heads * head_size)
# 具体：pypto.set_vec_tile_shapes(8, 512)
```

### 3.3 替代方案

| 备选 tile | 否决理由 |
|-----------|---------|
| tile_tokens = 32 | UB 占用 32 × 2052B = 64KB，接近 UB 上限，风险高 |
| tile_tokens = 1 | 展开次数 16384，接近上限 18000，编译效率低 |
| tile_tokens = 64 | UB 占用 128KB，超出 UB 容量 |
| 4D tile `[tile_tokens, 1, num_heads, head_size]` | 不适用，使用 2D scatter_update |

---

## 4. Loop 与数据流

### 4.1 维度判定

| 轴 | 维度大小 | 编译期 / 运行期 | Loop 处理 |
|----|---------|----------------|----------|
| num_tokens | [1, 16384] | 运行期 | `pypto.loop(num_tokens_loop, name="scatter_loop")` |
| tile_tokens | 8（常量） | 编译期 | loop 内固定 tile |
| num_heads * head_size | 512（常量） | 编译期 | 不需要 loop |

**Loop 策略**：
- 计算 `num_tokens_loop = (num_tokens + tile_tokens - 1) // tile_tokens`
- 使用 `pypto.loop` 遍历 num_tokens_loop
- 每个 iteration 处理 tile_tokens 个 token 的 scatter update

### 4.2 完整伪代码

```python
@pypto.frontend.jit(runtime_options={"run_mode": "npu"})
def scatter_pa_kv_cache_kernel(
    key: pypto.Tensor([pypto.DYNAMIC, num_heads, head_size], pypto.DT_BF16),        # [num_tokens, 2, 256]
    key_cache: pypto.Tensor([num_blocks, block_size, num_heads, head_size], pypto.DT_BF16),  # [9760, 128, 2, 256]
    slot_mapping: pypto.Tensor([pypto.DYNAMIC], pypto.DT_INT32),                    # [num_tokens]
    value: pypto.Tensor([pypto.DYNAMIC, num_heads, head_size], pypto.DT_BF16),      # [num_tokens, 2, 256]
    value_cache: pypto.Tensor([num_blocks, block_size, num_heads, head_size], pypto.DT_BF16),  # [9760, 128, 2, 256]
):
    # 获取 shape 信息（注意类型标注）
    num_tokens = key.shape[0]            # SymbolicScalar（动态轴）
    num_blocks = key_cache.shape[0]      # int（编译期常量）
    block_size = key_cache.shape[1]      # int（=128）
    num_heads = key.shape[1]             # int（=2）
    head_size = key.shape[2]             # int（=256）
    
    kv_dim = num_heads * head_size       # int（=512）
    
    # Step 1: reshape cache to 2D（inplace）
    # 合并 block_size 和 num_heads 维度到最后一维
    cache_2d_shape = [num_blocks * block_size, kv_dim]  # [9760*128, 512] = [1249280, 512]
    key_cache_2d = pypto.reshape(key_cache, cache_2d_shape, inplace=True)     # BF16
    value_cache_2d = pypto.reshape(value_cache, cache_2d_shape, inplace=True)  # BF16
    
    # Step 2: reshape src to 2D（inplace）
    key_2d = pypto.reshape(key, [num_tokens, kv_dim], inplace=True)     # BF16, [num_tokens, 512]
    value_2d = pypto.reshape(value, [num_tokens, kv_dim], inplace=True)  # BF16, [num_tokens, 512]
    
    # Step 3: reshape slot_mapping to 2D
    slot_mapping_2d = pypto.reshape(slot_mapping, [num_tokens, 1], inplace=True)  # INT32, [num_tokens, 1]
    
    # Tiling 配置
    tile_tokens = 8  # int
    pypto.set_vec_tile_shapes(tile_tokens, kv_dim)  # [8, 512]
    
    # Loop 计算
    num_tokens_loop = (num_tokens + tile_tokens - 1) // tile_tokens  # SymbolicScalar
    
    # Step 4-6: scatter update with loop
    for loop_idx in pypto.loop(num_tokens_loop, name="scatter_loop", idx_name="loop_idx"):
        # 计算当前 tile 的 offset 和实际 token 数（处理尾块）
        offset = loop_idx * tile_tokens                                  # SymbolicScalar
        actual_tokens = (num_tokens - offset).min(tile_tokens)           # SymbolicScalar
        
        # view 切出当前 tile（使用 valid_shape 处理动态边界）
        index_view = pypto.view(
            slot_mapping_2d, 
            [tile_tokens, 1], 
            [offset, 0], 
            valid_shape=[actual_tokens, 1]
        )  # INT32, [tile_tokens, 1]，实际 [actual_tokens, 1]
        
        key_view = pypto.view(
            key_2d, 
            [tile_tokens, kv_dim], 
            [offset, 0], 
            valid_shape=[actual_tokens, kv_dim]
        )  # BF16, [tile_tokens, 512]，实际 [actual_tokens, 512]
        
        value_view = pypto.view(
            value_2d, 
            [tile_tokens, kv_dim], 
            [offset, 0], 
            valid_shape=[actual_tokens, kv_dim]
        )  # BF16, [tile_tokens, 512]，实际 [actual_tokens, 512]
        
        # scatter_update（原地更新 cache）
        # 注意：scatter_update 返回更新后的 input tensor
        key_cache_2d_result = pypto.scatter_update(key_cache_2d, -2, index_view, key_view)   # BF16
        value_cache_2d_result = pypto.scatter_update(value_cache_2d, -2, index_view, value_view)  # BF16
        
        # 写回原始 4D cache（使用 .move() 方法）
        key_cache.move(key_cache_2d_result)
        value_cache.move(value_cache_2d_result)
```

### 4.3 跨迭代状态

| 状态名 | 初始化 | 更新方式 | submit_before_loop |
|--------|--------|---------|--------------------|
| 无跨迭代状态 | — | — | — |

**说明**：scatter_update 每次独立更新不同的 cache 位置，无跨迭代依赖。

### 4.4 尾块处理

- **方案**：使用 `valid_shape` 参数
- **处理逻辑**：
  ```python
  actual_tokens = (num_tokens - offset).min(tile_tokens)
  index_view = pypto.view(..., valid_shape=[actual_tokens, 1])
  ```
- **作用**：最后一个 tile 可能少于 tile_tokens，valid_shape 确保只更新实际存在的 token

---

## 5. 约束自检清单

| # | 约束 | 是否满足 | 备注 |
|---|------|---------|------|
| 1 | 所有 sum 输入已转 FP32 | ✓ N/A | 本算子不使用 sum |
| 2 | matmul 两侧 dtype 一致 | ✓ N/A | 本算子不使用 matmul |
| 3 | TileShape 维度数 = 操作数维度数 | ✓ 满足 | TileShape [8, 512] 与 src [tile_tokens, 512] 维度一致（2D） |
| 4 | 尾轴满足对齐 | ✓ 满足 | kv_dim = 512 > 16（BF16 对齐要求） |
| 5 | 同阶段 UB 占用 ≤ 容量 | ✓ 满足 | 8 × 2052B = 16KB << UB（≈128KB） |
| 6 | 表达式展开 < 18000 | ✓ 满足 | 16384 / 8 × 3 = 6144 < 18000 |
| 7 | 输出经 `[:]` / `assemble` / `.move()` 显式写回 | ✓ 满足 | 使用 `key_cache.move(result)` |
| 8 | 无 view/assemble 同张量回环 | ✓ 满足 | view 用于 src/index，move 用于 cache，无回环 |
| 9 | 动态轴标 `pypto.DYNAMIC` | ✓ 满足 | num_tokens 标为 DYNAMIC |
| 10 | 动态 loop 提供 `unroll_list` | ✗ 未提供 | 建议添加 `unroll_list=[8, 4, 2, 1]` |
| 11 | 跨迭代状态用 `submit_before_loop=True` | ✓ N/A | 无跨迭代状态 |
| 12 | 尾块用 `valid_shape` 处理 | ✓ 满足 | 使用 `valid_shape=[actual_tokens, kv_dim]` |
| 13 | 无 SymbolicScalar 用作 `**` / list index / Python `if` | ✓ 满足 | offset 只用于 view 参数，未用作 list index |

### 开放问题

| # | 问题 | 影响范围 | 待解决方式 |
|---|------|---------|-----------|
| 1 | unroll_list 是否必需？ | Loop 编译性能 | 实现阶段验证是否需要 `unroll_list=[8, 4, 2, 1]` |
| 2 | num_blocks 动态是否需要特殊处理？ | Cache reshape | 当前设计使用编译期 num_blocks，需验证动态场景 |

---

## 6. 验证方案

### 6.1 测试配置

| 用例 | 输入 shape | dtype | 重点验证 |
|------|----------|-------|---------|
| 配置1_性能P0 | key: [2633, 2, 256]; key_cache: [9760, 128, 2, 256]; slot_mapping: [2633] | BF16 | 适中序列长度，验证 scatter 正确性 |
| 配置2_功能P0 | key: [7902, 2, 256]; key_cache: [9760, 128, 2, 256]; slot_mapping: [7902] | BF16 | 较长序列，验证动态 shape 处理 |
| 配置3_边界P0 | key: [16384, 2, 256]; key_cache: [9760, 128, 2, 256]; slot_mapping: [16384] | BF16 | 最大序列长度，验证尾块处理 |
| 边界_case1 | num_tokens=1 | BF16 | 最小 tokens，验证单 token 更新 |
| 边界_case2 | slot_mapping 重复 | BF16 | 验证后写入覆盖前写入逻辑 |

### 6.2 精度容忍度

| dtype | rtol | atol |
|-------|------|------|
| BF16 | 0.0078125 | 0.0001 |

### 6.3 验证方法

1. **精度验证**：
   - 使用 `scatter_pa_kv_cache_golden.py` 作为参考实现
   - 对比 key_cache 和 value_cache 的更新结果
   - 验证每个 token 的 scatter 位置正确

2. **功能验证**：
   - 验证 slot_mapping 索引范围合法性
   - 验证重复 slot_mapping 的覆盖逻辑
   - 验证动态 num_tokens 的尾块处理

3. **性能验证**：
   - 使用典型配置测试性能
   - 目标：首跑精度成功性能的 2 倍

---

## 7. 参考资源

### 7.1 API 文档

- `docs/api/operation/pypto-scatter_update.md` — scatter_update API 签名与约束
- `docs/api/config/pypto-set_vec_tile_shapes.md` — Vector Tiling 配置
- `docs/api/others/pypto-from_torch.md` — 入口约束

### 7.2 参考实现

- `examples/scatter_pa_kv_cache.py` — Golden 实现（100% 匹配）
- `models/glm_v4_5/glm_attention_fusion.py` — 生产级实现（Line 461-462）
- `models/deepseek_v32_exp/mla_prolog_quant_impl.py` — scatter_update 用法参考

---

## 8. 设计总结

### 8.1 核心决策

1. **API 选择**：使用 `pypto.scatter_update`（专为 PA KV cache 优化）
2. **数据格式**：2D 格式（reshape cache/src/index）
3. **Tiling**：`[tile_tokens, kv_dim]`，tile_tokens = 8
4. **Loop**：按 num_tokens loop，使用 valid_shape 处理尾块
5. **原地更新**：使用 `.move()` 方法写回原始 cache

### 8.2 实现复杂度评估

- **API 调用复杂度**：低（单一 scatter_update API，无复杂组合）
- **Tiling 复杂度**：低（2D tile，尾轴不可切，约束简单）
- **Loop 复杂度**：低（单层 loop，无跨迭代依赖）
- **精度路由复杂度**：低（无 dtype 转换，全程 BF16）
- **总体复杂度**：**低**

### 8.3 预估风险点

1. **动态 num_blocks 处理**：需验证 num_blocks 动态时 reshape 是否正确
2. **inplace reshape**：需确认 `inplace=True` 的 reshape 与 scatter_update 配合正确
3. **unroll_list**：可能需要添加以优化 loop 编译性能

---

**设计完成时间**: 2026-05-13
**设计状态**: 已收敛
**迭代轮数**: 4 轮（API → Tiling → Loop → 约束验证）