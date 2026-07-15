# gated_delta_rule

PyPTO fused kernel for the chunk gated delta rule attention used during
Qwen3.5-9B prefill. Replaces the upstream
`fla.ops.gated_delta_rule.chunk_gated_delta_rule` (or the torch fallback)
called from `Qwen3_5GatedDeltaNet.forward`.


## 产品支持情况

- Ascend 950PR：不支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## Parameter glossary

| Symbol | Meaning | Value / constraint |
|--------|---------|-------------------|
| `B`  | Batch size                       | Fixed to 1 |
| `S`  | Sequence length (prefill)        | Any S ≥ 1 (padded to a multiple of L internally) |
| `L`  | Chunk length                     | Fixed to 128 |
| `Nv` | Value head count                 | Fixed to 32 |
| `D`  | Per-head dimension               | Fixed to 128 |

## Function

```python
def gated_delta_rule_wrapper(
    query: torch.Tensor,
    key:   torch.Tensor,
    value: torch.Tensor,
    *,
    g:     torch.Tensor,
    beta:  torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
```

Signature mirrors the upstream `chunk_gated_delta_rule` so the modeling-layer
hook can swap implementations transparently.

### Inputs

| Argument | Shape | Dtype | Notes |
|----------|-------|-------|-------|
| `query`  | `[B, S, Nv, D]` | bfloat16 | not L2-normalized (done in kernel) |
| `key`    | `[B, S, Nv, D]` | bfloat16 | not L2-normalized (done in kernel) |
| `value`  | `[B, S, Nv, D]` | bfloat16 | |
| `g`      | `[B, S, Nv]`    | float32  | per-token gate (pre-cumsum) |
| `beta`   | `[B, S, Nv]`    | bfloat16 | |
| `initial_state` | `[B, Nv, D, D]` or `None` | float32 | must be `None` (prefill only) |
| `output_final_state` | — | bool | if False, second return is `None` |
| `use_qk_l2norm_in_kernel` | — | bool | must be `True` |

### Outputs

| Return | Shape | Dtype |
|--------|-------|-------|
| `core_attn_out` | `[B, S, Nv, D]` | bfloat16 |
| `last_state`    | `[B, Nv, D, D]` | float32 (or `None`) |

### Algorithm

Per chunk of L=128 rows:

1. L2-normalize q, k.
2. Decay mask `D = exp(g_cum - g_cum^T) * lower_tri`.
3. `A0 = -(k_β @ k_n^T * D) * strict_lower_tri`;
   `A = (I - A0)^-1` via 8-term truncated power series.
4. `v_out = A @ (v * β)`, `kcd = A @ (k_β * exp(g_cum))`.
5. Recurrent state carry:
   `v_new = v_out - kcd @ state`,
   `out = (q_scaled * exp(g_cum)) @ state + (q_scaled @ k_n^T * D) @ v_new`,
   `state' = state * exp(g_last) + k_decay^T @ v_new`.

### Out-of-scope behavior

If any constraint is violated, the wrapper raises `NotImplementedError`. The
modeling-layer hook (see `src/pypto_gym/transformers/qwen3_5_9b/modeling_qwen3_5.py`)
falls back to the upstream chunk function in that case.

## Testing

Tests under [`tests/ops/qwen3_5_9b/`](../../../../../../tests/ops/qwen3_5_9b/). Run
`python3 tests/ops/qwen3_5_9b/test_gated_delta_rule_qwen3_5_9b.py` (NPU) for the
`[PRECISION_PASS]` marker; the chunk case is precision-checked against the torch
golden and the recurrent/decode case is asserted to fall back upstream.

> **验证范围说明（scenario-B 限制）：** 被替换的原算子是 FLA 的融合
> `chunk_gated_delta_rule`（Triton/CUDA 实现），在 Ascend 上**不可用**——故无法在本硬件直接
> 对拍 kernel 与真实上游融合算子。golden 采用纯 torch 的 `chunk_gated_delta_rule_golden`
> （chunk 算法的逐步等价实现）作为替代基准；算法等价性由构造保证并经精度测试核对，但未对拍
> 上游 CUDA kernel。**残余风险**：golden 是 KDA/gated-delta 数学的本仓库转写，关闭方式（需
> GPU 机器）为在一台 CUDA 卡上对拍 `chunk_gated_delta_rule_golden` 与真实 `fla.ops.
> gated_delta_rule.chunk_gated_delta_rule` 并记录差异。

## 状态

| 维度 | 状态 |
|------|------|
| 单算子精度 | ✅ chunk 路径测试通过（`[PRECISION_PASS]`）— `test_gated_delta_rule_qwen3_5_9b.py`，max abs err **7.6e-5**（bf16 I/O，rtol 1e-2 / atol 5e-2）。recurrent/decode 路径显式断言回退上游（非静默跳过）。test_cases.json 的 output shape 由测试强制校验 |
| 整网集成 | ✅ 已接入 — `sys.modules` 注入 + `USE_PTO_GATED_DELTA_RULE` 开关（模型上 NPU 后再开启，step-22 时序）+ `NotImplementedError` → 上游 torch fallback |
| ACLGraph | ✅ 算子级已验证 — 已 `torch.library` 注册 `pypto::gated_delta_rule_qwen3_5`（Meta+NPU），`gated_delta_rule_pypto` 为图捕获入口；kernel 无 in-kernel synchronize（stream 顺序），故 capture-clean。`test_gated_delta_rule_aclgraph_qwen3_5_9b.py` 验证：注册 op == wrapper 逐位一致、Meta 推导正确、`torch.compile(fullgraph)` 无 graph break、**真实 torchair aclgraph（reduce-overhead）capture+replay 逐位一致（diff=0）**。整网 aclgraph 经实测（真实 9B 模型）被 **vendored modeling 的 in-place/aliasing op** 阻塞（torchair 捕获报 `refer to a single memory location, clone() needed`，与 Kimi 同类）——经注册 op 路由后捕获可越过本算子、止于该 vendored op，故**本算子非瓶颈**；整网捕获需上游 modeling 修复（trust_remote_code 范围） |
