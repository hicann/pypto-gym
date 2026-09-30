# 保留运算契约

记录完整 cast、accumulation、saturation、rounding 和括号顺序。`(key * beta) * exp(g)` 与
`key * (beta * exp(g))` 在 bf16/fp32 边界可能不同。cast 越过 reduction 或 saved state 是
语义决策，即使最终 dtype 不变。
浮点 cube 常用 fp32 累加，整数路径常用 int32，以实际声明操作组合为准。保留初始化 tile
的 bias 和输出/state downcast。half workspace 后接 matmul 的 reference 应模拟该 half
边界；Torch 整数截断不能替代不同 tie 行为的 DSL round mode。

比较所有命名输出的名称/个数、shape、dtype、alias、defined lane 和数值。exact/bit/tolerant
由单元契约决定；零容差适用于 exact，没有通用 float 容差或禁止 exact 的规定。非零容差必须
有理由与范围。必要时检查 NaN/Inf 位置、signed zero、saturation；正态随机输入不能覆盖
overflow、cancellation 和 subnormal。

数学比较目标可以具有不同于存储输出的显式精度。`allclose` rule 可声明
`reference_dtype: float32` 或 `float64`；output/stage descriptor 仍声明实际浮点存储 dtype，
runner 分别核验两者。`exact`/`bitwise` 拒绝该字段；未声明时，actual/reference dtype 必须相同。
M058恢复 KDA 原先在
BF16 舍入前的 scaling 数学目标，`rtol=atol=0.006` 保持不变；后续 stage 消费的物理 BF16
checkpoint 也不变。保留源码定义的公式与原数值/L2 门槛；不能借此将错误的 actual dtype
转换成合规结果，或修改所有 saved-state reference。

给 validator 故意传入错误输出：全零、反号、缺失输出、错误 dtype/shape、单个损坏 stage，以及
domain 不允许的非有限值。stage 真值很小时，宽松 atol 可能接受全零。保留有依据的逐点预算；
若确有此漏洞，可增加 unit runner 的 `max_relative_l2`：
`norm(actual - expected) / norm(expected)`。expected norm 为零时 residual 必须为零。
同时记录正确实现的实测误差与被拒绝的对照，不能只因为当前 candidate 能过而选择阈值。

前向和反向的 state preparation 可能有不同精度契约。Delta Rule fused forward 在 matrix
product 后乘 beta；backward preparation 则先经过 bf16 `key * beta` 的独立舍入边界。
saved-state variant 必须命名 pre/post-chunk state、order、layout、dtype。shape 相同不表示
state 可互换，backward 所需 preparation 必须包含在 backward 单元内。

A5 FP8 causal attention demo
（[`a5_fp8_causal`](../../../kernels/ascriptor_kernels/attention/a5_fp8_causal)）先以 FP32
计算 row sum，然后才将 probability cast 为 e5m2 供 value product 使用；它的 `formula` 就是
这么写的，顺序落在 `reference.py` 里。该 cast 改变分子，不决定分母，把 row sum 放在 cast
之后的说法描述的是另一个 kernel。保留公开的 rowmax/rowsum 输出并分别比较——正是那条严格的
rowsum 比较能抓住顺序写反。

packed 格式需要独立 bit reference，两次相同 codec 不能证明 codec。uint2 要明写
`a | (b << 2) | (c << 4) | (d << 6)` 等 bit order 及是否只接受 `[0,3]` 整数。FP4 明写
nibble order、signed zero；hif8/e8m0/MX 明写 carrier dtype、scale layout、rounding 和 domain。
Register cast 的稀疏 lane 可能需要 pack/unpack 才能存储。

涉及以下边界时，按当前 A5 操作契约编写和验证：

| 边界 | 编写与验证规则 |
|---|---|
| [Predicate 路由（M10-051）](../../../library/docs/api/mask-write-semantics.md) | `compare` 将非活动逻辑 lane 置 false；带执行 mask 的 NOT/AND/OR/XOR/MOV 将非活动 predicate bit 清零——没有任何算子会保留非活动的目的**寄存器** lane，[逐算子表](../../../library/docs/api/mask-write-semantics.md)给出 66 个带 mask 算子各属六种角色中的哪一种。数据和 predicate selector 都选择两个输入分支。mask 保留完整 256 个物理 bit：b32 每四 bit 观察一位，pack/unpack 通过 128-bit 半区路由物理 bit。检查或搬运 mask 时保留完整 payload。 |
| [Online MX subnormal（M10-052）](../../../kernels/ascriptor_kernels/algorithms/online_mx) | online MX demo 的有限 FP32 domain 包括带符号 subnormal，幅值上限为 `2**40`——它的 `subnormal_payload` case 就是构造这些值的，该 domain 写在 `reference.py` 里。修复后的归一化乘以精确的 FP32 scale 倒数，该倒数是正常数范围内的 2 的整数次幂；改回 native division 可能改变 subnormal 结果。保留 E8M0 scale byte，并独立比较原始 payload/scale 输出。该结果属于指定算法和 domain，不是所有 VF subnormal 运算的保证。 |
| Scalar 平方根（M10-053） 与 BF16 转换（M10-055） | 在已验证的有限非负 FP32/FP16 domain 使用明确类型的 `scalar_sqrt`；SIMT FP32 是独立路径。即使后续 cell 被消除，模型也在操作处舍入到声明的结果类型。CCE/PTO 的普通 BF16 scalar abs/sqrt 通过显式 FP32 widening 与 SDK narrowing 执行，保留的边界 probe 覆盖其已修复范围。PyPTO scalar 限制仍属上游 gap。BF16 SIMT、register conversion 和 matrix arithmetic 的验收范围另行判断。 |
| SIMT 零符号（M10-054） | Rint/round/floor/ceil/trunc 的结果为零时保留输入符号。应比较 FP32 bit，因为浮点相等无法区分两种零。CCE wrapper 只恢复零结果的符号；`fmod` 的 truncation 中间步骤保持独立。声明硬件支持前，检查对应产物的 board 验收记录。 |
| [Native exp 与 HiFloat8 midpoint（M10-059）](../../../kernels/ascriptor_kernels/attention/a5_v8_p_stage) | V8 P-stage demo 把实测的指数误差预算写在它自己的 `main.py` 里：native `exp` 与正确舍入的 exp 相差不超过一个 FP32 ULP，在量化边界上这一个 ULP 会选到相邻编码。`reference_candidates` 把这个区间变成一组可接受的 HiFloat8 编码，而不是放宽容差；padding 保持精确。保留原始输出和错误输出对照。该预算属于 A5 上的这个 stage，不是通用 native-exp 精度保证。 |

M10-051 的原始物理路由测量覆盖 b8/b16/b32。compare/init/update 生成但尚未观察的细粒度
bit，以及单寄存器 b64 的原始布局，仍是模型约定。推导物理 mask 行为时必须保留这些限制。

NaN compare、1-ULP division、FTZ、stochastic rounding 不能跨架构推断。模型结果与硅片
测量分别带 source/toolchain/version。旧竞赛阈值和“最快”不是新产品契约。更改 compensation
或非有限值 guard 前，用边界输入区分假设。
相关所有者：library `ascriptor/ir/saturation.py`、`ascriptor/backends/sim/cast_rounding.py`、
`cast_saturation.py`、`vf_ops.py`、`vec_ops.py` 和 public facade declaration；可做[白盒探索](simulator-white-box.md)。
确证行为回写 library owner，更新受影响单元预算。
