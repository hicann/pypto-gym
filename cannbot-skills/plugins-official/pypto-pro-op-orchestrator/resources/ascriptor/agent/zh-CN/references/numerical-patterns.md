# 有明确存储边界的数值模式

在[preflight](authoring-preflight.md)后选一个模式，精确 overload 查已接受 API。
以下是有具体样例所有者的数据流方法，不代表所有 shape 通用支持。

| 模式 | 编写前推导 | 首个样例 |
|---|---|---|
| 逐元素行与尾块 | Register live lane 与 store footprint、GM 有效范围、核间行归属 | [非对齐行](../../../library/examples/api/unaligned_rows)与[mask](../../../library/examples/api/mask_semantics) |
| 行规约与广播 | Reduction 轴、结果 lane、下游 scratch 读取、规约前对无效 lane 使用 identity | [寄存器规约](../../../library/examples/api/register_reductions)；代码见[下面的摘录](#register-code) |
| 稳定 softmax | Max 前 mask、shift、exp、sum、归一化、全 masked 行契约、sum 更新后再做规定 cast | 整体看 [PFA BF16 demo](../../../kernels/ascriptor_kernels/attention/a5_pfa_bf16)；只看向量侧看 [V8 P stage](../../../kernels/ascriptor_kernels/attention/a5_v8_p_stage) |
| K 分块 matmul | 操作数 layout、首块初始化、后续累加、bias 只一次、padding 后物理 tile | [Cube bias](../../../library/examples/api/cube_bias) |
| Packed/低精度 | 逻辑格式与 carrier、bit order、defined lane、rounding/saturation、packed store 字节数 | [物理格式](../../../library/examples/api/physical_formats)与[cast](../../../library/examples/api/cast_formats) |
| Saved-state composition | Stage ABI、状态精度/版本、producer/最后 consumer、独立 leaf/整体 reference | [分解](../playbooks/decompose.md) |

## 推导短行写回

64 个 FP16 值的有效数据为 128 byte，但无 mask 的完整 B16 register store 可能访问
256 byte。保持正确物理分配，显式选择 live lane。最后 slot 会越过 allocation，前面的
行则可能静默覆盖邻行；检查 canary、最后行和最后 slot。Packed store 的 carrier 到输出
映射不同，按[内存与尾块](memory-and-tails.md#specific-boundaries)逐条推导。

<a id="reason-reduction"></a>
## 推导规约

先找到首个可能消费 padding 的操作，在该处应用 identity，再跟踪下游读取哪些结果 lane。
一个逻辑 scalar 不能证明一个元素的 scratch 足以承受完整宽度的 consumer。保留累加
精度和 cast，测试全负 max、抵消、不同行值、tie 和有效 tail。不能为通过测试悄悄改变
全 masked 行的定义。

## 推导 cast 边界

写清顺序，例如 FP32 matmul 累加 → FP32 scale/bias → ReLU → FP16 RNE → 下一次 matmul。
将 scale 移入其他指令或推迟 FP16 cast 会改变比较目标，必须独立论证。Packed 契约用
独立 host bit codec，数学目标另用数值 reference，见[精度](precision.md)。

FP16 输入用 FP32 计算——所有归一化的第一步——就是在偶数车道上走一个来回：

```text
ub_to_reg_unpack      64 个连续 f16 UB 元素 -> f16 寄存器的偶数车道
cast ZERO             这些偶数车道 -> 64 车道的 f32 寄存器      （在这里计算）
cast ZERO             f32 -> f16 寄存器的偶数车道
reg_to_ub_downsample  偶数车道 -> 64 个连续 f16 UB 元素
```

整个来回，连同规约的返回路径，就是一个可运行的 unit：
[row_norm_fp16](../../../library/examples/api/row_norm_fp16)。它把四行 FP16 在 FP32 里
归一化——`cadd` 归到 lane 0，`reg_to_ub_single` 存进一个 UB 单元，`ub_to_reg_single` 广播回每条
车道，缺的 `rsqrt` 用 `sqrt` 加 `div` 顶上——在 `sim`、`pipesim` 和卡上的 CCE 三处都与 FP64
参考逐位相等。

`eps` 也在那里才有对照。它的 `zero_row` case 送一行全零进去：加了偏置这一行是有限的零，不加就
除以零、结果是 NaN。在任何普通输入域里 `eps` 都远低于一个 FP16 ulp，所以没有这一行的用例矩阵，
分不出实现了 eps 和漏掉 eps 的两种写法——**不可能失败的对照不是对照**。

`reg_layout` 在 half/single 这一对上选的是车道奇偶（f16 和 bf16 一样），不是寄存器的前后半段；
`ONE`（奇数车道）必须配 `MaskReg(DT.half)`：用 b32 mask 时同一条 cast 返回全零且不报错，卡上和
模型上都如此。`ONE` 在 PyPTO-Pro 上也发射不出来。这个字段会不会被读取，取决于那一对属于哪个形状族：64 位和同宽的那些形式
不带 part 选择子，在那里 `ONE` 静默等同于 `ZERO`。族表见
[cast 策略](../../../library/docs/api/formats.md#cast-policy-and-destination-state)。这两条事实和背后的实测都归
[cast 策略](../../../library/docs/api/formats.md#cast-policy-and-destination-state)。
A5 VF 没有 `rsqrt`——[manifest](../../../library/docs/api/manifest.json) 只在 `a2_vector`
scope 里声明它——所以归一化要写成先 `sqrt` 再 `div`。

每个方法检查首个相关合法/非法边界、输出覆盖、相邻数据保留，以及重复存储时的同核复用。
保留错误 cast/mask 或漏输出作为比较器对照。Primitive、模型或 lowering 不一致时进入
[调试](../playbooks/debug.md)。

<a id="register-code"></a>
## 从真实寄存器代码落地

下面摘自 [register_reductions/kernel.py](../../../library/examples/api/register_reductions/kernel.py) 的 `reduce_vf`。
它展示 load → 规约 → 按结果布局 store；完整 kernel 的 DMA、imports、输入域与 reference 由 owner 提供。

<!-- code-anchor:reduction:start -->
```python
@vf()
def reduce_vf(xf: Tensor, xi: Tensor, xl: Tensor, of: Tensor, oi: Tensor, ol: Tensor):
    rf = Reg(DT.float)
    ri = Reg(DT.int)
    rl = Reg(DT.int64)
    rf <<= xf[0]
    ri <<= xi[0]
    rl <<= xl[0]
    for src, out, dt, cols in ((rf, of, DT.float, 64), (ri, oi, DT.int, 64), (rl, ol, DT.int64, 32)):
        r_add = Reg(dt)
        r_max = Reg(dt)
        r_min = Reg(dt)
        cadd(r_add, src)
        cmax(r_max, src)
        cmin(r_min, src)
        out[0] <<= r_add  # one full register per row (element offsets)
        out[cols] <<= r_max
        out[2 * cols] <<= r_min
```
<!-- code-anchor:reduction:end -->
