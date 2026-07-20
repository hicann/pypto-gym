# moe_routing kernel reference

> Note: token（行）轴 loop 切分；专家轴（最后轴）整块在 tile 内（归约）。sigmoid/sum 仅支持 FP32。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def moe_routing_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="token", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros_in)
        pypto.set_vec_tile_shapes(1, *inner)
        w = pypto.sigmoid(a_s)  # sigmoid 仅支持 FP32
        s = pypto.sum(w, dim=-1, keepdim=True)
        r = pypto.div(w, s)
        pypto.assemble(r, [i] + zeros_in, out)
```
