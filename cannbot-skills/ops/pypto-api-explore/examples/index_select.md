# index_select kernel reference

> Note: 外层 batch 轴 loop 切分；被选轴（形参 dim）整块保留，由 index_select 处理。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def index_select_kernel(x: pypto.Tensor(sl, pypto_dtype),
                        index: pypto.Tensor(il, pypto.DT_INT32),
                        out: pypto.Tensor(ol, pypto_dtype),
                        dim: int):
    for i in pypto.loop(batch, name="row", unroll_list=[1]):
        x_s = pypto.view(x, [1] + inner, [i] + zeros_in)
        r = pypto.index_select(x_s, dim, index)
        pypto.assemble(r, [i] + zeros_out, out)
```
