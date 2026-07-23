# prod kernel reference

> Note: batch 轴 loop 切分；最后轴（归约轴）整块在 tile 内，由 pypto.prod 归约。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def prod_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros_in)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.prod(a_s, -1)
        pypto.assemble(r, [i] + zeros_out, out)
```
