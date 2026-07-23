# cumsum kernel reference

> Note: batch 轴 loop 切分；最后轴前缀依赖由 pypto.cumsum 在 tile 内处理，batch 行间独立。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def cumsum_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros_in)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.cumsum(a_s, -1)
        pypto.assemble(r, [i] + zeros_in, out)
```
