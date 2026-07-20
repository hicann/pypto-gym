# argmax kernel reference

> Note: batch 轴 loop 切分；最后轴整块，由 pypto.argmax 求索引（keepdim 保持 rank）。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def argmax_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto.DT_INT32)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros_in)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.argmax(a_s, -1, True)
        pypto.assemble(r, [i] + zeros_out, out)
```
