# topk kernel reference

> Note: batch 轴 loop 切分；最后轴取 top-k（轴整块）；输出末轴 = k。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def topk_kernel(x: pypto.Tensor(sl, pypto_dtype),
                val: pypto.Tensor(ol, pypto_dtype),
                idx: pypto.Tensor(ol, pypto.DT_INT32),
                k: int):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        x_s = pypto.view(x, [1] + inner, [i] + zeros_in)
        pypto.set_vec_tile_shapes(1, *inner)
        v, ii = pypto.topk(x_s, k, -1, True)
        pypto.assemble(v, [i] + zeros_out, val)
        pypto.assemble(ii, [i] + zeros_out, idx)
```
