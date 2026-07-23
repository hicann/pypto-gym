# gt kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def gt_kernel(a: pypto.Tensor(sl, pypto.DT_FP32),
              out: pypto.Tensor(sl, pypto.DT_BOOL)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.gt(a_s, threshold)
        pypto.assemble(r, [i] + zeros, out)
```
