# logical_not kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def logical_not_kernel(a: pypto.Tensor(sl, pypto.DT_BOOL),
                       out: pypto.Tensor(sl, pypto.DT_BOOL)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.logical_not(a_s)
        pypto.assemble(r, [i] + zeros, out)
```
