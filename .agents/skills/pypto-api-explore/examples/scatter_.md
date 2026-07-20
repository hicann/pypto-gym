# scatter_ kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def scatter__kernel(a: pypto.Tensor(sl, pypto_dtype), 
           idx: pypto.Tensor(sl, pypto.DT_INT32), 
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        idx_s = pypto.view(idx, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.scatter_(a_s, 1, idx_s, 1.0)
        pypto.assemble(r, [i] + zeros, out)
```
