# masked_fill kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def masked_fill_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        zero = pypto.full([1] + inner, 0.0, pypto_dtype)
        mask = pypto.gt(a_s, zero)
        fill_val = pypto.full([1] + inner, -1e9, pypto_dtype)
        r = pypto.where(mask, fill_val, a_s)
        pypto.assemble(r, [i] + zeros, out)
```
