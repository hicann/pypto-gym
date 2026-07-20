# reciprocal kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def reciprocal_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.rsqrt(pypto.mul(pypto.add(pypto.abs(a_s), pypto.full([1] + inner, 1e-6, a_s.dtype)), pypto.add(pypto.abs(a_s), pypto.full([1] + inner, 1e-6, a_s.dtype))))
        pypto.assemble(r, [i] + zeros, out)
```
