# maximum kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def maximum_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           b: pypto.Tensor(sl, pypto_dtype), 
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        b_s = pypto.view(b, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.maximum(a_s, b_s)
        pypto.assemble(r, [i] + zeros, out)
```
