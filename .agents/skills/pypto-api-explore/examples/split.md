# split kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def split_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros_in)
        pypto.set_vec_tile_shapes(1, *o_inner)
        r = pypto.view(a_s, [1] + o_inner, [0] * len([1] + inner))
        pypto.assemble(r, [i] + zeros_out, out)
```
