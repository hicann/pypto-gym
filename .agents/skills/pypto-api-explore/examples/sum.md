# sum kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def sum_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros_in)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.sum(a_s, dim=-1, keepdim=True)
        pypto.assemble(r, [i] + zeros_out, out)
```
