# clip kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def clip_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.clip(a_s, -1.0, 1.0)
        pypto.assemble(r, [i] + zeros, out)
```
