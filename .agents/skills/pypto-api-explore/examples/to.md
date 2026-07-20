# to kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def to_kernel(a: pypto.Tensor(sl, src_dtype),
              out: pypto.Tensor(sl, dst_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.cast(a_s, dst_dtype)
        pypto.assemble(r, [i] + zeros, out)
```
