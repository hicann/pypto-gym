# pad kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def pad_kernel(a: pypto.Tensor(sl, pypto_dtype),
               out: pypto.Tensor(ol, pypto_dtype),
               padding: list,
               value: float):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *o_inner)
        r = pypto.pad(a_s, padding, "constant", value)
        pypto.assemble(r, [i] + zeros_out, out)
```
