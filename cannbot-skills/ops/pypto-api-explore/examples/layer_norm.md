# layer_norm kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def layer_norm_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, slice_shape, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        mean = pypto.div(pypto.sum(a_s, dim=-1, keepdim=True), pypto.full(slice_out, float(D), pypto_dtype))
        centered = pypto.sub(a_s, mean)
        var = pypto.div(pypto.sum(pypto.mul(centered, centered), dim=-1, keepdim=True), pypto.full(slice_out, float(D), pypto_dtype))
        std = pypto.sqrt(pypto.add(var, pypto.full(slice_out, 1e-5, pypto_dtype)))
        normed = pypto.div(centered, std)
        pypto.assemble(normed, [i] + zeros, out)
```
