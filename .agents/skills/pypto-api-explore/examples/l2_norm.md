# l2_norm kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def l2_norm_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        norm_sq = pypto.sum(pypto.mul(a_s, a_s), dim=-1, keepdim=True)
        inv_norm = pypto.rsqrt(pypto.add(norm_sq, pypto.full([1] + inner[:-1] + [1], 1e-6, a_s.dtype)))
        r = pypto.mul(a_s, inv_norm)
        pypto.assemble(r, [i] + zeros, out)
```
