# linear kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def linear_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           b: pypto.Tensor([N, K], pypto_dtype), 
           bias: pypto.Tensor([N], pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + a_inner, [i] + a_zeros)
        pypto.set_cube_tile_shapes([16, 16], [16, 16], [16, 16])
        r = pypto.matmul(a_s, b, pypto_dtype, b_trans=True, extend_params={"bias_tensor": bias})
        pypto.assemble(r, [i] + o_zeros, out)
```
