# gelu_tanh kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def gelu_tanh_kernel(a: pypto.Tensor(sl, pypto_dtype), 
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.mul(a_s, pypto.mul(pypto.full([1] + inner, 0.5, a_s.dtype), pypto.add(pypto.full([1] + inner, 1.0, a_s.dtype), pypto.tanh(pypto.mul(pypto.full([1] + inner, 0.7978845608, a_s.dtype), pypto.add(a_s, pypto.mul(pypto.full([1] + inner, 0.044715, a_s.dtype), pypto.mul(a_s, pypto.mul(a_s, a_s)))))))))
        pypto.assemble(r, [i] + zeros, out)
```
