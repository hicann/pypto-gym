# glu kernel reference

> Note: batch 轴 loop 切分；最后轴折半，前半 × sigmoid(后半)。sigmoid 仅支持 FP32，非 FP32 需先 cast。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def glu_kernel(a: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(ol, pypto_dtype)):
    for i in pypto.loop(batch, name="batch", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros_in)
        pypto.set_vec_tile_shapes(1, *inner_out)
        a1 = pypto.view(a_s, [1] + inner_out, [0] * (len(inner) + 1))
        a2 = pypto.view(a_s, [1] + inner_out, [0] * len(inner) + [half])
        r = pypto.mul(a1, pypto.sigmoid(a2))  # sigmoid 仅支持 FP32
        pypto.assemble(r, [i] + zeros_out, out)
```
