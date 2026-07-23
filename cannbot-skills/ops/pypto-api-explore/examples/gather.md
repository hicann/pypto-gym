# gather kernel reference

> Note: 索引 batch 轴 loop 切分；被查表轴（gather_dim，即形参 dim）整块保留，由 gather HW 处理。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def gather_kernel(input_tensor: pypto.Tensor(sl, pypto_dtype),
                  index_tensor: pypto.Tensor(il, pypto.DT_INT32),
                  out: pypto.Tensor(ol, pypto_dtype),
                  dim: int):
    for i in pypto.loop(batch, name="row", unroll_list=[1]):
        idx_s = pypto.view(index_tensor, [1] + idx_inner, [i] + idx_zeros)
        r = pypto.gather(input_tensor, dim, idx_s)
        pypto.assemble(r, [i] + zeros_out, out)
```
