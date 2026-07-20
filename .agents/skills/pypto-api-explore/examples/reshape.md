# reshape kernel reference

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def reshape_kernel(in_tensor: pypto.Tensor([B, D], pypto.DT_FP32),
                   out_tensor: pypto.Tensor([B, N, D], pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(64, 64)
    for b_idx in pypto.loop(B, name="b_loop", unroll_list=[1]):
        a0 = pypto.view(in_tensor, [1, D], [b_idx, 0])
        pypto.assemble(a0, [b_idx, 0, 0], out_tensor)
```
