# arange kernel reference

> Note: 无 batch 轴，沿唯一输出轴按 tile（tile_len）loop 切分；每 tile 独立生成一段（示意，段间偏移由 assemble 拼接）。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def arange_kernel(out: pypto.Tensor(ol, pypto.DT_INT32)):
    for i in pypto.loop(num_tiles, name="out_tile", unroll_list=[1]):
        seg = pypto.arange(tile_len)
        pypto.assemble(seg, [i * tile_len], out)
```
