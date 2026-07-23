# index_add_ kernel reference

> Note: batch 轴 loop 切分（近似：按行 add）；真实 scatter 的目标索引轴依赖不可切，此处以逐行 add 示意。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def index_add__kernel(a: pypto.Tensor(sl, pypto_dtype),
           src: pypto.Tensor(sl, pypto_dtype),
           out: pypto.Tensor(sl, pypto_dtype)):
    for i in pypto.loop(batch, name="row", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros_in)
        src_s = pypto.view(src, [1] + inner, [i] + zeros_in)
        pypto.set_vec_tile_shapes(1, *inner)
        r = pypto.add(a_s, src_s)
        pypto.assemble(r, [i] + zeros_in, out)
```
