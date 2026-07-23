# one_hot kernel reference

> Note: 输入行轴 loop 切分；类别轴（num_classes）整块在 tile 内。

```python
@pypto.frontend.jit(runtime_options={"run_mode": pypto.RunMode.NPU})
def one_hot_kernel(a: pypto.Tensor(sl, pypto.DT_INT32),
                   out: pypto.Tensor(ol, pypto.DT_INT32)):
    for i in pypto.loop(batch, name="row", unroll_list=[1]):
        a_s = pypto.view(a, [1] + inner, [i] + zeros_in)
        r = pypto.one_hot(a_s, num_classes)
        pypto.assemble(r, [i] + zeros_out, out)
```
