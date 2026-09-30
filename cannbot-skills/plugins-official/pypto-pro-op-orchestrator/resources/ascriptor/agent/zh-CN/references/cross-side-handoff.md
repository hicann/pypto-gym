# 把 tile 交到另一侧

`auto_sync` 只排同侧依赖。vector 写、cube 读的缓冲——或者反过来——只能由显式 mutex 定序，
因为 event 不能跨侧，所以缺一个 mutex 时 `auto_sync` 怎么加都修不了。契约在
[跨侧归属](../../../library/docs/api/synchronization.md#cross-side-ownership)，
credit 规则在 `VcMutex` / `CvMutex` 的 docstring 里。本页给的是调用的形状、
其中一条从调用本身看不出来的次序，以及手写周期会坏的三种方式。

## 四个调用，每侧两个，都必须有

| 调用 | 所在侧 | 作用 |
|---|---|---|
| `lock()` | 生产侧 | 取一个槽，没有可用 credit 时阻塞 |
| `ready()` | 生产侧 | 发布写好的槽；第 *i* 个 `ready` 为第 *i* 个 `wait` 定序 |
| `wait()` | 消费侧 | 取得已发布的槽 |
| `free()` | 消费侧 | 在对该缓冲的**最后一次读**之后归还 credit |

`VcMutex` 的生产侧是 vector、消费侧是 cube；`CvMutex` 相反。优先写缓冲而不是数字
——`VcMutex(0, guards=left, ...)` 直接从它的类型读出 credit 数——并且一块 hand-off 缓冲
配一个 mutex：双向要两个 mutex，一个 mutex 守多块缓冲时不存在处处正确的 `depth`。

## 板卡跑过的那个周期

代码归 [A5 roundtrip](../../../library/examples/api/cube_vector_roundtrip) 所有，
下面的摘录与那个函数逐字节一致。

<!-- code-anchor:cross-side-handoff:start -->
```python
@api.kernel(mode="mix", block_dim=1)
def roundtrip(x: api.GM[api.f16, (3, 32, 128)], y: api.GM[api.f16, (16, 128)],
    o: api.GM[api.f32, (3, 32, 16)]):
    incoming = api.DBuff(api.f16, [16, 128], api.Position.UB)
    packed = api.DBuff(api.f16, [17, 128], api.Position.UB)
    left = api.DBuff(api.f16, [32, 128], api.Position.L1)
    right = api.Tensor(api.f16, [16, 128], api.Position.L1)
    product = api.DBuff(api.f32, [32, 16], api.Position.L0C)
    middle = api.DBuff(api.f32, [16, 16], api.Position.UB)
    outgoing = api.DBuff(api.f32, [16, 16], api.Position.UB)
    vector_to_cube = api.VcMutex(0, guards=left, src_end_pipe=api.Pipe.MTE3, dst_end_pipe=api.Pipe.FIX)
    cube_to_vector = api.CvMutex(1, guards=middle, src_end_pipe=api.Pipe.FIX, dst_end_pipe=api.Pipe.V)
    with api.auto_sync():
        right <<= y
        for beat in range(3):
            vector_to_cube.lock()
            begin = api.GetSubBlockIdx() * 16
            incoming[beat] <<= x[beat, begin : begin + 16, :]
            preprocess(incoming[beat], packed[beat])
            left[beat][begin : begin + 16, :] <<= packed[beat][0:16, :].nz()
            vector_to_cube.ready()

            vector_to_cube.wait()
            api.matmul(product[beat], left[beat], right, m=32, n=16, k=128)
            cube_to_vector.lock()
            middle[beat] <<= product[beat]
            cube_to_vector.ready()
            vector_to_cube.free()

            cube_to_vector.wait()
            postprocess(middle[beat], outgoing[beat])
            cube_to_vector.free()
            o[beat, begin : begin + 16, :] <<= outgoing[beat]
    return o
```
<!-- code-anchor:cross-side-handoff:end -->

这里有两个周期交错，各管一个方向。`vector_to_cube` 发布 matmul 要读的 L1 tile；
`cube_to_vector` 发布 `postprocess` 要读的 UB tile。两个消费者都严格落在各自的
`wait` 与 `free` 之间。

真正值得照抄的次序是 `vector_to_cube.free()` 在 `cube_to_vector.ready()` **之后**。
L1 tile 的最后一个读者是 matmul，而 matmul 只有在它的乘积被排出之后才算读完，
所以槽在那里归还，不在这一拍的末尾。运行序不等于源码序：每个 `free` 的位置由
它自己那块缓冲的最后一次读推出来，不是由循环的排版推出来。

## 手写周期会坏的三种方式

| 错法 | 源码读起来是什么样 | 代价 |
|---|---|---|
| 消费者落在配对之外 | `wait()` 紧接着 `free()`，真正的读在循环里更后面 | 槽在任何人读之前就被归还，而那次读没有任何东西为它定序；credit 数仍然是平的，所以 hazard prover 可能一声不响 |
| 有一侧完全没写 | 写了生产侧的 `lock`/`ready`，消费侧的 `wait`/`free` 从来没写 | 另一侧的下一个调用永远等不到——在 functional 模拟器里表现为死锁 |
| 两侧计数不配平 | 循环外多一次 `ready`，或者两次 `lock` 只配一次 `free` | 发布了没有读者消费的 token，或者生产侧越过消费侧 |

## 卡住或者结果不对的时候

卡住表现为 functional 模拟器的 `SimDeadlock`，它会点名哪个 mutex、在等哪个调用、
以及两侧的 token 计数——一直不动的那个计数就指向缺失的调用。`SimTimeout` 不是卡住：
到期限时仍有 lane 在运行，应调大 `--timeout`（主机有负载时长用例需要），不要去改握手。而一个**计数配平但结果错**
的周期是 credit 数的问题：读
[M10-076](../../../library/docs/api/synchronization.md#cross-side-ownership)，
并用 `pipesim`，它会把 hand-back 报成一对点名的未定序操作。两者的分流写在
[同步诊断](../../../library/docs/diagnosing-sync.md)里。

生命周期、复用与声明规则仍在[同步](synchronization.md)；完整的多拍推导见
[CVC 生命周期表](cube-vector-cube.md#具体生命周期表)。
