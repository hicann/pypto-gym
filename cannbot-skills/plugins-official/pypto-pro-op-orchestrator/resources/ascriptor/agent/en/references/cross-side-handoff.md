# Hand a tile to the other side

`auto_sync` orders one side. A buffer written by vector and read by cube — or the reverse — is
ordered only by an explicit mutex, because events cannot cross sides, so no amount of `auto_sync`
repairs a missing one. The contract is
[cross-side ownership](../../../library/docs/api/synchronization.md#cross-side-ownership) and the
credit rule is in the `VcMutex` / `CvMutex` docstrings. This page is the shape of the calls, the
one ordering that is not obvious from them, and the three ways a hand-written cycle breaks.

## Four calls, two a side, all required

| Call | Side | What it does |
|---|---|---|
| `lock()` | producer | Take a slot, blocking until a credit is free |
| `ready()` | producer | Publish the written slot; `ready` of cycle *i* orders `wait` of cycle *i* |
| `wait()` | consumer | Acquire the published slot |
| `free()` | consumer | Return the credit, after the LAST read of that buffer |

`VcMutex` makes vector the producer and cube the consumer; `CvMutex` is the reverse. Name the
buffer rather than a number — `VcMutex(0, guards=left, ...)` reads the credit count out of its
type — and give one hand-off buffer one mutex: two directions need two mutexes, and one mutex
guarding several buffers has no `depth` that is right everywhere.

## The cycle a board has run

The [A5 roundtrip](../../../library/examples/api/cube_vector_roundtrip) owns this code;
the excerpt follows that function exactly.

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

Two cycles interleave here, one a direction. `vector_to_cube` publishes the L1 tile the matmul
reads; `cube_to_vector` publishes the UB tile `postprocess` reads. Each consumer sits strictly
between its own `wait` and `free`.

The ordering worth copying is `vector_to_cube.free()` **after** `cube_to_vector.ready()`. The L1
tile's last reader is the matmul, and the matmul is only finished once its product has been
drained, so the slot comes back there and not at the end of the beat. Running order is not source
order: derive each `free` from the last read of its own buffer, not from the layout of the loop.

## Three ways the cycle breaks

| Mistake | How the source reads | What it costs |
|---|---|---|
| The consumer is outside the pair | `wait()` immediately followed by `free()`, with the real read later in the loop | The slot returns before anything read it and the read is ordered by nothing; the credits still balance, so the hazard prover can stay silent |
| One side is missing entirely | producer `lock`/`ready` written, consumer `wait`/`free` never | The other side's next call never unblocks — a deadlock in the functional simulator |
| The counts do not balance | an extra `ready` outside the loop, or one `free` for two `lock`s | A published token no reader consumes, or a producer that overruns the consumer |

## When it stalls or comes out wrong

A stall is a `SimDeadlock` from the functional simulator, and it names the mutex, the call it is
waiting for and both token counters — a counter that never moves is the call that is missing. A
`SimTimeout` is not a stall: the time limit passed while a lane was still running, so raise
`--timeout` (long cases on a loaded host need it) and leave the handshake alone. A
*wrong result* with a balanced cycle is the credit count instead: read
[M10-076](../../../library/docs/api/synchronization.md#cross-side-ownership), and use
`pipesim`, which reports the hand-back as a named unordered pair. The split between the two is in
[diagnosing sync](../../../library/docs/diagnosing-sync.md).

Lifetimes, reuse and declaration rules stay in [synchronization](synchronization.md); the worked
multi-beat derivation is the [CVC lifetime table](cube-vector-cube.md#concrete-lifetime-table).
