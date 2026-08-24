# Wrapper boundary: what may run on the host

## The rule

**The public callable includes the wrapper.** Every host-side tensor operation
may dispatch a real device kernel (`aclnnInplaceCopy_CastAiCore_Cast`,
`..._TransposeAiCore_Transpose`, `..._SliceAiCore_Slice`,
`aclnnCat_ConcatD_ConcatD`, …) around the PyPTO-Pro kernel launch.

So:

> **Casting, slicing, transposing, padding, concatenating, and any other data
> shape or dtype processing MUST happen inside the `@pl.jit` kernel.
> The wrapper does argument validation, output allocation, and one kernel
> launch — nothing else.**

This is not a style preference. It is the single-kernel delivery boundary and
it also determines end-to-end device cost for any caller.

## What this costs when ignored

The wrapper's share of device time is a lever for every operator. In the
measurements from which this rule was distilled, wrappers with substantial data
shaping consumed 10% to 62% of total device time.

**And it is not only time — a host-side op can fail outright on the eval
runner.** The eval container's CANN is not the dev box's (observed: eval
CANN 9.1.0 vs dev 9.2.0), and its operator inventory is not a superset: a
wrapper `.to(torch.float32)` that ran fine locally raised
`aclnnInplaceCopy failed, error code is 561103` (`EZ1013:
aclnnInplaceCopy_1_CastAiCore cannot be found`) on every fp16/bf16 case of a
real delivery, where only the fp32 cases passed. A wrapper that dispatches no
device op has no such dependency.

Two consequences follow, and they are the durable part:

- **Kernel time and callable time can move in opposite directions.** One
  operator's kernel improved by roughly a third while its end-to-end callable
  slowed, because its wrapper grew faster than the kernel shrank. The sound
  performance unit is wrapper plus kernel together.
- **Host shaping is not rare.** Across that run's generated kernels the dominant
  calls were `.to()`, `.contiguous()` and `.reshape()`, by a wide margin over
  everything else. Assume a generated wrapper has one unless the profile says
  otherwise.

> The per-operator figures behind this are a single run on one platform at one
> revision. They are not reproduced here because a threshold copied out of one
> run is exactly what this knowledge base is not for: measure your own operator
> with [`pypto-pro-op-perf-tune`](../../pypto-pro-op-perf-tune/SKILL.md) and
> read `wrapper_share` from your own profile.

## The anti-pattern

This is a real generated wrapper, renamed. Every marked line becomes a
measured device kernel:

```python
def op_wrapper(input_tensor, dim=-1, ...):
    x_fp32 = input_tensor.to(torch.float32)              # cast          <- measured
    x_transposed = x_fp32.movedim(dim, -1).contiguous()  # transpose+copy <- measured
    x_2d = x_transposed.reshape(M, D_full)
    y_2d = torch.empty(M, D_out, ...)
    op_kernel(x_2d, y_2d, ...)                      # the actual work
    y_transposed = y_2d.reshape(*non_dim_shape, D_out)
    y = y_transposed.movedim(-1, dim).contiguous()       # transpose+copy <- measured
    return y.to(out_dtype)                               # cast          <- measured
```

Four measured device ops surround one kernel launch. The kernel was written to
want a canonical FP32 contiguous 2-D input, and the host was made to produce
it. That convenience is charged at full price.

This shape recurs: the wrapper is written to hand the kernel a canonical input,
and every step of that convenience is a device kernel in the measured window.

## What to do instead

| Host does this | Move it into the kernel as |
|---|---|
| `.to(torch.float32)` on input | load the native dtype into UB, then convert on-chip — `pl.cast` at tile level or `vf.astype` at register level (there is no `vf.cast`) |
| `.to(out_dtype)` on output | `pl.cast` / `vf.astype` per tile before `store` |
| `.movedim` / `.permute` / `.transpose` | index the axis by stride/offset in the tile loop |
| `.contiguous()` | strided `DataCopy`, or fold the stride into the loop bounds |
| `.reshape` to 2-D | pass the real shape and compute the flat offset in the kernel |
| slicing an operand | pass base pointer plus an offset argument |
| `torch.cat` of operands | pass both operands and select inside the loop |
| padding to a tile multiple | `pl.set_validshape` for the tail |
| `torch.zeros` / `zeros_like` init | write every output element, or initialize in-kernel |
| `arange` / index construction | compute the index arithmetically in the kernel |
| `torch.npu.synchronize()` | delete it — synchronization belongs to the caller; this only adds an unnecessary stream wait inside the wrapper |
| a `for` loop over a TensorList launching one kernel per element | pack the elements' addresses and shapes into tiling parameters and launch **once**; iterate inside the kernel |

The last row is the expensive one for `foreach`-style operators. Their baseline is a
single fused call — eliminating per-tensor launch overhead is the entire reason those
operators exist — so launching per element reintroduces exactly what the baseline
avoids, and the gap widens with list length.

The on-chip conversion APIs in the first two rows are platform-gated; confirm
they are supported on the detected target before relying on them, and see
[precision.md](precision.md) for which dtype the converted chain has to be in.
Moving a cast into the kernel is the rule, but an unsupported API is not a
migration — escalate rather than leaving the `.to()` on the host.

A reshape that is a pure view of a contiguous tensor costs nothing and does not
appear in the profile — it is `.contiguous()`, `.to()` and the movement ops
that are charged. If in doubt, read the profile: anything named `aclnn*` in
`op_times.device_kernels` is wrapper time.

## What this boundary governs

It governs the **delivered wrapper** — the `{op}_wrapper` in `custom/<op>/test_{op}.py`
that ships with the operator and whose host time is measured.

It does **not** govern the driver functions in this KB's study samples under
`examples/samples/`. Those exist to make a sample runnable on its own, and they allocate,
reshape and synchronize to do that. They are harnesses, not delivery shapes — see
[examples/README.md](../examples/README.md). Do not copy one into a delivery and do not
read one as evidence that a call is permitted here.

## There is no pre-approval exception

`torch.empty` for output allocation is the only Torch call a delivered wrapper
may make. Nothing in `DESIGN.md` can widen that: an earlier revision of this
page let a transformation stay if the reason was recorded, which turns a hard
boundary into a promise and is how host-side `.t().contiguous()`, `torch.zeros`
and `torch.npu.synchronize` reached delivered wrappers.

If a transformation appears impossible to express in the kernel, that is a
design problem to escalate at Stage 3, not something a wrapper may absorb.
Record the blocker and the measured cost of the alternative in `DESIGN.md` so it
can be judged -- but the recording is the escalation, never the authorization,
and the wrapper stays inside the boundary while it is judged.

### A bit reinterpretation is never that exception

`pl.Ptr` formal parameters are **not dtype-checked at all**, so a tensor may be
handed to the kernel under one dtype and read under another. Whatever
reinterpretation the kernel needs — signed data carried through a `UINT32`
tile, an int64 pair viewed as two 32-bit words — happens **inside** the kernel,
and the wrapper needs no `.view()`.

This matters beyond tidiness. A host-side `.view()` or `.to()` is measured
time, and an `aclnn`-dispatching host op is also a **compatibility** risk on any
machine whose CANN inventory differs from the dev box -- the op set is not
guaranteed to be a superset. A kernel-side reinterpretation
depends on nothing outside the delivered kernel itself. (Measured on
Ascend950PR / CANN 9.2.0.)

## How this is checked

`KB_USAGE.json` must record the invariant, and the verifier runs a wrapper
audit that fails a class whose wrapper performs data shaping without a recorded
justification. Anti-cheat rules still apply in the other direction: the wrapper
must not perform the operator's *arithmetic* either, and there is still exactly
one `@pl.jit` kernel launched once.
