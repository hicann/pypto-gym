# PyPTO-Pro per-launch `block_dim` findings

Scope: A5 MIX kernels with host-selected launch geometry.  本页同时收录两类证据：
**源码级语义**与**一次锁板 A/B 的结论**（按下方方法可复算）。两类分别标注——
采用某个 policy 前，源码级部分可直接引用，锁板部分需在自己的目标上复现。

> **源码引用怎么核对。** 下面的 `pypto_pro/runtime/jit.py:NNNN` 指的是**已安装的
> PyPTO-Pro 发行版**，不在本仓（pypto-gym 不含 `pypto_pro/`）。核对方式是在自己的
> 安装里定位该文件，例如
> `python -c "import pypto_pro, pathlib; print(pathlib.Path(pypto_pro.__file__).parent)"`。
> 行号随版本漂移，以函数名为准而非行号；跨版本差异见第 5 条的版本表。

## Confirmed runtime semantics

1. `kernel[stream, block_dim, tiling_key]` called as `(...)` parses `block_dim`
   as a normal launch-key value.  `_TileJitKernel.__getitem__` captures it in the returned
   launcher and calls `_launch` with that value
   (`pypto_pro/runtime/jit.py:1428-1447`).
2. The shared-library caller receives `uint32_t blockDim` and launches
   `kernel<<<blockDim, nullptr, stream>>>`; the same path is used for kernels
   carrying FFTS cross-core synchronization
   (`pypto_pro/runtime/jit.py:599-623`).
3. Compilation cache identity is static shape + datatype + tiling key, not
   `block_dim` (`pypto_pro/runtime/jit.py:1552-1577`).  One compiled kernel can
   therefore be launched with a different positive core count on later calls.
4. `pl.get_block_num()` is the device API for the number of blocks in that
   launch (`pypto_pro/language/_api.py:1111-1113`).  A kernel that strides work
   by this value adjusts ownership to the selected launch geometry.
5. **The caller must clamp. Do not rely on the runtime to do it.** Compute
   `block_dim = min(requested, cores_of_the_execution_domain, task_count)` in the
   wrapper, where the domain is `PlatformInfo.core_num` for MIX kernels and
   `vector_core_num` for AIV-only kernels.

   Whether the runtime also clamps is **installation-dependent, and the current
   direction is that it does not**:

   | Build | Behaviour |
   |---|---|
   | `v9.1.0-beta.3` lineage (e.g. `d1839f034`) | `_clamp_block_dim` silently lowers an oversized value to the core count (`runtime/jit.py:1203-1227`, called at `:1446`) |
   | `9.2.0-beta.2` lineage, from `4c835a12` (2026-08-07, *"delete wrong core num check"*) | `_clamp_block_dim` is gone. `_validate_block_dim` raises `ValueError` on a non-`int` or non-positive value and the launch uses the value **unchanged** |

   This is not a detail to leave to the runtime. `pl.get_block_num()` returns the
   value actually launched, so task ownership (`range(core_id, total, get_block_num())`)
   and the count of MIX-barrier participants both derive from it. On a build that
   does not clamp, an oversized request means blocks that own no work still have
   to reach every barrier — or do not exist to reach them. Clamping in the caller
   makes the launched value knowable at the point the ownership arithmetic is
   written, on every build.

This permits a host wrapper to compute one `block_dim` from ordinary tensor
shape integers and pass it to the same JIT kernel.  It does not require, and
must not use, mutation of cached `PlatformInfo.core_num`.

> **与 `constraints/wrapper-boundary.md` 的边界关系（该页是 topology-map 里唯一的
> `mandatory_constraints`，对 architect / coder / verifier 全类生效）。** 那条强制
> 条款说的是"数据的 shape/dtype 处理必须在 kernel 内，wrapper 只做参数校验、输出
> 分配和一次 launch"。这里允许的是**从 shape 读整数算出一个 launch 几何标量**，
> 不触碰任何张量数据、不改变任何张量形状——属于"一次 launch"的参数计算，
> 不是被禁止的数据处理。
>
> 判定边界：wrapper 里只允许出现**读 `.shape` 得到 int、做整数运算、得到 `block_dim`**
> 这一条链路。一旦开始 cast / slice / transpose / pad / concat 张量，或按 shape 选择
> 不同的数据布局，就已经越界，按 wrapper-boundary 处理。

## Safety conditions when varying `block_dim` on a staged multi-stage kernel

- Select one core count for the whole launch.  It cannot change between hard
  stages because all launched AIC/AIV participants must reach the same five
  paired MIX-barrier boundaries.
- Keep each task loop in the form `range(core_id, total_tasks, get_block_num())`
  and prove exact one-owner coverage at every selected core count.
- Keep the AIV physical-id normalization and subblock barrier participation
  unchanged.  Lowering `block_dim` changes how many physical blocks launch; it
  does not turn the two AIV subblocks into independent GM workers.
- Use a positive Python `int`, cap it by both the platform count and the largest
  stage task count, and keep unseen shapes on the production fallback until the
  policy has evidence on shapes you have not measured.
- Give A/B variants distinct JIT function names.  The build directory is keyed
  by function identity, so same-name variants can otherwise reuse stale code.

实验与机械审计的产物**不随本仓保留**，因此这里不列路径——指向读者打不开的文件，
只会诱使人重跑一次已经付过钱的测量。下面这组扫描按如下方法可自行重做：
锁板独占、每个点五次 profiled launch、只改 `block_dim` 一个变量、其余（shape、dtype、
tiling key、launch 网格）全部固定。

锁板扫描（单位：微秒，同一算子的五个代表性 case，按耗时升序记为 A–E）：

| cores | case A | case B | case C | case D | case E | geomean time / 28 cores |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 203.637 | 291.351 | 426.549 | 861.328 | 3422.529 | 2.0041 |
| 12 | 148.549 | 231.659 | 347.043 | 633.982 | 2487.767 | 1.5218 |
| 16 | 123.933 | 206.311 | 306.427 | 509.048 | 1997.335 | 1.2812 |
| 20 | 103.111 | 184.970 | 276.798 | 460.163 | 1723.907 | 1.1266 |
| 24 | 92.724 | 174.526 | 260.211 | 417.375 | 1559.180 | 1.0350 |
| 28 | 92.994 | 171.138 | 254.997 | 397.086 | 1431.979 | 1.0000 |

The single smallest case favored 24 cores by well under a percent, while every
other measured case and the geomean favored the full 28.  That kernel therefore
retains the full-core launch and the dispatch candidate is rejected for that
workload.  Read this as the shape of the result — one small case dissenting by
noise does not outweigh the geomean — not as a claim that lowering `block_dim`
can never help.

## Framework limitations

- Validation of `block_dim` is not portable across builds: the older lineage
  returns a non-positive or non-`int` value **unchanged** from
  `_clamp_block_dim` (`runtime/jit.py:1213-1214`), while `4c835a12` onward
  **raises** `ValueError`. A wrapper that validates positivity itself behaves
  the same on both.
- There is no built-in per-shape launch-policy object or measured auto-tuner;
  the user wrapper owns dispatch, fallback, and regression evidence.
- No build warns about under-occupancy or barrier-vs-work trade-offs, and the
  newer lineage does not bound an oversized value at all.  Only on-board
  profiling can choose the count.
