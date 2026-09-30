# SIMT scalar work and atomics

`simt` is available on A5/A5PR. A decorated function is called from a kernel
and runs on its vector side. Use typed tensor/scalar arguments and the
restricted source language described in [RFC-0002](../rfc/0002-frontend-static-subset.md).
The current compiler specializes and caches callees by argument signature;
the historical `SimtModule` object and first-call dtype-lock lifecycle are
not the public API.

Thread counts are 64, 128, 256, 512, 1024 or 2048. `simt_thread_id()` and
`simt_thread_num()` describe the local thread group. Block queries describe
the core topology; they must not be substituted for a global vector
participant index when each mixed core invokes both vector subblocks.
The [atomic family example](../../examples/api/simt_atomics) passes
the vector identity explicitly and derives the independent contributor count
from its declared launch.

The example covers add, subtract, min, max, exchange, and/or/xor and CAS.
Its results are independent of thread order: exchange contributors agree
on the final value, and the CAS case has one eligible successful candidate.
The [wrapping atomic example](../../examples/api/simt_ring_atomics)
initializes its targets before increment/decrement and retains a thread
fence. A fence is not a rendezvous among all participants. These examples
use atomic calls as update statements and do not promise a returned prior
value or a particular racing thread order.

The [math example](../../examples/api/simt_math) includes elementary
functions, classification, FMA, rounding, bit counts and high-word multiply.
Its independent host formulas distinguish ties-to-even from ties-away and
check signed zero. Transcendental and `fmod` rows have explicit measured
budgets; integer and selected rounding rows compare carrier bits. Supported
dtype combinations are checked by frontend/IR/backend consumers rather than
the old Python tracing implementation. `cvt` is an explicit scalar cast
inside SIMT; it is not an A5 register `cast`.

Place surrounding transfers and the SIMT call in an appropriate `auto_sync`
region. Cross-side data ownership still needs an explicit mutex or matching
events. Pipe simulation checks the actual lowered accesses and atomic
classification, in addition to numerical output; it does not prove silicon
timing or vendor acceptance.

Each linked directory contains a complete local reference and runner:

```sh
python run.py reference
python run.py check --launcher sim
python run.py check --launcher pipesim
python run.py emit --backend cce
```

No simulator implementation or recorded expected output is a reference
dependency. The contract records the tested shapes, seeds and launch count.
