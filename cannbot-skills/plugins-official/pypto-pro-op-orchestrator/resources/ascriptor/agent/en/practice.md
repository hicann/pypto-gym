# Practice with explicit contracts

These are authoring exercises, not claims that every formula below is already a released unit.
Work in an ignored `tmp/practice/<task>/` directory with the accepted installed library. Start
from [author](playbooks/author.md), generate inputs and write an independent Torch reference.
Look up the current API coverage and unit contract before borrowing a pattern. No prototype,
recorded tensors or fixed process-global facade is needed.

| Exercise | Formula or precision boundary | Main question |
|---|---|---|
| Scale and bias | `2*x+1` | GM/UB/register ownership, tails and core partitioning |
| Sigmoid | `1/(1+exp(-x))` | Expression order, overflow domain and approximation error |
| Row sum | `x.sum(-1, keepdim=True)` | Reduction result lanes and broadcast layout |
| One-tile softmax | Stable row max, subtraction, exp, sum and divide | Defined lanes and masked tails |
| Basic matmul | `x @ y.T` | L1/L0 storage and cube handshakes |
| KM/KN matmul | `x.T @ y`, with `x[K,M]`, `y[K,N]` | Transpose at the legal consumer and explicit dimensions |
| Large-K matmul | The same product, split over K | Initialization tile and subsequent accumulation |
| Bias/ReLU epilogue | `relu(x.float() @ y.float().T + bias)` | Cube-to-vector handoff and bias broadcast |
| Matmul row softmax | `softmax(x.float() @ y.float().T, -1)` | Row statistics and output lifetime |
| Matmul L2 normalization | Divide each product row by its norm | Zero-norm policy and two-pass statistics |
| Blockwise quantization | Block128 absmax, stated scale rule and FP8 output | Scale ownership, independent bit semantics and carrier layout |
| ReLU then matmul | `relu(x).half().float() @ y.float().T` | Preserve the half boundary before vector-to-cube publication |
| Two mixed stages | `abs((x*2).half().float() @ y.float().T)+1` | Both mutex directions and independent checkpoints |
| Decode attention | `softmax(q @ k.T / sqrt(D)) @ v`, L=1,D=128 | Online state, lookahead, warmup/drain and reused slots |
| Causal attention | The declared attention formula with `k_pos <= q_pos` | Query-position convention, causal boundary and masked PV footprint |

Each exercise chooses a precise dtype/layout/domain and launch topology. A2/A3 workspace bridges
and A5 on-chip bridges are different contracts. A probability cast and its normalization
denominator must preserve the reference's order; a tag in the gallery index is not authority
to move them.
Host code allocates, packs under an explicit ABI contract, dispatches and compares; it does not
silently replace a requested kernel's mathematical stages.

Cover aligned/tail and supported multi-tile/core cases, initialized outputs where applicable,
and a same-core slot reuse case. Use exact or tolerant comparison according to [precision](references/precision.md),
and reject deliberate missing/zero/corrupt outputs. Record functional, lowered pipe, emission,
vendor and board evidence separately. Preserve unresolved warnings as diagnosed limitations.
Before promoting a solution, copy the folder alone into an unrelated directory and run it
there with the declared installed dependencies; anything it still imports from the tree it
came from means it is not self-contained. Promotion goes to the owner: minimal API teaching
in library, a complete runnable algorithm in the kernels gallery as `kernel.py` +
`reference.py` + `main.py` + `metadata.json` and nothing else.

## Mixed-pipeline and guidance exercises

Use the [runnable mixed-pipeline demo](../../kernels/ascriptor_kernels/tutorials/mixed_pipeline)
for CVC, VCV, CVCV, VCVC and CVCVC. Inside that folder `python main.py --list` prints its
cases and `python main.py --pattern CVC --mode pipeline` runs one graph on one schedule.
Read [the generic method](references/pipeline-model.md) before copying a delay/depth, and
compare both streamed and resident workloads with their own matched serial controls — the
demo carries `serial`, `pipeline`, `resident_serial` and `resident` of every graph for
exactly that. A legal candidate can have no compute overlap or lose time at a small shape;
retain that result. Nothing in the folder is measured, so a speed claim has to be your own
measurement.

To assess the documentation itself, use the independent-context protocol
and task contracts. They separate numerical authoring,
constrained scheduling and open performance tasks. A case sweep of one kernel
is not multiple independent generation trials. Current scope and results are
in the validation receipt.
