# RFC-0009: slot buffers for pipelined kernels (GMBuff)

Status: **M2 implemented, batches 1-3 complete** — `GMBuff` + the `gmbuff` pass are live and all
nine `lowering_xfail` kernels are rewritten and board-verified (§6-§9). §1 and §2 record the
situation this replaced, in the vocabulary of the edge planner that has since retired
(RFC-0005 §5): the `lowering_xfail` list is empty (§6) and the D-034 run-ahead bound no longer
exists on the A2 family. (Maintainer directive 2026-08-29: "把 workspace 做成
multi-buffer 的形式 … GMBuff(slot=xx)" — replace the pipeline-primitive plan of the earlier
design discussion with a data-structure declaration.)

## 1. Motivation

Nine a2 flash-attention kernels are `lowering_xfail`: their hand-written software-pipeline event
protocols fail the D-034 run-ahead bound (`%p_ub_slot_free (depth 1) can have 21 tokens in
flight`). Seven siblings with **structurally identical** protocols pass — the pass/fail line is
where D-034's replay happens to converge, not a structural property (`flash_attn_fullmask_bf16`
passes, `mha_flash_d256` fails; the two declare the same five events with the same pipes and
presets, and differ only in unroll counts). A parameter tweak can flip a passing kernel over the
line.

Independently, the passing kernels' GM workspace rings rest on a hand-maintained invariant. The
author's own comment in `flash_attn_full_pj_half_block32_causal_v5` states it:

> The shared GM workspaces (score_ws / p_ws / pv_ws) and the cross-side mutex depths form a
> ring; the producer overwrites slot (g % GROUP_STAGE_SLOTS) while the consumer reads slot
> ((g - GROUP_LOOKAHEAD) % GROUP_STAGE_SLOTS). To guarantee they never alias on real hardware
> the ring MUST be strictly larger than the lookahead. **The simulator does not model mutex
> depth, so this constraint is correctness-critical and only observable on the NPU.**

Both are machine-checked now: `gmbuff` checks the ring against the reader lag, and `crosssync`
refuses a credits finding where a mutex declares more credits than the buffer has slots
(RFC-0005 §3.1, M10-076).

## 2. Survey (53 pipelined / mixed C-V kernels)

### 2.1 The four layers

| layer | expression | kernels | compiles? |
|---|---|---|---|
| L1: plain hand-offs | `auto_sync` alone | a5 matmul / pipeline_patterns family (~15), `a2_matmul_subblock` | all pass |
| L2: constructed pipelining | mutex `depth` + `DBuff/TBuff` (+ `split_workspace`) | **a5 pfa family** (PRELOAD_N=2, zero hand-written events), a5 `mha_ifa` family, **a2 `attn_backward`** (13 buffers, zero events, 2-slot workspaces), `flash_attn_score_pv` | all pass |
| L3: hand-written UB-credit events | 2–5 `SEvent`s + 3 mutexes + `GROUP_LOOKAHEAD` phases | a2 flash_attn family, 16 kernels | **9 fail** (D-034), 7 pass by replay luck |
| misc | — | a5 `gdn_legacy` ×2 (xfail: frontend cell/dim typing, not pipelining), a5 `v9_allhif8` / `test_mla_entire` (few events, compile) | — |

### 2.2 The decisive finding: what the hand-written events guard

Every event of the L3 protocols guards **an on-chip resource's cross-iteration reuse**, none
guards GM directly (GM is the mutexes' job). `flash_attn_full` (representative, all 9 failers
isomorphic):

| event | pipes | guards | nature |
|---|---|---|---|
| `q_load_valid` | MTE2→MTE1 | `l1q` (already a Buff!) | one-shot hand-off across auto_sync blocks |
| `score_slot_free` | V→MTE2, preset | `ub_score` region across group iterations | depth-1 credit on a **plain Tensor** |
| `p_ub_slot_free` | MTE3→V, preset | `ub_p` across group iterations | depth-1 credit on a plain Tensor |
| `accum_store_ready` | V→MTE3 | `accum_ub` flush (set;wait adjacent) | serialisation point |
| `accum_store_valid` | MTE3→V, preset | `accum_ub` across M-tile iterations | depth-1 credit on a plain Tensor |

The hand-written protocol is a manual single-buffer credit system over plain Tensors — exactly
what `Buff` declarations + autosync's loop-carried machinery (D-031) automate. `attn_backward`
is the living proof: same C-V deep-pipeline shape, UB fully Buff-ised, zero hand-written events,
compiles, board-passes (bf16 1 ULP after D-066).

### 2.3 Single-beat indexing holds on causal

`v5`/`v6` (causal, passing) already index the workspace ring with one logical beat: producer
`g % SLOTS`, consumer `(g - GROUP_LOOKAHEAD) % SLOTS` — the distance is a literal difference.
The failing v1–v4 use two independent counters (`stage1_cnt` / `stage2_cnt`), whose phase
relation no analysis can see. Migration rule: one beat counter, reader indexes `beat - K`.

### 2.4 What GM needs (and does not need)

`autosync_gm` is off by default: GM workspace hazards are **not analysed at all** today — their
correctness rests entirely on the mutex-depth trust layer plus the hand-maintained
`slots > lookahead` invariant (§1). GMBuff's job is therefore **not** to unlock compilation (the
blockers are UB credits) but to make the trust layer checkable.

## 3. Design (implemented)

1. **`GMBuff(dtype, shape, slots=N, name='', per_core=True)`** — a GM member of the Buff
   family; `ws[beat]` (or `ws[beat, r0:r1, c0:c1]`) selects slot `beat % N`. Replaces the bare
   `split_workspace` + `var_mod` idiom; `per_core=True` (the default) gives every cube core its
   own ring (the old `GetCubeNum()` leading dim).
2. **Checks, the actual payload** (the `gmbuff` pass, after `device_lower`, before `autosync`):
   - **single beat** — every slot index of one ring is the same counter plus a static offset
     (walked through `scalar.add`/`sub`); two independent counters = error;
   - **reader lag < slots** — the offset span `K` must satisfy `K < slots` (the v5 comment's
     invariant, machine-checked);
   - **mutex cover** — on a ring touched from both sides, every write sits in a `lock..ready`
     window and every read in a `wait..free` window (the crosscore-flag spelling of the mutex
     protocol), and each covering mutex has `depth <= slots`. GM stays out of autosync
     (`autosync_gm` off); this window scan is the machine check the trust layer gets.
3. **Lowering = the proven spelling.** After the checks, each slot selection is rewritten to
   `scalar.mod(beat, slots)` + one masked root-level `mem.slice` of the plain
   `[cube_num, slots, rows, cols]` workspace — a user slice of the slot composes into that one
   slice (GM view chains stay one level deep; the simulator and backends support nothing
   deeper). Downstream passes see IR that is byte-for-byte the hand-written kernels' shape:
   the migrated `flash_attn_full` emits identical CCE modulo the hoisted `%`/`GetCubeIdx()`
   locals, and replays bit-exact against its golden.
4. **UB migration rule** (no new code, the unlock): L3 kernels' credit-guarded plain Tensors
   become Buff declarations inside `auto_sync` scope; the hand-written SEvents are deleted;
   autosync generates the events (worst case conservative depth or partial serialisation — a
   **performance** outcome, not a compile failure; D-034's bound is an output here, not a
   user-declared input it must verify).
5. **Explicitly not doing**: no pipeline control-flow primitive. The phase guards
   (`g < active_groups` / `g >= LOOKAHEAD`) stay as-is; phase information enters the analysis
   through the single-beat index difference.
6. **Reference interpreter**: a ring is stored as one 2D piece per (core, slot)
   (`ws:<name>:<i>`), so slot selections are plain 2D windows and the ideal no-alias ring
   semantics hold by construction (the interpreter never modelled mutex depth; the pass checks
   are what guard the hardware ring).

## 4. Migration plan

| batch | kernels | action | gate | status |
|---|---|---|---|---|
| 1 | the 9 lowering_xfail | UB Buff-isation + delete events + single-beat GMBuff | replay bit-exact → T1 → board (+ perf vs old framework) | **done** (§6) |
| 2 | the 7 luck-passing siblings | same rewrite (retire the replay-luck dependency) | same three gates | **done** (§8) |
| 3 | L2 layer (a5 pfa, attn_backward, …) | **no rewrite**; adopt GMBuff checks where a GM ring exists | compile + existing goldens | **done** (§9) |

First target: `flash_attn_full` (the smallest failer, board-comparable against the old
framework's build directly on the box).

Batch-1 notes gathered on the way:

- **The two-counter form collapses to the loop beat.** In every stage1_cnt/stage2_cnt kernel
  the counters equal `tile_base + ni` / `tile_base + ni − K` at their use sites, and the
  cross-side hand-backs through `p_ws`/`pv_ws` serialise the run-ahead at tile boundaries — so
  the tile-local beat (`ni`, or `group_id` in the grouped kernels) indexes the rings soundly,
  exactly as the already-single-beat v5/v6 do. The counters and their `var_mod` lines are
  deleted wholesale.
- **The one exception: v2's next-tile prefetch.** Its prefetch branch interleaves TWO tiles'
  production streams into one ring, so the slot algebra is not the linear one-beat form the
  gmbuff pass checks — v2 keeps `split_workspace` + the global counters (the mutex-depth trust
  layer, like the L3 layer) with a header comment. Its five events still die: the row-state
  double buffer became a Buff beat, and the rowmax/rowsum prefetch now travels UB → UB directly
  (the old GM round-trip's MTE3 → MTE2 same-side hand-off was guarded ONLY by a hand event —
  with `autosync_gm` off, deleting that event would have left the GM leg with no guard at all;
  the UB path is same-pipe program-ordered and autosync-analysed).
- **`l1q` fixed-slot preloads** (v3's `l1q[0]`, mha's `l1q[0]`/`l1q[1]`) drop their
  `q_load_valid`/`q_slot_free` events: the cross-tile WAR is autosync's loop-carried case,
  proven on `flash_attn_full`.

## 5. Open questions for M2 — answered on the first rewrite

- **D-036 bound quality: no serialisation.** The autosync-generated protocol for the rewritten
  `flash_attn_full` is structurally isomorphic to the deleted hand-written one — the same three
  preset depth-1 UB credits on the same pipe pairs (`V->MTE2` for `ub_score`, `MTE3->V` for
  `ub_p` and `accum_ub`), the cross-block `l1q` hand-off, every loop-carried event "1
  iteration(s) apart" — plus one hand-off the hand protocol implicitly serialised, which D-036
  widened to depth 2 (`ev_mte1_m_ready_4`, MTE1->M on `_l0b`: "pipe MTE1 can issue 2 sets
  before pipe M consumes the first").
- Remaining for batch 2: `v9_allhif8` / `test_mla_entire` event triage; `accum_store_ready`'s
  set-then-wait flush idiom (the rewrite expressed it as a plain autosync hand-off — watch
  whether the pattern recurs).

## 6. Batch-1 results (all nine lowering_xfail kernels rewritten)

`lowering_xfail` is now **empty**. Every kernel: hand events deleted, autosync generates the
protocol, lowering + T1 bisheng (89/89 corpus-wide) pass; balance was clean except the noted
armed tails, then pinned by D-066 and excused in `balance_xfail` — both of which retired with the
edge planner (RFC-0005 §5.5). Silicon:

| kernel | rings (slots / reader lag) | board evidence |
|---|---|---|
| `flash_attn_full` | GMBuff 4 / 0, 2, 0 | golden 2.380e-05 (same bytes as pre-GMBuff); probe (256,256,BH3) 9.7e-05; **31.1 us vs old 33.5 us** |
| `..._block32_causal` (v1) | GMBuff 2 / 0, 1, 0 | probe (1,4,256,256): out 1.25e-04, rowmax 2.6e-06, rowsum 5.7e-05 @ (5e-3, 1e-5) |
| `..._causal_v2` | split_workspace kept (prefetch interleaves two tiles' streams; §4 notes) | probe (1,4,**257,513**): out 1.18e-04, rowmax 1.9e-06 @ (5e-3, 1e-5) |
| `..._causal_v3` | GMBuff 4 / 0, 3, 0 | probe (1,4,256,256): out 1.25e-04 @ (5e-3, 1e-5) |
| `..._causal_v4` | GMBuff 4 / 0, 2, 0 | probe (1,4,256,256): out 1.25e-04 @ (5e-3, 1e-5) |
| `..._pj_hif8` | GMBuff 2 / 0, 1, 0 | probe (1,1,5376,258): out 4.46e-04 @ 1e-2 (hif8-quantised P), rowmax 2.4e-06 |
| `..._pj_hif8_commonub` | GMBuff 3 / 0, **2**, 0 | probe (1,1,5376,258): out 1.95e-03 = one bf16-output ULP; rowmax 1.2e-06; rowsum 3.4e-05 (fp32 device-exp accumulation over 258 terms — above the old main's fp16-era 1e-5 atol, at the honest floor) |
| `mha_flash_d256` | GMBuff 4 / 0, 2, 0 | replay bit-exact + golden board 5.07e-05 |
| `mha_flash_d256_bf16` | GMBuff 4 / 0, 2, 0 | replay bit-exact + golden board 19 bytes @ 1.95e-03 (bf16 1 ULP) |

One trap worth recording: the shared rewrite script assumed the hif8 pair shared a lag-1 ring,
but `commonub` drains `+2` and consumes `prev_nt = ni − 2` (that is WHY it declares 3 slots).
The wrong beat passed every static check — the gmbuff algebra validates consistency, not the
author's intended lag — and produced out ≈ 1.8 wrong on the board while rowmax/rowsum stayed
perfect (stage 1 untouched). The silicon probe is the gate that catches this class; batch 2
keeps per-kernel board runs mandatory.

## 7. M2 results (first rewrite: `flash_attn_full`)

| gate | result |
|---|---|
| lowering | 19 cube + 16 vec autosync events, all bounds provable; gmbuff ring notes: `score_ws` lag 0, `p_ws` lag 2 < 4 slots, `pv_ws` lag 0, all three cross-side rings mutex-covered |
| balance | one armed tail token on the vec side (attn_backward's family); D-066 pinned the flag id, then a recorded `balance_xfail` |
| replay | bit-exact against the golden, before and after the GMBuff migration |
| T1 | both units rc=0; zero flag-id reuse; CCE identical to the pre-GMBuff emission modulo hoisted `%`/`GetCubeIdx()` locals |
| board | golden case (S1=256, S2=512, BH=1): max abs diff 2.38e-05; probe (256, 256, BH=3): 9.7e-05 |
| perf | same shape (256, 256, BH=3), msprof Task Duration: **31.1 us rewritten vs 33.5 us old framework** (hand protocol) — the generated protocol is ~7% faster, no serialisation cost |

## 8. Batch-2 results (the seven luck-passing siblings)

All seven rewrites lower with the same shape as batch 1 (events deleted, autosync + gmbuff
checks in charge; the stream kernels key their rings on the global loop var `g` directly).
Two deliberate exceptions to "rings everywhere": `mla_b1_hq8_hkv4` keeps `accum_ws` as a plain
`split_workspace` whose adjacent `set;wait` MTE3→MTE2 flush (`accum_store_done`) orders the
hand-back — its old companion `accum_ws_valid` was redundant (the flush already orders every
later MTE2 load) and D-034 rejects its depth-1 shape; and the v2 reasoning of §6 carries over
unchanged.

The board pass exposed **two c220 printer bugs, not kernel bugs** (D-068): the V-V hazard
tracker was control-flow-blind (a then-branch barrier discharged the else scan's pendings) and
`dma.ub_to_ub` — a V-queue instruction — never joined the tracking. Only this family wraps its
vec reductions and running-max copies in dynamic `gi < group_len_v` branches, which is exactly
why these four kernels failed on silicon while every static-unroll kernel of batch 1 passed,
and why the pipe-level simulator (program-order semantics) saw nothing. Diagnosis: classify
each bad row's board value as an exact partial-prefix max (all 25 bad rows = tile-0-only;
the single row whose true max lives in the tail column = missing-tail), an S2=200 control
(86/86 bad = tile-0-only, all in-chunk positions), and an instrumented dump whose extra
autosync event made the corruption vanish (the copy raced the fusion on the V queue).
Softmax's shift invariance kept `out` near-correct through the second bug while
rowmax/rowsum stayed wrong — `out` alone is not a sufficient probe for these kernels.

| kernel | rings (slots / reader lag) | board evidence (after D-068 fixes) |
|---|---|---|
| `flash_attn_fullmask_bf16` | GMBuff 5 / 0, 3, 0 | probe (1,1,257,257): out 7.45e-04, rowmax 7.2e-07, rowsum 1.9e-05 @ (2e-2; bf16 P) |
| `..._causal_v6` | GMBuff 5 / 0, 3, 0 | probe (1,1,257,257): out 1.06e-04, rowmax 1.4e-06, rowsum 3.4e-05 @ (5e-3, 1e-5) |
| `..._causal_v5` | GMBuff 5 / 0, 3, 0 | probe (1,1,2177,2177): out 9.83e-05, rowmax 2.6e-06, rowsum 2.4e-04 @ (5e-3) |
| `flash_attn_fullmask_gqa_bf16` | GMBuff 5 / 0, 3, 0 | probe (1,8,1,257,257) MQA: out 9.00e-04, rowmax 1.2e-06, rowsum 2.3e-05 @ (2e-2); the 507015 aicore abort vanished with the same fixes |
| `flash_attn_full_pj_hif8_causal` | GMBuff 2 / 0, 1, 0 | probe (1,1,129,133): out 2.44e-04, rowmax 7.2e-07, rowsum 1.0e-05 @ (2e-2) |
| `mla_b1_hq8_hkv4` | score/p rings lag 1; pv ring beat = `l1v_cnt` cell; accum split_workspace + flush | probe S=128/SKV=256: out 7.88e-05; S=257/SKV=260: out 6.82e-05 @ (2e-2, out-only, `v = k_nope.clone()`) |
| `sage2_vnomean_int4` | — | out of scope: DT.int4 port gap (RFC-0008 open item), stays excluded |

## 9. Batch-3 results (GMBuff checks on the L2 rings; no rewrite)

Survey of every `split_workspace` in the corpus. Two kernels hold true beat rings and are
migrated to `GMBuff` declarations — protocol untouched, the counters collapse to the
tile-local beat exactly as in batch 1 (`ni` / `ni - 1`):

| kernel | rings migrated | evidence |
|---|---|---|
| `flash_attn_score_pv` | score_ws, p_ws (2 slots / lag 0, 1) | CCE equivalent modulo the slot mod's source and hoisted locals (events byte-identical); board vs old-main ref: (1,1,256,512) 1.47e-03, (1,3,256,512) 1.81e-03 @ 2e-2 |
| `attn_backward_…hif8_output_cast` | qk_ws, dp_ws, p_ws, dqk_ws (2 slots / lag 0, 0, 1, 1) | golden replay bit-exact; CCE equivalent (same normalisation, address arithmetic byte-identical); board golden A/B: gq/gk 9.77e-04 = bf16 1 ULP, gv bit-exact - the recorded pre-migration board figures exactly |

Everything else keeps `split_workspace` deliberately — there is no beat to check:

- `gq/gk/gv_acc_ws` (attn_backward) are core-shared flat accumulators fed by `atomic_add`;
- the a5 pfa family's `fd_accum/fd_max/fd_sum` are flash-decode **spatial** partitions
  (the slot index is a work-unit id — `tail_ws`/`head_ws`/`fd_lo`/`fd_hi` — written once and
  merged, never rotated by a beat);
- `conv_integrated` and `matmul_l0c_to_l1_demo` are flat stage hand-offs;
- `mla`'s accum_ws (§8) and v2's interleaved rings (§6) keep their argued exceptions.

With this, every beat-rotated GM ring in the corpus is a `GMBuff` under the gmbuff pass's
machine checks; the mutex-depth trust layer remains only where there is genuinely no ring.
