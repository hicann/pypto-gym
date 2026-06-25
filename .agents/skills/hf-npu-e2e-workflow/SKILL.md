---
name: hf-npu-e2e-workflow
description: Orchestration workflow — take an HF model from a card id to a measured E2E throughput on Ascend NPU. Pipelines download (download_hf_model.py) → runtime patch / auto_map (runtime_patch.py) → run → measure eager / NPUGraph(graph) tok/s, and invokes pypto-fused-op-integration only when a fused kernel is actually needed. 把 HF 模型从 model-id 一路跑到昇腾 NPU 上的 E2E 吞吐：下载 → runtime patch → 运行 → 测速（eager / 图捕获），需要融合算子时才调用 pypto-fused-op-integration。Triggers on "/hf-npu-e2e-workflow", "run HF model on NPU end to end", "NPU E2E workflow", "download and benchmark on NPU", "prefill/decode tok/s", "NPUGraph 捕获测速", "NPU에서 E2E로 돌리고 재".
---

# hf-npu-e2e-workflow

An **orchestration workflow** (not a how-to skill). It wires the existing pypto-gym assets so
an HF model goes from a card id to a measured E2E number on Ascend NPU. Every step is
independently skippable — **not every run needs fusion**, so fusion is the last, optional step.

## When to use which steps / 何时用哪步

| 목표 | 단계 |
|---|---|
| 그냥 NPU에서 돌리고 재고 싶다 | 1 → 2 → 3 → 4 (download → patch → run → measure). **fusion 없음** |
| 융합 커널 이득까지 보고 싶다 | + 5 (`pypto-fused-op-integration` 호출) → 4 재실행 |

## Pipeline

### 1. Download — `download_hf_model.py`
```bash
python modeling/transformers/download_hf_model.py --model-id <org/repo> --output-dir <DIR>
# [--revision R] [--token T] [--allow-pattern P] [--ignore-pattern P]
```
Resumable `snapshot_download`; set `HF_ENDPOINT=https://hf-mirror.com` if the mirror is needed.

### 2. Runtime patch (auto_map + in-repo modeling) — `runtime_patch.py`
```bash
# preset families: gemma4_31b_it | llada2_moe | minimax_m27 | minimax_m3
python modeling/transformers/runtime_patch.py --model-family <FAMILY> --model-path <DIR>
# new model: --auto-map AutoConfig=core.configuration_X.XConfig --copy core/modeling_X.py=<src>
```
Installs `auto_map` and copies the in-repo modeling so the model loads **without** depending on a
cached/remote `modeling_*.py`.

### 3. Run on NPU (smoke)
Load with `local_files_only=True` (+ `trust_remote_code` only if the model needs it), generate a few
tokens, confirm it emits coherent text on the NPU. This is the migration gate — if eager doesn't run,
fix that before measuring.

### 4. E2E measure — `scripts/bench_npu.py`  (this workflow owns this layer)
```bash
python .agents/skills/hf-npu-e2e-workflow/scripts/bench_npu.py \
    --model-path <DIR> --device 0 --regime decode --gen 128 --arms eager,graph
```
- **regime** = `prefill` (seq/forward) · `decode` (N/generate) — pick ONE and label it; they are not
  comparable. See [references/npu-run-and-measure.md](references/npu-run-and-measure.md) §1.
- **arms** = `eager` / `graph` (static-shape NPUGraph). The harness auto-neutralizes the
  capture-hostile ops (SDPA fused kernel → eager attention; mask prep & rotary host syncs → prebuilt
  mask + cached cos/sin; MoE argsort → static route) and, on a capture failure, names the cause by ACL
  code. Full gotcha table + capture-safe recipe + the primitives (`npu_capture.py`) live in the
  references file.

### 5. (Optional) Fusion — invoke `pypto-fused-op-integration`
Only when a model needs a **new fused kernel** (grouped GEMM / attention). Hand the whole
golden → op-dev → precision → **integration (USE_PTO_<OP> switch + sys.modules injection)** flow to the
`pypto-fused-op-integration` skill — this workflow does **not** re-implement any of it. Once that skill
has wired and switched the kernel on, re-run **step 4** (`--arms eager,graph`): the `graph` arm now
captures the PyPTO-accelerated forward, giving the `pypto+graph` number.

> **Read `pypto-fused-op-integration`'s SKILL.md for the fusion methodology, but apply these
> overrides on THIS (workflow) side — do not edit that skill:**
> - Its onboarding is stale: skip its *download* step (raw `snapshot_download`) and its *code-deploy*
>   step (manual `core/` copy + `fix_imports.py` + hand-written `auto_map`). Use **this workflow's
>   steps 1–2** instead — `download_hf_model.py` + `runtime_patch.py` supersede them.
> - Its `pip install torch==2.7.1 torch-npu==2.7.1` pin is **environment-specific**. The durable rule
>   is "torch == torch_npu, matched to your box's CANN": look up the `torch_npu` release that matches
>   your installed CANN, then pin `torch` to that exact version — don't hard-follow a fixed number.

## Bundled (the net-new layer this workflow owns)
- [scripts/bench_npu.py](scripts/bench_npu.py) — generalized eager/graph E2E bench (capture
  auto-mitigations + ACL-code diagnostic).
- [scripts/npu_capture.py](scripts/npu_capture.py) — capture-safe primitives
  (`apply_capture_mitigations`, `CaptureCache`, `capture_and_replay`, `explain_capture_error`).
- [references/npu-run-and-measure.md](references/npu-run-and-measure.md) — regimes, 3-arm, the
  NPUGraph capture **gotcha table** (error → cause → fix), capture-safe recipe, measurement procedure.

## Scope — what this workflow does NOT own (no duplication)
- **HF onboarding internals** → `download_hf_model.py` + `runtime_patch.py` (steps 1-2). It does not
  re-document download/auto_map/import-fixing. (`migrate-huggingface-to-npu` in cann/pypto covers the
  same ground but is stale — **not used**; these two scripts supersede it.)
- **Op-fusion methodology + the PyPTO `USE_PTO_<OP>` switch / sys.modules injection** →
  `pypto-fused-op-integration`. This workflow only *invokes* it and then *measures* the result.
