// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
/**
 * The PyPTO-Pro state machine had no tests at all.
 *
 * It is not the classic one with a test file next to it -- it is a rewrite: schema 2.0,
 * four stages instead of seven, a different action set, and a SPEC freeze that exists
 * only here. So nothing covered it, and two defects below survived because of that.
 *
 * Run with Node's built-in runner, which strips the types natively:
 *
 *     node --test "cannbot-skills/plugins-official/pypto-pro-op-orchestrator/hooks/opencode/__tests__/*.test.ts"
 *
 * The glob is required: .ts is not among the runner's default test-file patterns,
 * so a bare directory argument is treated as a module path and fails to resolve.
 *
 * No bun, no package.json, no install step. The plugin wiring file is deliberately not
 * imported here: it pulls in @opencode-ai/plugin, which is not present outside an
 * OpenCode install, and the transition logic it delegates to is what needs covering.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { applyTransition } from "../lib/state-transition-core.ts";

const OP = "custom/demo";

/** A freshly initialised state, built the way the hook builds one. */
function initial(maxStage = 4) {
  const status: Record<string, string> = {};
  const retry: Record<string, number> = {};
  for (let i = 1; i <= maxStage; i += 1) {
    status[String(i)] = "pending";
    retry[String(i)] = 0;
  }
  return {
    operator_name: "demo",
    schema_version: "2.0",
    max_stage: maxStage,
    current_stage: 1,
    stage_status: status,
    stage_retry_count: retry,
  } as any;
}

function init(state = initial()) {
  return applyTransition(state, { action: "init", stage: 1, max_stage: 4, opDir: OP } as any);
}

function advanceTo(stage: number) {
  let s = init();
  for (let i = 1; i < stage; i += 1) {
    s = applyTransition(s, { action: "complete_stage", stage: i, opDir: OP } as any);
  }
  return s;
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

test("init starts stage 1 in progress", () => {
  const s = init();
  assert.equal(s.current_stage, 1);
  assert.equal(s.stage_status["1"], "in_progress");
});

test("init must target stage 1", () => {
  assert.throws(
    () => applyTransition(initial(), { action: "init", stage: 2, opDir: OP } as any),
    /must target stage 1/,
  );
});

test("init refuses to run over work already in progress", () => {
  assert.throws(() => init(init()), /already in_progress/);
});

test("init clears rollback history and artifact hashes from a previous run", () => {
  const stale = initial();
  stale.rollback_history = [{ from_stage: 3, to_stage: 1, reason: "old" }];
  stale.artifact_hashes = { spec_md: "deadbeef" };
  const s = init(stale);
  assert.equal(s.rollback_history, undefined);
  assert.equal(s.artifact_hashes, undefined);
});

// ---------------------------------------------------------------------------
// complete_stage
// ---------------------------------------------------------------------------

test("completing the current stage advances to the next", () => {
  const s = applyTransition(init(), { action: "complete_stage", stage: 1, opDir: OP } as any);
  assert.equal(s.stage_status["1"], "completed");
  assert.equal(s.current_stage, 2);
  assert.equal(s.stage_status["2"], "in_progress");
});

test("only the current stage may be completed", () => {
  assert.throws(
    () => applyTransition(init(), { action: "complete_stage", stage: 2, opDir: OP } as any),
    /current_stage is 1/,
  );
});

test("a stage that is not in progress cannot be completed", () => {
  const s = applyTransition(init(), { action: "complete_stage", stage: 1, opDir: OP } as any);
  assert.throws(
    () => applyTransition(s, { action: "complete_stage", stage: 1, opDir: OP } as any),
    /current_stage is 2/,
  );
});

test("completing the last stage does not advance past it", () => {
  const s = applyTransition(advanceTo(4), { action: "complete_stage", stage: 4, opDir: OP } as any);
  assert.equal(s.stage_status["4"], "completed");
  assert.equal(s.current_stage, 4);
});

// ---------------------------------------------------------------------------
// fail_stage and retry accounting
// ---------------------------------------------------------------------------

test("failing a stage records the failure and counts the attempt", () => {
  const s = applyTransition(init(), {
    action: "fail_stage",
    stage: 1,
    reason: "compile error",
    opDir: OP,
  } as any);
  assert.equal(s.stage_status["1"], "failed");
  assert.equal(s.stage_retry_count["1"], 1);
});

test("a failed stage can be re-entered", () => {
  const failed = applyTransition(init(), {
    action: "fail_stage",
    stage: 1,
    reason: "x",
    opDir: OP,
  } as any);
  const s = applyTransition(failed, { action: "start_stage", stage: 1, opDir: OP } as any);
  assert.equal(s.stage_status["1"], "in_progress");
});

// ---------------------------------------------------------------------------
// rollback
// ---------------------------------------------------------------------------

test("rollback only goes backwards", () => {
  assert.throws(
    () =>
      applyTransition(advanceTo(2), {
        action: "rollback_to_stage",
        target_stage: 3,
        reason: "x",
        opDir: OP,
      } as any),
    /target/,
  );
});

test("rollback requires a reason", () => {
  assert.throws(
    () =>
      applyTransition(advanceTo(3), {
        action: "rollback_to_stage",
        target_stage: 1,
        opDir: OP,
      } as any),
    /reason/,
  );
});

test("rollback resets everything downstream and counts the retry", () => {
  const s = applyTransition(advanceTo(4), {
    action: "rollback_to_stage",
    target_stage: 2,
    reason: "design gap",
    opDir: OP,
  } as any);
  assert.equal(s.current_stage, 2);
  assert.equal(s.stage_status["2"], "in_progress");
  assert.equal(s.stage_status["3"], "pending");
  assert.equal(s.stage_status["4"], "pending");
  assert.equal(s.stage_retry_count["2"], 1);
  assert.equal(s.stage_status["1"], "completed", "work before the target is kept");
});

test("rollback history is appended, never replaced", () => {
  let s = applyTransition(advanceTo(3), {
    action: "rollback_to_stage",
    target_stage: 1,
    reason: "first",
    opDir: OP,
  } as any);
  s = applyTransition(s, { action: "complete_stage", stage: 1, opDir: OP } as any);
  s = applyTransition(s, { action: "complete_stage", stage: 2, opDir: OP } as any);
  s = applyTransition(s, {
    action: "rollback_to_stage",
    target_stage: 1,
    reason: "second",
    opDir: OP,
  } as any);
  assert.equal(s.rollback_history.length, 2);
  assert.deepEqual(
    s.rollback_history.map((h: any) => h.reason),
    ["first", "second"],
  );
});

test("rolling back drops the artifact hashes of the stages it undid", () => {
  let s = advanceTo(4);
  s.artifact_hashes = { spec_md: "a", golden_py: "b", design_md: "c" };
  s = applyTransition(s, {
    action: "rollback_to_stage",
    target_stage: 2,
    reason: "golden wrong",
    opDir: OP,
  } as any);
  assert.equal(s.artifact_hashes.spec_md, "a", "stage 1's hash predates the target");
  assert.equal(s.artifact_hashes.golden_py, "b", "the target stage's own hash is kept");
  assert.equal(s.artifact_hashes.design_md, undefined, "downstream hashes are dropped");
});

test("only spec_md is written automatically, which limits what the drop map can do", () => {
  // golden_py and design_md appear in the rollback drop map, but nothing writes them
  // unless a caller opts into record_artifact_hash. Documented so the gap is visible
  // rather than looking like coverage the machine does not have.
  let s = applyTransition(init(), { action: "complete_stage", stage: 1, opDir: OP } as any);
  s = applyTransition(s, { action: "complete_stage", stage: 2, opDir: OP } as any);
  assert.equal(s.artifact_hashes?.golden_py, undefined);
});

// ---------------------------------------------------------------------------
// record_artifact_hash
// ---------------------------------------------------------------------------

test("an artifact hash can be recorded", () => {
  const s = applyTransition(init(), {
    action: "record_artifact_hash",
    name: "design_md",
    hash: "abc123",
    opDir: OP,
  } as any);
  assert.equal(s.artifact_hashes.design_md, "abc123");
});

// ---------------------------------------------------------------------------
// Defects this file was written to catch
// ---------------------------------------------------------------------------

test("an under-populated stage map is repaired rather than silently obeyed", () => {
  // `if (!next.stage_status)` treats {} as present, so a partially written or
  // older-schema file kept its gaps -- and then `nextKey in statusMap` was false and
  // complete_stage stopped auto-advancing without saying anything.
  const partial = initial();
  partial.stage_status = { "1": "in_progress" };
  const s = applyTransition(partial, { action: "complete_stage", stage: 1, opDir: OP } as any);
  assert.equal(s.stage_status["1"], "completed");
  assert.equal(s.current_stage, 2, "auto-advance must not depend on the file being complete");
  assert.equal(s.stage_status["4"], "pending", "missing stages are filled in");
});

test("an empty stage map is repaired too", () => {
  const empty = initial();
  empty.stage_status = {};
  empty.stage_retry_count = {};
  const s = init(empty);
  assert.equal(s.stage_status["1"], "in_progress");
  assert.equal(Object.keys(s.stage_status).length, 4);
});

test("init honours the max_stage it is given", () => {
  // max_stage was read from the previous state and never from the input, so a caller
  // passing one to init was ignored whenever a state file already existed.
  const existing = initial(4);
  const s = applyTransition(existing, {
    action: "init",
    stage: 1,
    max_stage: 6,
    opDir: OP,
  } as any);
  assert.equal(s.max_stage, 6);
  assert.equal(s.stage_status["6"], "pending");
});

test("an unknown action is rejected", () => {
  assert.throws(
    () => applyTransition(init(), { action: "teleport", opDir: OP } as any),
    /teleport|unknown|unsupported/i,
  );
});

// --- regressions for three defects found by review, each verified to fail before its fix ---

test("init drops stage keys above the max_stage it is given", () => {
  // The classic 7-stage orchestrator and this 4-stage one write the same
  // custom/<op>/.orchestrator_state.json. The spread-merge filled gaps below max_stage but
  // never pruned above it, so '5'..'7' survived.
  const carried: any = {
    op_dir: OP, max_stage: 7, current_stage: 1,
    stage_status: { "1": "pending", "2": "pending", "3": "pending", "4": "pending",
                    "5": "pending", "6": "pending", "7": "pending" },
    stage_retry_count: {},
  };
  const s = applyTransition(carried, { action: "init", opDir: OP, stage: 1, max_stage: 4 } as any);
  assert.deepEqual(Object.keys(s.stage_status).sort(), ["1", "2", "3", "4"]);
});

test("complete_stage does not advance current_stage past the final stage", () => {
  // The symptom of the above: auto-advance tests `nextKey in statusMap`, which was true for
  // '5' on a 4-stage workflow. Every later action then throws at ensureStageNumber(5, 4),
  // so only rollback_to_stage escapes.
  const carried: any = {
    op_dir: OP, max_stage: 7, current_stage: 1,
    stage_status: { "1": "pending", "2": "pending", "3": "pending", "4": "pending",
                    "5": "pending", "6": "pending", "7": "pending" },
    stage_retry_count: {},
  };
  let s = applyTransition(carried, { action: "init", opDir: OP, stage: 1, max_stage: 4 } as any);
  for (const stage of [1, 2, 3, 4]) {
    s = applyTransition(s, { action: "complete_stage", opDir: OP, stage } as any);
  }
  assert.equal(s.current_stage, 4);
});

test("an L1 plan with zero modules is refused", () => {
  // The wrapper coerces a missing module_count to 0, which built an empty module map --
  // then complete_stage(4) certified the operator with nothing built, and fail_module's
  // `cycles >= max_cycles_per_module` stayed permanently false so `blocked` never fired.
  let s: any = init();
  for (const stage of [1, 2, 3]) {
    s = applyTransition(s, { action: "complete_stage", opDir: OP, stage } as any);
  }
  assert.throws(
    () => applyTransition(s, { action: "plan_stage4", opDir: OP, is_fusion: true, module_count: 0 } as any),
    /module_count >= 1/,
  );
});

test("init still refuses a live ledger whose in-progress stage is above max_stage", () => {
  // Regression: pruning stale keys ran BEFORE init's hasInProgress guard read them, so a
  // live 7-stage ledger in progress at stage 6 was accepted and overwritten -- stages 5-7
  // and their retry counts gone, no rollback_history entry.
  const live: any = {
    op_dir: OP, max_stage: 7, current_stage: 6,
    stage_status: { "1": "completed", "2": "completed", "3": "completed", "4": "completed",
                    "5": "completed", "6": "in_progress", "7": "pending" },
    stage_retry_count: { "6": 3 },
  };
  assert.throws(
    () => applyTransition(live, { action: "init", opDir: OP, stage: 1, max_stage: 4 } as any),
    /already in_progress/,
  );
});

test("complete_stage(4) refuses an L1 ledger with no modules", () => {
  // The producer guard (plan_stage4 module_count >= 1) does not cover a hand-maintained
  // ledger, which CLAUDE.md sanctions; `if (phases)` short-circuited the precondition.
  const hand: any = {
    op_dir: OP, max_stage: 4, current_stage: 4,
    stage_status: { "1": "completed", "2": "completed", "3": "completed", "4": "in_progress" },
    stage_retry_count: {}, stage4_path: "L1",
  };
  assert.throws(
    () => applyTransition(hand, { action: "complete_stage", opDir: OP, stage: 4 } as any),
    /stage4_modules is absent or empty/,
  );
});

// ---------------------------------------------------------------------------
// Stage 4 L1 happy paths.
//
// These did not exist. The suite covered only the two refusal branches, so
// `phases.status` -- a field Stage4Modules never declares -- read undefined,
// `?? {}` made the emptiness guard unconditionally true, and every L1/fusion
// delivery was permanently stuck at Stage 4 while all tests stayed green.
// ---------------------------------------------------------------------------

/** Drive a fresh ledger to Stage 4 with `count` planned modules on the L1 path. */
function atStage4(count: number) {
  let s: any = init();
  for (const stage of [1, 2, 3]) {
    s = applyTransition(s, { action: "complete_stage", opDir: OP, stage } as any);
  }
  return applyTransition(
    s,
    { action: "plan_stage4", opDir: OP, is_fusion: true, module_count: count } as any,
  );
}

/** Take one module through start -> submit_for_verify -> complete. */
function verifyModule(s: any, module: string) {
  s = applyTransition(s, { action: "start_module", opDir: OP, module } as any);
  s = applyTransition(s, { action: "submit_for_verify", opDir: OP, module } as any);
  return applyTransition(s, { action: "complete_module", opDir: OP, module } as any);
}

test("complete_stage(4) succeeds once every L1 module is verified (multi-module)", () => {
  let s = atStage4(2);
  for (const module of ["1", "2"]) {
    s = verifyModule(s, module);
  }
  assert.deepEqual(s.stage4_modules.module_status, { "1": "verified", "2": "verified" });

  s = applyTransition(s, { action: "complete_stage", opDir: OP, stage: 4 } as any);
  assert.equal(s.stage_status["4"], "completed");
});

test("complete_stage(4) succeeds for a single-module L1 delivery", () => {
  let s = verifyModule(atStage4(1), "1");
  s = applyTransition(s, { action: "complete_stage", opDir: OP, stage: 4 } as any);
  assert.equal(s.stage_status["4"], "completed");
});

test("complete_stage(4) still refuses when one planned module is unverified", () => {
  const s = verifyModule(atStage4(2), "1");
  assert.throws(
    () => applyTransition(s, { action: "complete_stage", opDir: OP, stage: 4 } as any),
    /module 2 status is/,
  );
});

test("complete_stage(4) succeeds after a failed module is debugged and re-verified", () => {
  let s = atStage4(1);
  s = applyTransition(s, { action: "start_module", opDir: OP, module: "1" } as any);
  s = applyTransition(s, { action: "fail_module", opDir: OP, module: "1" } as any);
  assert.equal(s.stage4_modules.module_status["1"], "in_debug");

  s = applyTransition(s, { action: "submit_for_verify", opDir: OP, module: "1" } as any);
  s = applyTransition(s, { action: "complete_module", opDir: OP, module: "1" } as any);
  s = applyTransition(s, { action: "complete_stage", opDir: OP, stage: 4 } as any);
  assert.equal(s.stage_status["4"], "completed");
});

test("complete_stage(4) refuses an L1 ledger whose module map is present but empty", () => {
  // module_count >= 1 is the reviewer-requested half of the guard: a hand-maintained
  // ledger can carry a stage4_modules object with a zeroed count and an empty map.
  const hand: any = {
    op_dir: OP, max_stage: 4, current_stage: 4,
    stage_status: { "1": "completed", "2": "completed", "3": "completed", "4": "in_progress" },
    stage_retry_count: {}, stage4_path: "L1",
    stage4_modules: { module_count: 0, module_status: {}, modules_verified: [], module_retry_count: {} },
  };
  assert.throws(
    () => applyTransition(hand, { action: "complete_stage", opDir: OP, stage: 4 } as any),
    /stage4_modules is absent or empty/,
  );
});

test("complete_stage(4) refuses an L1 ledger whose module_count is zero but map is not", () => {
  // The earlier test for the module_count clause supplied an EMPTY module_status, so the
  // pre-existing emptiness clause threw first and the count clause was never reached --
  // deleting `moduleCount >= 1` left all tests green. This is the state that isolates it:
  // a populated map with a zeroed count, where `for (i = 1; i <= 0; i++)` never runs.
  const hand: any = {
    op_dir: OP, max_stage: 4, current_stage: 4,
    stage_status: { "1": "completed", "2": "completed", "3": "completed", "4": "in_progress" },
    stage_retry_count: {}, stage4_path: "L1",
    stage4_modules: {
      module_count: 0, module_status: { "1": "verified" },
      modules_verified: ["1"], module_retry_count: {},
    },
  };
  assert.throws(
    () => applyTransition(hand, { action: "complete_stage", opDir: OP, stage: 4 } as any),
    /stage4_modules is absent or empty/,
  );
});

test("complete_stage(4) refuses an L1 ledger whose module_count is not a number", () => {
  // NaN-safety: `NaN < 1` is false so the old guard let this through, and `i <= NaN` is
  // false so the all-verified loop never ran -- certifying a fusion operator with an
  // unverified module. Both halves have to be NaN-safe for this to throw.
  const hand: any = {
    op_dir: OP, max_stage: 4, current_stage: 4,
    stage_status: { "1": "completed", "2": "completed", "3": "completed", "4": "in_progress" },
    stage_retry_count: {}, stage4_path: "L1",
    stage4_modules: {
      module_count: "many", module_status: { "1": "pending" },
      modules_verified: [], module_retry_count: {},
    },
  };
  assert.throws(
    () => applyTransition(hand, { action: "complete_stage", opDir: OP, stage: 4 } as any),
    /stage4_modules is absent or empty/,
  );
});
