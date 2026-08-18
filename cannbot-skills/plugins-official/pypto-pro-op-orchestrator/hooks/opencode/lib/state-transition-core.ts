// state-transition-core.ts — schema v2.2 (PyPTO-Pro)
//
// Pro workflow is 5 stages (1=planner, 2=mathematician, 3=architect,
// 4=coder, 5=optimizer). Stage 4 has two dispatch paths:
//   - L0 (is_fusion=false): single coder dispatch → stage4-check
//   - L1 (is_fusion=true):  per-Module loop with module-check gates
// The L1 path is driven by plan_stage4 + start_module / submit_for_verify /
// complete_module / fail_module actions.
//
// rollback_to_stage walks the workflow backwards: any stage strictly
// after target_stage is reset to pending, retry_count[target_stage] is
// incremented, and an entry is appended to rollback_history (append-only).
// When target_stage < 4, stage4_path and stage4_modules are also cleared
// (the Module contract will be re-designed).
//
// All write actions are restricted to pypto-pro-op-orchestrator at the
// plugin boundary; this file only enforces the state-machine invariants.

export type TransitionAction =
  | "init"
  | "start_stage"
  | "complete_stage"
  | "fail_stage"
  | "record_artifact_hash"
  | "rollback_to_stage"
  | "plan_stage4"
  | "start_module"
  | "submit_for_verify"
  | "complete_module"
  | "fail_module";

export type StageStatus = "pending" | "in_progress" | "completed" | "failed";

export type ModuleStatus =
  | "pending"
  | "in_progress"
  | "in_debug"
  | "awaiting_verify"
  | "verified"
  | "blocked";

export type RollbackEntry = {
  from_stage: number;
  to_stage: number;
  reason: string;
  failure_category?: string;
  timestamp: string;
};

export type ArtifactHashes = Record<string, string>;

export type Stage4Modules = {
  module_count: number;
  max_cycles_per_module: number;
  module_status: Record<string, ModuleStatus | string>;
  module_retry_count: Record<string, number>;
  modules_verified: string[];
  active_module?: string;
};

export type OrchestratorState = {
  operator_name?: string;
  schema_version?: string;
  max_stage: 5;
  current_stage: number;
  stage_status: Record<string, StageStatus | string>;
  stage_retry_count: Record<string, number>;
  artifact_hashes?: ArtifactHashes;
  rollback_history?: RollbackEntry[];
  stage4_path?: "L0" | "L1";
  stage4_modules?: Stage4Modules;
  last_updated?: string;
  [key: string]: unknown;
};

export type TransitionInput =
  | { action: "init"; stage?: number }
  | { action: "start_stage"; stage: number; reason?: string }
  | { action: "complete_stage"; stage: number }
  | { action: "fail_stage"; stage: number; reason?: string }
  | { action: "record_artifact_hash"; name: string; hash: string }
  | {
      action: "rollback_to_stage";
      target_stage: number;
      reason: string;
      failure_category?: string;
    }
  | { action: "plan_stage4"; module_count: number; is_fusion: boolean }
  | { action: "start_module"; module: string }
  | { action: "submit_for_verify"; module: string }
  | { action: "complete_module"; module: string }
  | {
      action: "fail_module";
      module: string;
      failure_category: string;
      failing_module_boundary?: string;
      last_error?: string;
    };

const WORKFLOW_STAGE_COUNT = 5;
const CURRENT_SCHEMA_VERSION = "2.2";
const DEFAULT_MAX_CYCLES_PER_PHASE = 10;

function cloneState(prev: OrchestratorState): OrchestratorState {
  return JSON.parse(JSON.stringify(prev));
}

function ensureStageNumber(val: unknown, max: number): number {
  const n = Number(val);
  if (!Number.isFinite(n) || n < 1 || n > max) {
    throw new Error(`invalid stage: ${val} (must be 1..${max})`);
  }
  return Math.floor(n);
}

function emptyStageStatus(maxStage: number): Record<string, StageStatus> {
  const out: Record<string, StageStatus> = {};
  for (let i = 1; i <= maxStage; i++) out[String(i)] = "pending";
  return out;
}

function emptyRetryCount(maxStage: number): Record<string, number> {
  const out: Record<string, number> = {};
  for (let i = 1; i <= maxStage; i++) out[String(i)] = 0;
  return out;
}

/**
 * Compute the cumulative suffix for a given module number.
 * Module 1 → "1", Module 2 → "12", Module 3 → "123", etc.
 */
function moduleSuffix(phaseNum: number): string {
  let suffix = "";
  for (let i = 1; i <= phaseNum; i++) suffix += String(i);
  return suffix;
}

/**
 * Ensure stage4_modules exists and is well-formed. Throws if stage4_path
 * is not "L1" (Module actions are L1-only).
 */
function ensureStage4Modules(state: OrchestratorState): Stage4Modules {
  if (state.stage4_path !== "L1") {
    throw new Error(
      `Module action requires stage4_path=="L1", but stage4_path is "${state.stage4_path ?? "unset"}"`,
    );
  }
  if (!state.stage4_modules) {
    throw new Error(
      `Module action requires stage4_modules to be initialized (call plan_stage4 first)`,
    );
  }
  return state.stage4_modules;
}

export function applyTransition(
  prev: OrchestratorState,
  input: TransitionInput,
): OrchestratorState {
  const next = cloneState(prev);
  const maxStage = WORKFLOW_STAGE_COUNT;
  if (next.max_stage !== maxStage) {
    throw new Error(
      `invalid workflow state: max_stage must be ${maxStage}, got ${next.max_stage}`,
    );
  }

  // Fill missing entries and discard malformed/out-of-range keys so transitions always
  // operate on the fixed five-stage schema.
  const prune = <T,>(base: Record<string, T>, carried: Record<string, T> | undefined) => {
    const merged = { ...base, ...(carried ?? {}) };
    for (const key of Object.keys(merged)) {
      const n = Number(key);
      if (!Number.isInteger(n) || n < 1 || n > maxStage) delete merged[key];
    }
    return merged;
  };
  // Snapshot before normalization so init still detects any active stage in the input.
  const carriedStatus = { ...(next.stage_status ?? {}) };
  next.stage_status = prune(emptyStageStatus(maxStage), next.stage_status);
  next.stage_retry_count = prune(emptyRetryCount(maxStage), next.stage_retry_count);

  const statusMap = next.stage_status;
  const retryMap = next.stage_retry_count;

  switch (input.action) {
    case "init": {
      const stage = ensureStageNumber(input.stage ?? 1, maxStage);
      if (stage !== 1) {
        throw new Error(`init action must target stage 1, got stage ${stage}`);
      }
      // Judge the ledger as it was found, not as pruning left it.
      const hasInProgress = Object.values(carriedStatus).some((s) => s === "in_progress");
      if (hasInProgress) {
        throw new Error(`cannot init: a stage is already in_progress`);
      }
      next.current_stage = stage;
      statusMap[String(stage)] = "in_progress";
      next.schema_version = CURRENT_SCHEMA_VERSION;
      // Clear residual state from any previous run so a re-init cannot
      // leak stale rollback history, artifact hashes, or Stage 4 path state.
      next.rollback_history = undefined;
      next.artifact_hashes = undefined;
      next.stage4_path = undefined;
      next.stage4_modules = undefined;
      break;
    }

    case "start_stage": {
      const stage = ensureStageNumber(input.stage, maxStage);
      const otherInProgress = Object.entries(statusMap).some(
        ([k, s]) => s === "in_progress" && k !== String(stage),
      );
      if (otherInProgress) {
        throw new Error(`cannot start stage ${stage}: another stage is already in_progress`);
      }
      const key = String(stage);
      if (statusMap[key] === "completed") {
        throw new Error(`cannot start stage ${stage}: already completed`);
      }
      if (stage > 1) {
        const prevKey = String(stage - 1);
        const prevStatus = statusMap[prevKey];
        if (prevStatus !== "completed") {
          throw new Error(
            `cannot start stage ${stage}: previous stage ${stage - 1} is "${prevStatus ?? "unknown"}", not "completed"`,
          );
        }
      }
      next.current_stage = stage;
      statusMap[key] = "in_progress";
      break;
    }

    case "complete_stage": {
      const stage = ensureStageNumber(input.stage, maxStage);
      if (stage !== prev.current_stage) {
        throw new Error(
          `cannot complete stage ${stage}: current_stage is ${prev.current_stage}`,
        );
      }
      const compKey = String(stage);
      if (statusMap[compKey] !== "in_progress") {
        throw new Error(
          `cannot complete stage ${stage}: status is "${statusMap[compKey]}", not "in_progress"`,
        );
      }
      // Stage 4 L1 precondition: at least one module must exist, and all must be verified.
      if (stage === 4 && next.stage4_path === "L1") {
        const phases = next.stage4_modules;
        // The producer-side guard (plan_stage4 requires module_count >= 1) does not cover
        // a hand-maintained ledger, which CLAUDE.md explicitly sanctions. Without this,
        // `if (phases)` short-circuited the all-modules-verified precondition and an L1
        // operator was certified with zero modules built -- the outcome that guard exists
        // to refuse. Read `module_status`: that is the field Stage4Modules declares.
        // `!(n >= 1)`, not `n < 1`: a non-numeric module_count coerces to NaN, and
        // `NaN < 1` is false, so the refusal was skipped -- while `i <= NaN` is also
        // false, so the all-verified loop below never ran either. A hand-maintained
        // ledger with module_count: "many" was certified with nothing verified. This
        // matches how plan_stage4 already spells the same precondition.
        const moduleCount = Number(phases?.module_count);
        if (
          !phases ||
          !(moduleCount >= 1) ||
          Object.keys(phases.module_status ?? {}).length === 0
        ) {
          throw new Error(
            `cannot complete Stage 4 on the L1 path: stage4_modules is absent or empty, so ` +
            `no module was ever verified. Call plan_stage4 with module_count >= 1 first.`,
          );
        }
        for (let i = 1; i <= moduleCount; i++) {
          const ps = phases.module_status[String(i)];
          if (ps !== "verified") {
            throw new Error(
              `cannot complete_stage(4): module ${i} status is "${ps}", not "verified"`,
            );
          }
        }
      }
      statusMap[compKey] = "completed";
      // Auto-advance to next stage if it exists.
      const nextStage = stage + 1;
      const nextKey = String(nextStage);
      if (nextKey in statusMap) {
        next.current_stage = nextStage;
        statusMap[nextKey] = "in_progress";
      }
      break;
    }

    case "fail_stage": {
      const stage = ensureStageNumber(input.stage, maxStage);
      const failKey = String(stage);
      if (stage !== prev.current_stage) {
        throw new Error(
          `cannot fail_stage ${stage}: current_stage is ${prev.current_stage}`,
        );
      }
      if (statusMap[failKey] !== "in_progress") {
        throw new Error(
          `cannot fail_stage ${stage}: status is "${statusMap[failKey]}", not "in_progress"`,
        );
      }
      retryMap[failKey] = (Number(retryMap[failKey]) || 0) + 1;
      statusMap[failKey] = "failed";
      break;
    }

    case "record_artifact_hash": {
      if (!input.name) throw new Error(`record_artifact_hash requires name`);
      if (!input.hash) throw new Error(`record_artifact_hash requires hash`);
      if (!next.artifact_hashes) next.artifact_hashes = {};
      next.artifact_hashes[input.name] = input.hash;
      break;
    }

    case "rollback_to_stage": {
      const target = ensureStageNumber(input.target_stage, maxStage);
      if (target >= prev.current_stage) {
        throw new Error(
          `rollback target stage ${target} must be strictly less than current_stage ${prev.current_stage}`,
        );
      }
      if (!input.reason || !input.reason.trim()) {
        throw new Error(`rollback_to_stage requires a non-empty reason`);
      }
      // Reset every stage strictly after target to pending.
      for (const key of Object.keys(statusMap)) {
        const k = Number(key);
        if (k > target) statusMap[key] = "pending";
      }
      // Increment retry count on target stage and re-enter it.
      retryMap[String(target)] = (Number(retryMap[String(target)]) || 0) + 1;
      statusMap[String(target)] = "in_progress";
      next.current_stage = target;
      // Drop artifact hashes that were computed at stages strictly after target.
      const HASH_STAGE: Record<string, number> = {
        spec_md: 1,
        golden_py: 2,
        design_md: 3,
      };
      if (next.artifact_hashes) {
        for (const [name, owningStage] of Object.entries(HASH_STAGE)) {
          if (owningStage > target && name in next.artifact_hashes) {
            delete next.artifact_hashes[name];
          }
        }
      }
      // Clear Stage 4 path state when target < 4 (Module contract will be re-designed).
      if (target < 4) {
        next.stage4_path = undefined;
        next.stage4_modules = undefined;
      }
      // Append rollback history (append-only).
      if (!next.rollback_history) next.rollback_history = [];
      next.rollback_history.push({
        from_stage: prev.current_stage,
        to_stage: target,
        reason: input.reason,
        failure_category: input.failure_category,
        timestamp: new Date().toISOString(),
      });
      break;
    }

    // ── Stage 4 Module actions (L1 only) ──

    case "plan_stage4": {
      if (prev.current_stage !== 4) {
        throw new Error(
          `plan_stage4 requires current_stage==4, got ${prev.current_stage}`,
        );
      }
      next.stage4_path = input.is_fusion ? "L1" : "L0";
      if (input.is_fusion && !(Number(input.module_count) >= 1)) {
        // An L1 plan with no modules is not a plan. The wrapper coerces a missing
        // module_count to 0, which built an empty module map -- and then complete_stage(4)
        // certified the operator with nothing ever built, while fail_module's
        // `cycles >= max_cycles_per_module` stayed permanently false so `blocked` never
        // fired either. Refuse it here rather than letting both guards read as satisfied.
        throw new Error(
          `plan_stage4 with is_fusion=true requires module_count >= 1, got ` +
          `${input.module_count === undefined ? "undefined" : input.module_count}`,
        );
      }
      if (input.is_fusion) {
        const phaseStatus: Record<string, ModuleStatus> = {};
        const phaseRetry: Record<string, number> = {};
        for (let i = 1; i <= input.module_count; i++) {
          phaseStatus[String(i)] = "pending";
          phaseRetry[String(i)] = 0;
        }
        next.stage4_modules = {
          module_count: input.module_count,
          max_cycles_per_module: DEFAULT_MAX_CYCLES_PER_PHASE,
          module_status: phaseStatus,
          module_retry_count: phaseRetry,
          modules_verified: [],
          active_module: undefined,
        };
      } else {
        next.stage4_modules = undefined;
      }
      break;
    }

    case "start_module": {
      const phases = ensureStage4Modules(next);
      const phaseKey = input.module;
      const phaseNum = Number(phaseKey);
      if (!Number.isFinite(phaseNum) || phaseNum < 1 || phaseNum > phases.module_count) {
        throw new Error(
          `start_module: invalid module "${phaseKey}" (must be 1..${phases.module_count})`,
        );
      }
      const cur = phases.module_status[phaseKey];
      if (cur !== "pending" && cur !== "blocked") {
        throw new Error(
          `start_module(${phaseKey}): status is "${cur}", must be "pending" or "blocked"`,
        );
      }
      phases.module_status[phaseKey] = "in_progress";
      phases.active_module = moduleSuffix(phaseNum);
      break;
    }

    case "submit_for_verify": {
      const phases = ensureStage4Modules(next);
      const phaseKey = input.module;
      const cur = phases.module_status[phaseKey];
      if (cur !== "in_progress" && cur !== "in_debug") {
        throw new Error(
          `submit_for_verify(${phaseKey}): status is "${cur}", must be "in_progress" or "in_debug"`,
        );
      }
      phases.module_status[phaseKey] = "awaiting_verify";
      break;
    }

    case "complete_module": {
      const phases = ensureStage4Modules(next);
      const phaseKey = input.module;
      const phaseNum = Number(phaseKey);
      const cur = phases.module_status[phaseKey];
      if (
        cur !== "in_progress" &&
        cur !== "in_debug" &&
        cur !== "awaiting_verify"
      ) {
        throw new Error(
          `complete_module(${phaseKey}): status is "${cur}", must be an active state`,
        );
      }
      phases.module_status[phaseKey] = "verified";
      const suffix = moduleSuffix(phaseNum);
      if (!phases.modules_verified.includes(suffix)) {
        phases.modules_verified.push(suffix);
      }
      phases.active_module = undefined;
      break;
    }

    case "fail_module": {
      const phases = ensureStage4Modules(next);
      const phaseKey = input.module;
      const cur = phases.module_status[phaseKey];
      if (cur === "verified") {
        throw new Error(
          `fail_module(${phaseKey}): module is already "verified", cannot fail`,
        );
      }
      const retryKey = phaseKey;
      phases.module_retry_count[retryKey] =
        (Number(phases.module_retry_count[retryKey]) || 0) + 1;
      const cycles = phases.module_retry_count[retryKey];
      if (cycles >= phases.max_cycles_per_module) {
        phases.module_status[phaseKey] = "blocked";
      } else {
        phases.module_status[phaseKey] = "in_debug";
      }
      break;
    }

    default: {
      const a = (input as { action?: string }).action ?? "unknown";
      throw new Error(`unsupported action: ${a}`);
    }
  }

  next.last_updated = new Date().toISOString();
  return next;
}
