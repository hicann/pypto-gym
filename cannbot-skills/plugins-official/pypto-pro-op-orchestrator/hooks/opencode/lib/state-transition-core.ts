// state-transition-core.ts — schema v2.0 (PyPTO-Pro)
//
// Pro workflow is 4 stages (1=planner, 2=mathematician, 3=architect,
// 4=coder). There is no inner Phase loop and no Stage-5 module
// decomposition — Pro produces a single test_{op}.py file containing
// both kernel and test. Therefore this core omits the Phase and
// submit_design machinery found in the classic (7-stage) core.
//
// rollback_to_stage walks the workflow backwards: any stage strictly
// after target_stage is reset to pending, retry_count[target_stage] is
// incremented, and an entry is appended to rollback_history (append-only).
//
// All write actions are restricted to pypto-pro-op-orchestrator at the
// plugin boundary; this file only enforces the state-machine invariants.

export type TransitionAction =
  | "init"
  | "start_stage"
  | "complete_stage"
  | "fail_stage"
  | "record_artifact_hash"
  | "rollback_to_stage";

export type StageStatus = "pending" | "in_progress" | "completed" | "failed";

export type RollbackEntry = {
  from_stage: number;
  to_stage: number;
  reason: string;
  failure_category?: string;
  timestamp: string;
};

export type ArtifactHashes = Record<string, string>;

export type OrchestratorState = {
  operator_name?: string;
  schema_version?: string;
  max_stage: number;
  current_stage: number;
  stage_status: Record<string, StageStatus | string>;
  stage_retry_count: Record<string, number>;
  artifact_hashes?: ArtifactHashes;
  rollback_history?: RollbackEntry[];
  last_updated?: string;
  [key: string]: unknown;
};

export type TransitionInput =
  | { action: "init"; stage?: number; max_stage?: number }
  | { action: "start_stage"; stage: number; reason?: string }
  | { action: "complete_stage"; stage: number }
  | { action: "fail_stage"; stage: number; reason?: string }
  | { action: "record_artifact_hash"; name: string; hash: string }
  | {
      action: "rollback_to_stage";
      target_stage: number;
      reason: string;
      failure_category?: string;
    };

const DEFAULT_MAX_STAGE = 4;

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

export function applyTransition(
  prev: OrchestratorState,
  input: TransitionInput,
): OrchestratorState {
  const next = cloneState(prev);
  const maxStage = next.max_stage ?? DEFAULT_MAX_STAGE;
  next.max_stage = maxStage;

  // Ensure required maps exist before any branch reads them.
  if (!next.stage_status) next.stage_status = emptyStageStatus(maxStage);
  if (!next.stage_retry_count) next.stage_retry_count = emptyRetryCount(maxStage);

  const statusMap = next.stage_status;
  const retryMap = next.stage_retry_count;

  switch (input.action) {
    case "init": {
      const stage = ensureStageNumber(input.stage ?? 1, maxStage);
      if (stage !== 1) {
        throw new Error(`init action must target stage 1, got stage ${stage}`);
      }
      const hasInProgress = Object.values(statusMap).some((s) => s === "in_progress");
      if (hasInProgress) {
        throw new Error(`cannot init: a stage is already in_progress`);
      }
      next.current_stage = stage;
      statusMap[String(stage)] = "in_progress";
      next.schema_version = next.schema_version ?? "2.0";
      // Clear residual state from any previous run so a re-init cannot
      // leak stale rollback history or artifact hashes.
      next.rollback_history = undefined;
      next.artifact_hashes = undefined;
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

    default: {
      const a = (input as { action?: string }).action ?? "unknown";
      throw new Error(`unsupported action: ${a}`);
    }
  }

  next.last_updated = new Date().toISOString();
  return next;
}
