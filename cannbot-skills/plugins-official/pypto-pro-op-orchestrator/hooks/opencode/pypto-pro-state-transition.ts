import { type Plugin, tool } from "@opencode-ai/plugin";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  applyTransition,
  type OrchestratorState,
  type TransitionAction,
  type TransitionInput,
} from "./lib/state-transition-core";

const ALLOWED_ACTIONS = new Set<TransitionAction>([
  "init",
  "start_stage",
  "complete_stage",
  "fail_stage",
  "record_artifact_hash",
  "rollback_to_stage",
]);

/** The AGENTS.md primary runs as "build"; the legacy named primary remains compatible. */
const ALLOWED_AGENTS = new Set<string>([
  "build",
  "pypto-pro-op-orchestrator",
]);

const DEFAULT_MAX_STAGE = 4;

function parseState(content: string): OrchestratorState {
  const parsed = JSON.parse(content);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("invalid orchestrator state file content");
  }
  return parsed as OrchestratorState;
}

function buildInitialState(opDir: string, maxStage: number): OrchestratorState {
  const stageStatus: Record<string, string> = {};
  const stageRetry: Record<string, number> = {};
  for (let i = 1; i <= maxStage; i++) {
    stageStatus[String(i)] = "pending";
    stageRetry[String(i)] = 0;
  }
  return {
    operator_name: path.basename(opDir),
    schema_version: "2.0",
    max_stage: maxStage,
    current_stage: 1,
    stage_status: stageStatus,
    stage_retry_count: stageRetry,
    artifact_hashes: {},
    rollback_history: [],
    last_updated: new Date().toISOString(),
  };
}

function readStateOrInit(statePath: string, maxStage: number): OrchestratorState {
  if (!fs.existsSync(statePath)) {
    return buildInitialState(path.dirname(statePath), maxStage);
  }
  const raw = fs.readFileSync(statePath, "utf8");
  return parseState(raw);
}

function writeStateAtomically(statePath: string, state: OrchestratorState): void {
  const tmpPath = `${statePath}.tmp`;
  fs.writeFileSync(tmpPath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
  fs.renameSync(tmpPath, statePath);
}

function computeFileHash(filePath: string): string | null {
  if (!fs.existsSync(filePath)) return null;
  const content = fs.readFileSync(filePath, "utf8");
  return crypto.createHash("sha256").update(content).digest("hex");
}

/**
 * Build the TransitionInput payload for applyTransition() based on the
 * tool args. Only fields relevant to each action are forwarded.
 */
function buildTransitionInput(action: TransitionAction, args: Record<string, unknown>): TransitionInput {
  switch (action) {
    case "init":
      return {
        action,
        stage: args.stage !== undefined ? Number(args.stage) : 1,
        max_stage: args.max_stage !== undefined ? Number(args.max_stage) : undefined,
      };
    case "start_stage":
      return { action, stage: Number(args.stage), reason: args.reason as string | undefined };
    case "complete_stage":
      return { action, stage: Number(args.stage) };
    case "fail_stage":
      return { action, stage: Number(args.stage), reason: args.reason as string | undefined };
    case "record_artifact_hash":
      return {
        action,
        name: String(args.name ?? ""),
        hash: String(args.hash ?? ""),
      };
    case "rollback_to_stage":
      return {
        action,
        target_stage: Number(args.target_stage),
        reason: String(args.reason ?? ""),
        failure_category: args.failure_category as string | undefined,
      };
    default: {
      throw new Error(`unsupported action: ${action}`);
    }
  }
}

export const PyptoProStateTransitionPlugin: Plugin = async (input) => {
  const client = input.client;
  const baseDir = input.directory || input.worktree || process.cwd();

  return {
    tool: {
      state_transition: tool({
        description:
          "Safely transition .orchestrator_state.json for the PyPTO-Pro workflow (schema v2.0, 4 stages). " +
          "Stage actions: init (stage=1 only, first call), start_stage (set stage to in_progress for retry), " +
          "complete_stage (mark done + auto-advance to next stage), fail_stage (mark failed + increment retry). " +
          "Other actions: record_artifact_hash (snapshot SPEC.md/golden/DESIGN.md hashes), " +
          "rollback_to_stage (return to an earlier stage with reason and optional failure_category — wipes downstream stages). " +
          "Pro has no lint gate; the verifier agent is the gate. SPEC.md freeze is enforced: " +
          "complete_stage(1) records the SPEC.md hash, and complete_stage(>=3) rejects if SPEC.md changed.",
        args: {
          opDir: tool.schema.string(),
          action: tool.schema.string(),
          // Stage actions
          stage: tool.schema.number().optional(),
          max_stage: tool.schema.number().optional(),
          reason: tool.schema.string().optional(),
          // Artifact hash
          name: tool.schema.string().optional(),
          hash: tool.schema.string().optional(),
          // Rollback
          target_stage: tool.schema.number().optional(),
          failure_category: tool.schema.string().optional(),
        },
        execute: async (args, context) => {
          // ── Permission check: AGENTS.md primary only ──
          const callerAgent = context?.agent ?? "";
          if (!ALLOWED_AGENTS.has(callerAgent)) {
            throw new Error(
              `permission denied: state_transition is restricted to the PyPTO-Pro primary, ` +
              `but was called by "${callerAgent || "(unknown)"}". ` +
              `Subagents must not modify .orchestrator_state.json; return stage results to the orchestrator instead.`,
            );
          }

          const action = String(args.action) as TransitionAction;
          if (!ALLOWED_ACTIONS.has(action)) {
            throw new Error(`unsupported action: ${args.action}`);
          }

          const opDir = path.isAbsolute(args.opDir)
            ? args.opDir
            : path.resolve(baseDir, args.opDir);
          const customRoot = path.resolve(baseDir, "custom");
          const relativeToCustomRoot = path.relative(customRoot, opDir);
          if (
            relativeToCustomRoot === "" ||
            relativeToCustomRoot === ".." ||
            relativeToCustomRoot.startsWith(`..${path.sep}`) ||
            path.isAbsolute(relativeToCustomRoot)
          ) {
            throw new Error(
              `permission denied: state_transition may only update an operator directory under ` +
              `${customRoot}, but received ${opDir}.`,
            );
          }
          const statePath = path.join(opDir, ".orchestrator_state.json");
          if (action === "init") {
            fs.mkdirSync(opDir, { recursive: true });
          }
          const desiredMaxStage = args.max_stage !== undefined ? Number(args.max_stage) : DEFAULT_MAX_STAGE;
          const prevState = readStateOrInit(statePath, desiredMaxStage);

          // ── SPEC.md freeze enforcement ──
          // complete_stage(1) auto-records the SPEC.md hash into artifact_hashes.spec_md.
          // From stage 3 onward, every complete_stage rejects if SPEC.md changed.
          // To legitimately re-edit SPEC.md, the orchestrator must call
          // rollback_to_stage(target_stage=1, reason=...).
          const specPath = path.join(opDir, "SPEC.md");
          if (action === "complete_stage" && Number(args.stage) === 1) {
            const hash = computeFileHash(specPath);
            if (hash) {
              prevState.artifact_hashes = prevState.artifact_hashes ?? {};
              prevState.artifact_hashes.spec_md = hash;
            }
          }
          if (action === "complete_stage" && Number(args.stage) >= 3) {
            const savedHash = prevState.artifact_hashes?.spec_md;
            if (typeof savedHash === "string") {
              const currentHash = computeFileHash(specPath);
              if (currentHash && currentHash !== savedHash) {
                throw new Error(
                  `SPEC.md freeze violation: SPEC.md was modified after Stage 1 completion. ` +
                  `current hash=${currentHash.slice(0, 12)}… recorded hash=${savedHash.slice(0, 12)}…. ` +
                  `To legitimately revise the spec, call rollback_to_stage(target_stage=1, reason=...).`,
                );
              }
            }
          }

          // Build and apply the transition
          const transitionInput = buildTransitionInput(action, args as Record<string, unknown>);
          const nextState = applyTransition(prevState, transitionInput);

          writeStateAtomically(statePath, nextState);

          // Audit log (non-fatal)
          try {
            await client.app.log({
              body: {
                service: "pypto-pro-state-transition",
                level: "info",
                message: `[state_transition] ${action} completed`,
                extra: {
                  opDir,
                  action,
                  stage: args.stage,
                  target_stage: args.target_stage,
                  reason: args.reason ?? "",
                },
              },
            });
          } catch {
            // no-op
          }

          return JSON.stringify({
            ok: true,
            action,
            stage: args.stage,
            target_stage: args.target_stage,
            current_stage: nextState.current_stage,
            statePath,
          });
        },
      }),
    },
  };
};
