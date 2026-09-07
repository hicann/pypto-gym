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
import {
  formatGateFinding,
  parseGateSummary,
  type GateSummary,
} from "./lib/lint-output";

const ALLOWED_ACTIONS = new Set<TransitionAction>([
  "init",
  "start_stage",
  "complete_stage",
  "fail_stage",
  "record_artifact_hash",
  "rollback_to_stage",
  "plan_stage4",
  "start_module",
  "submit_for_verify",
  "complete_module",
  "fail_module",
]);

/** The AGENTS.md primary runs as "build"; the legacy named primary remains compatible. */
const ALLOWED_AGENTS = new Set<string>([
  "build",
  "pypto-pro-op-orchestrator",
]);

const WORKFLOW_STAGE_COUNT = 5;

function parseState(content: string): OrchestratorState {
  const parsed = JSON.parse(content);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("invalid orchestrator state file content");
  }
  return parsed as OrchestratorState;
}

function buildInitialState(opDir: string): OrchestratorState {
  const stageStatus: Record<string, string> = {};
  const stageRetry: Record<string, number> = {};
  for (let i = 1; i <= WORKFLOW_STAGE_COUNT; i++) {
    stageStatus[String(i)] = "pending";
    stageRetry[String(i)] = 0;
  }
  return {
    operator_name: path.basename(opDir),
    schema_version: "2.2",
    max_stage: WORKFLOW_STAGE_COUNT,
    current_stage: 1,
    stage_status: stageStatus,
    stage_retry_count: stageRetry,
    artifact_hashes: {},
    rollback_history: [],
    last_updated: new Date().toISOString(),
  };
}

function readStateOrInit(statePath: string): OrchestratorState {
  if (!fs.existsSync(statePath)) {
    return buildInitialState(path.dirname(statePath));
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
    case "plan_stage4":
      return {
        action,
        module_count: Number(args.module_count ?? 0),
        is_fusion: Boolean(args.is_fusion),
      };
    case "start_module":
      return { action, module: String(args.module ?? "") };
    case "submit_for_verify":
      return { action, module: String(args.module ?? "") };
    case "complete_module":
      return { action, module: String(args.module ?? "") };
    case "fail_module":
      return {
        action,
        module: String(args.module ?? ""),
        failure_category: String(args.failure_category ?? ""),
        failing_module_boundary: args.failing_module_boundary as string | undefined,
        last_error: args.last_error as string | undefined,
      };
    default: {
      throw new Error(`unsupported action: ${action}`);
    }
  }
}

export const PyptoProStateTransitionPlugin: Plugin = async (input) => {
  const $ = input.$;
  const client = input.client;
  const baseDir = input.directory || input.worktree || process.cwd();
  const lintScript = new URL(
    "../hooks/pypto-pro-op-lint/pypto_pro_op_lint.py",
    import.meta.url,
  ).pathname;

  async function runGateIfNeeded(
    opDir: string,
    action: TransitionAction,
    stage: number | undefined,
    module: string | undefined,
  ): Promise<GateSummary> {
    let lintCmd;
    if (action === "complete_stage" && stage !== undefined && stage <= 4) {
      lintCmd = $`python3 ${lintScript} --check-gate --op-dir ${opDir} --stage ${stage}`;
    } else if (
      (action === "submit_for_verify" || action === "complete_module") &&
      module !== undefined
    ) {
      lintCmd = $`python3 ${lintScript} --check-module-gate --op-dir ${opDir} --module ${module}`;
    } else {
      return { warnCount: 0, infoCount: 0, failFindings: [] };
    }

    const result = await lintCmd.cwd(baseDir).quiet().nothrow();
    const raw = result.stdout.toString();
    const summary = parseGateSummary(raw);
    const hasBlockingFinding = summary.failFindings.some(
      (finding) => finding.severity === "S0" || finding.severity === "S1",
    );

    if (result.exitCode !== 0 || hasBlockingFinding) {
      const details = summary.failFindings
        .map(formatGateFinding)
        .join("\n");
      const scope =
        action === "complete_module"
          ? `Module ${module ?? "?"} completion`
          : action === "submit_for_verify"
            ? `Module ${module ?? "?"} verify-handoff`
            : `Stage ${stage ?? "?"} completion`;
      const guidance =
        action === "complete_module" || action === "submit_for_verify"
          ? "\n\nRecommended next step: re-dispatch pypto-pro-op-coder for the same module, " +
            "instruct it to fix the violations listed above, and call complete_module (or " +
            "submit_for_verify) again."
          : "\n\nRecommended next step: re-dispatch the upstream agent with the violations " +
            "listed above. After the agent fixes the artifacts, call complete_stage again.";
      throw new Error(
        details
          ? `${scope} blocked by lint rule violations:\n${details}${guidance}`
          : `lint script returned exit code ${result.exitCode} with no parsable findings`,
      );
    }

    return summary;
  }

  return {
    tool: {
      state_transition: tool({
        description:
          "Safely transition .orchestrator_state.json for the PyPTO-Pro workflow (schema v2.2, 5 stages). " +
          "Stage actions: init (stage=1 only, first call), start_stage (set stage to in_progress for retry), " +
          "complete_stage (mark done + auto-advance to next stage), fail_stage (mark failed + increment retry). " +
          "Other actions: record_artifact_hash (snapshot SPEC.md/golden/DESIGN.md hashes), " +
          "rollback_to_stage (return to an earlier stage with reason and optional failure_category — wipes downstream stages). " +
          "Stage 4 path actions: plan_stage4 (set stage4_path L0/L1 based on is_fusion, init stage4_modules for L1), " +
          "start_module / submit_for_verify / complete_module / fail_module (L1 per-Module loop). " +
          "Lint gate fires on complete_stage / submit_for_verify / complete_module as a side effect " +
          "(before state mutation; complete_stage follows verifier PASS except default Stage 2, which follows mathematician success). " +
          "SPEC.md freeze is enforced: " +
          "complete_stage(1) records the SPEC.md hash, and complete_stage(>=3) rejects if SPEC.md changed.",
        args: {
          opDir: tool.schema.string(),
          action: tool.schema.string(),
          // Stage actions
          stage: tool.schema.number().optional(),
          reason: tool.schema.string().optional(),
          // Artifact hash
          name: tool.schema.string().optional(),
          hash: tool.schema.string().optional(),
          // Rollback
          target_stage: tool.schema.number().optional(),
          failure_category: tool.schema.string().optional(),
          // Stage 4 path
          module_count: tool.schema.number().optional(),
          is_fusion: tool.schema.boolean().optional(),
          // Module actions
          module: tool.schema.string().optional(),
          failing_module_boundary: tool.schema.string().optional(),
          last_error: tool.schema.string().optional(),
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
          const prevState = readStateOrInit(statePath);

          // ── Lint gate ──
          // Fires on complete_stage / submit_for_verify / complete_module as a side effect.
          // Runs before state mutation. submit_for_verify is the pre-verifier handoff;
          // complete_stage follows verifier PASS except default Stage 2, which follows
          // mathematician success. complete_module follows verifier PASS. A lint FAIL
          // always keeps state unchanged.
          let gateSummary: GateSummary = { warnCount: 0, infoCount: 0, failFindings: [] };
          if (
            action === "complete_stage" ||
            action === "submit_for_verify" ||
            action === "complete_module"
          ) {
            try {
              gateSummary = await runGateIfNeeded(
                opDir,
                action,
                args.stage as number | undefined,
                args.module as string | undefined,
              );
            } catch (error) {
              const detail = error instanceof Error ? error.message : String(error);
              throw new Error(
                `[pypto-pro-op-lint] gate blocked: action=${action}, ` +
                `stage=${args.stage ?? "-"}, module=${args.module ?? "-"}, op_dir=${opDir}\n${detail}`,
              );
            }
          }

          // ── SPEC.md freeze enforcement ──
          // complete_stage(1) auto-records the SPEC.md hash into artifact_hashes.spec_md.
          // From stage 3 onward, every complete_stage rejects if SPEC.md changed.
          // To legitimately re-edit SPEC.md, the orchestrator must call
          // rollback_to_stage(target_stage=1, reason=...).
          const specPath = path.join(opDir, "SPEC.md");
          if (action === "complete_stage" && Number(args.stage) === 1) {
            const hash = computeFileHash(specPath);
            if (!hash) {
              // Skipping the record when SPEC.md is absent disarmed the freeze for the
              // whole run: `typeof savedHash === "string"` stayed false and the spec was
              // then freely editable with no signal. Stage 1's deliverable IS SPEC.md, so
              // completing it without one is the earlier error.
              throw new Error(
                `cannot complete Stage 1: ${specPath} does not exist. SPEC.md is Stage 1's ` +
                `deliverable and the artifact the freeze is taken over; completing without ` +
                `it would leave the freeze unarmed for the rest of the run.`,
              );
            }
            prevState.artifact_hashes = prevState.artifact_hashes ?? {};
            prevState.artifact_hashes.spec_md = hash;
          }
          if (action === "complete_stage" && Number(args.stage) >= 3) {
            const savedHash = prevState.artifact_hashes?.spec_md;
            if (typeof savedHash === "string") {
              const currentHash = computeFileHash(specPath);
              if (currentHash === null) {
                // Deleting or renaming SPEC.md used to SATISFY the freeze: computeFileHash
                // returns null for a missing path and `currentHash &&` short-circuited.
                // Absence is the strongest form of "modified", not an exemption.
                throw new Error(
                  `SPEC.md freeze violation: ${specPath} no longer exists. It was recorded ` +
                  `at Stage 1 (hash=${savedHash.slice(0, 12)}…) and the freeze is taken over ` +
                  `that file; deleting or renaming it is a spec change. To legitimately ` +
                  `revise the spec, call rollback_to_stage(target_stage=1, reason=...).`,
                );
              }
              if (currentHash !== savedHash) {
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
                  module: args.module ?? "",
                  module_count: args.module_count,
                  is_fusion: args.is_fusion,
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
            stage4_path: nextState.stage4_path,
            module: args.module,
            statePath,
            warnCount: gateSummary.warnCount,
            infoCount: gateSummary.infoCount,
          });
        },
      }),
    },
  };
};
