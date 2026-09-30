import { type Plugin, tool } from "@opencode-ai/plugin";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const PyptoProScriptorPlugin: Plugin = async (input) => {
  const config = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const receipt = JSON.parse(fs.readFileSync(path.join(config, "scriptor-install.json"), "utf8"));
  const cli = path.join(config, "scriptor/scripts/scriptor.py");
  const base = input.directory || input.worktree || process.cwd();
  const agents = new Map<string, string>();
  const primary = new Set(["build", "pypto-pro-op-orchestrator"]);
  function remember(value: any) {
    if (typeof value.sessionID === "string" && typeof value.agent === "string") agents.set(value.sessionID, value.agent);
  }
  function operator(file: string): string | undefined {
    const resolved = path.resolve(base, file);
    const custom = path.resolve(base, "custom");
    const managed = (directory: string) => [".scriptor/state.json", ".scriptor/receipts/pro-bootstrap.json",
      ".scriptor/bootstrap-restart.json"].some((name) => fs.existsSync(path.join(directory, name)));
    if (path.dirname(resolved) === custom && managed(resolved)) return resolved;
    let directory = path.dirname(resolved);
    while (directory === custom || directory.startsWith(custom + path.sep)) {
      if (managed(directory)) return directory;
      const parent = path.dirname(directory);
      if (parent === directory) break;
      directory = parent;
    }
    return undefined;
  }
  return {
    "chat.message": async (value) => { remember(value); },
    "chat.params": async (value) => { remember(value); },
    "shell.env": async (_value, output) => {
      output.env.CANNBOT_CONFIG_ROOT = config;
      output.env.CANNBOT_PYTHON = receipt.python;
    },
    tool: {
      scriptor_transition: tool({
        description: "Read or advance the Scriptor state ledger from a JSON request. Only the primary orchestrator may call it. Start or resume with init/status, follow next_request, and use actual sealed report paths. New tasks use entry_mode=from_pro; complete_prepare is legacy recovery only. detail=full returns the ledger. Final output: reports/final/final.md; done means archived, not necessarily target met.",
        args: {
          request: tool.schema.string().describe("JSON object with action, opDir (custom/<op>), and fields required by next_request. Report paths are relative to the operator directory."),
        },
        async execute(args, context) {
          if (!primary.has(context.agent)) throw new Error("scriptor state is owned by the primary agent");
          const request = JSON.parse(args.request);
          const result = await input.$`${receipt.python} ${cli} --config-root ${config} state --project ${base} --actor ${context.agent} --request-json ${JSON.stringify(request)}`.quiet().nothrow();
          if (result.exitCode !== 0) throw new Error(result.stderr.toString() || result.stdout.toString());
          return result.stdout.toString();
        },
      }),
    },
    "tool.execute.before": async (value, output) => {
      const actor = agents.get(value.sessionID) || "";
      const name = String(value.tool).toLowerCase();
      const args = output.args as Record<string, any>;
      if (["write", "edit", "multiedit", "apply_patch"].includes(name)) {
        const files = [args.filePath, args.file_path, args.path,
          ...(Array.isArray(args.edits) ? args.edits.map((x: any) => x.filePath || x.file_path || x.path) : [])].filter((x) => typeof x === "string");
        const patch = String(args.patchText || args.patch || "");
        for (const match of patch.matchAll(/^\*\*\* (?:(?:Add|Update|Delete) File:|Move to:) (.+)$/gm)) files.push(match[1]);
        for (const file of files) {
          if (/(?:^|\/)\.scriptor(?:\/|$)/.test(file.replaceAll("\\", "/"))) throw new Error("the .scriptor directory is managed by the Scriptor CLI");
          const op = operator(file);
          if (!op) continue;
          if (path.resolve(base, file) === op) throw new Error("the Scriptor operator directory is managed by the Scriptor CLI");
          const relative = path.relative(op, path.resolve(base, file)).replaceAll("\\", "/");
          if (/^reports\/bootstrap-restarts(?:\/|$)/.test(relative)) throw new Error("bootstrap restart history is managed by the Scriptor CLI");
          const statePath = path.join(op, ".scriptor/state.json");
          if (!fs.existsSync(statePath)) {
            if (fs.existsSync(path.join(op, ".scriptor/receipts/pro-bootstrap.json"))
                && /^(?:SPEC\.md|PRO_MATERIAL_INDEX\.md|EXPLORE_REPORT\.md|DESIGN\.md|DESIGN_BINDINGS\.json|module_interfaces\.yaml|GOLDEN_VALIDATION\.json|[^/]+_golden(?:_cpu)?\.py|(?:.*\/)?KB_SELECTION\.json)$/.test(relative)) {
              throw new Error("sealed Pro inputs: use bootstrap-restart before changing Plan, Golden or Design");
            }
            continue;
          }
          const state = JSON.parse(fs.readFileSync(statePath, "utf8"));
          if (/^reports\/final(?:\/|$)/.test(relative)) throw new Error("final results are generated by the state/report CLI; edit the verified inputs through their owning workflow");
          if (state.current_stage !== "prepare" && Object.hasOwn(state.frozen || {}, relative)) throw new Error("frozen input: use rollback_prepare to return to the owning upstream before changing the contract");
          if (/^reports\/(?:sealed|reviews)\//.test(relative) && actor !== "pypto-pro-scriptor-verifier") throw new Error("semantic reviews and sealed reports belong to the independent verifier");
        }
      }
      if (["bash", "shell"].includes(name)) {
        const command = String(args.command || "");
        if (/scriptor\.py[\s\S]*\sstate(?:\s|$)/.test(command) && !primary.has(actor)) throw new Error("subagents cannot use the state CLI");
        if (/scriptor\.py[\s\S]*\sbootstrap-restart(?:\s|$)/.test(command) && !primary.has(actor)) throw new Error("only the primary can restart a sealed Pro bootstrap");
        if (/scriptor\.py[\s\S]*\sreport(?:\s|$)/.test(command) && !primary.has(actor)) throw new Error("only the primary can regenerate final result views");
        if (/scriptor\.py[\s\S]*\sseal(?:\s|$)/.test(command) && actor !== "pypto-pro-scriptor-verifier") throw new Error("only the verifier seals reports");
        // Duplicating an existing descriptor (e.g. ls ... 2>&1) does not write a file.
        const mutationText = command.replace(/\b\d*>\s*&\s*\d+\b/g, "");
        const mutates = /(?:write_text|write_bytes|json\.dump|\btee\b|\bcp\b|\bmv\b|\brm\b|\brmdir\b|\bmkdir\b|\btouch\b|\bln\b|\bsed\b.*-i|\.move\(|\.rename\(|\.replace\(|\.remove\(|\.unlink\(|rmtree\(|>)/.test(mutationText);
        if (mutates) {
          const custom = path.resolve(base, "custom");
          const operations = [...command.matchAll(/(?:^|[;&|])\s*(?:sudo\s+)?(?:mv|rm|rmdir|cp|ln)\b([^;&|]*)/g)];
          if (fs.existsSync(custom)) {
            for (const entry of fs.readdirSync(custom, { withFileTypes: true })) {
              if (!entry.isDirectory()) continue;
              const root = path.join(custom, entry.name);
              if (operator(root) && operations.some((operation) =>
                (operation[1].match(/[^\s"']+/g) || []).some((part) =>
                  part.replace(/\/$/, "") === root || part.replace(/\/$/, "") === `custom/${entry.name}`))) {
                throw new Error("the Scriptor operator directory is managed by the Scriptor CLI");
              }
            }
          }
        }
        if (/reports\/bootstrap-restarts\//.test(command) && mutates) throw new Error("bootstrap restart history is managed by the Scriptor CLI");
        if (/reports\/(?:sealed|reviews)\//.test(command) && mutates && actor !== "pypto-pro-scriptor-verifier") throw new Error("verifier reports cannot be written through another agent's shell");
        if (/(?:^|\/)\.scriptor(?:\/|\b)/.test(command.replaceAll("\\", "/")) && mutates) throw new Error("the .scriptor directory is managed by the Scriptor CLI");
        if (/reports\/final(?:\/|\b)/.test(command) && /(?:write_text|write_bytes|json\.dump|\btee\b|\bcp\b|\bmv\b|\brm\b|\bsed\b.*-i|>)/.test(mutationText)) throw new Error("final result files must be generated by the state/report CLI");
      }
    },
  };
};
