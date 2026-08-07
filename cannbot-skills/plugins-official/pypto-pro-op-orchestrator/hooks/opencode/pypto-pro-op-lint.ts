// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import type { Plugin } from "@opencode-ai/plugin";
import path from "node:path";
import {
  formatPluginError,
  isStateWriteCommand,
  parseHookOutput,
} from "./lib/lint-output";

function appendMessage(output: { output?: string }, message: string): void {
  output.output = `${output.output ?? ""}\n\n${message}`.trim();
}

function resolvePath(baseDir: string, filePath: string): string {
  if (!filePath) return "";
  return path.isAbsolute(filePath) ? filePath : path.resolve(baseDir, filePath);
}

const OP_ARTIFACT_RE = /\b\w+_(?:golden|golden_cpu|golden_stage\w*)\.py\b|\btest_\w+\.py\b/;

function isBashWriteToOpArtifact(command: string): boolean {
  if (!OP_ARTIFACT_RE.test(command)) return false;

  if (/>>?\s*\S*(?:\w+_golden\w*\.py|test_\w+\.py)/.test(command)) {
    return true;
  }

  if (/\bpython3?\b\s+(?:-\w+\s+)*-c\b/.test(command)) {
    const writeIndicators = [
      /\.write\w*\(/,
      /\.truncate\(/,
      /open\([^)]*['"][wax]b?\+?['"]/,
      /Path\([^)]*\)\.unlink\(/,
      /shutil\.(copy\w*|move|rmtree)\(/,
      /os\.(remove|unlink|rename)\(/,
    ];
    if (writeIndicators.some((p) => p.test(command))) return true;
  }

  const segments = command.split(/&&|\|\||;|\|/).map((s) => s.trim());
  for (const raw of segments) {
    if (!OP_ARTIFACT_RE.test(raw)) continue;
    const seg = raw.replace(/^sudo\s+/, "");
    if (/^python3?\b.*\s-c\b/.test(seg)) continue;
    if (/^(tee|cp|mv|rm|install|ln)\b/.test(seg)) return true;
    if (/^sed\b.*-i\b/.test(seg)) return true;
  }
  return false;
}

export const PyptoProOpLintPlugin: Plugin = async (input) => {
  const $ = input.$;
  const client = input.client;
  const baseDir = input.directory || input.worktree || process.cwd();
  const lintScript = new URL(
    "../hooks/pypto-pro-op-lint/pypto_pro_op_lint.py",
    import.meta.url,
  ).pathname;

  async function execHookJson(hook: string, payload: unknown): Promise<string> {
    return await $`python3 ${lintScript} --hook ${hook}`
      .env({
        PYPTO_PRO_OP_LINT_HOOK_INPUT: JSON.stringify(payload),
      })
      .quiet()
      .text();
  }

  const WRITE_TOOLS = new Set(["write", "edit", "multiedit"]);
  const BASH_TOOLS = new Set(["bash", "shell"]);
  function normTool(t: unknown): string {
    return typeof t === "string"
      ? t.toLowerCase()
      : typeof (t as { name?: unknown })?.name === "string"
        ? String((t as { name?: unknown }).name).toLowerCase()
        : "";
  }
  function extractFilePath(args: any): string {
    return String(
      args?.file_path ?? args?.filePath ?? args?.path ?? args?.target_file ?? ""
    );
  }
  const trace = process.env.PYPTO_PRO_OP_LINT_TRACE === "1";
  function tdebug(label: string, payload: Record<string, unknown>): void {
    if (!trace) return;
    try {
      console.error(`[pypto-pro-op-lint:trace] ${label}`, JSON.stringify(payload));
    } catch {
      // ignore
    }
  }

  const LINT_RUN_AGENTS = new Set([
    "pypto-pro-op-orchestrator",
    "pypto-pro-op-planner",
    "pypto-pro-op-mathematician",
    "pypto-pro-op-architect",
    "pypto-pro-op-coder",
    "pypto-pro-op-verifier",
  ]);
  const agentBySession = new Map<string, string>();
  function rememberAgent(input: { sessionID?: unknown; agent?: unknown }): void {
    if (typeof input.sessionID !== "string" || typeof input.agent !== "string") return;
    agentBySession.set(input.sessionID, input.agent);
  }
  function shouldRunLint(input: { sessionID?: unknown }, toolOutput?: unknown): boolean {
    const sessionID = typeof input.sessionID === "string" ? input.sessionID : "";
    const agent = sessionID ? agentBySession.get(sessionID) : undefined;
    const serialized = JSON.stringify([input, toolOutput]).replaceAll("\\\\", "/");
    const targetsCustomRoot = serialized.includes("custom/");
    const primaryOrUnknown = agent === undefined || agent === "build";
    const run = primaryOrUnknown ? targetsCustomRoot : LINT_RUN_AGENTS.has(agent);
    tdebug("gate", { sessionID, agent, run });
    return run;
  }

  return {
    "chat.message": async (chatInput, _output) => {
      rememberAgent(chatInput);
    },
    "chat.params": async (chatInput, _output) => {
      rememberAgent(chatInput);
    },
    "tool.execute.after": async (rawArgs, output) => {
      if (!shouldRunLint(rawArgs)) return;
      const { tool, args } = rawArgs as { tool: unknown; args: any };
      const toolName = normTool(tool);
      const filePath = extractFilePath(args);
      tdebug("after.fired", { rawTool: tool, toolName, filePath });

      if (
        WRITE_TOOLS.has(toolName) &&
        (filePath.endsWith("_golden.py") ||
          filePath.endsWith("_golden_cpu.py") ||
          /_golden_stage\w*\.py$/.test(filePath) ||
          /test_\w+\.py$/.test(filePath))
      ) {
        tdebug("after.match", { toolName, filePath });
        try {
          const raw = await execHookJson("post-edit", {
            tool_input: { file_path: filePath },
          });
          const parsed = parseHookOutput(raw);
          if (parsed.additionalContext) appendMessage(output, parsed.additionalContext);
          if (parsed.decision === "block") {
            throw new Error(parsed.reason || "[pypto-pro-op-lint] 产物写入后门禁未通过");
          }
        } catch (error) {
          if (error instanceof Error) {
            throw error;
          }
          throw new Error(formatPluginError("post-edit", error));
        }
        return;
      }

      return;
    },

    "tool.execute.before": async (rawArgs, output) => {
      if (!shouldRunLint(rawArgs, output)) return;
      const { tool } = rawArgs as { tool: unknown };
      const toolName = normTool(tool);
      const command = String(output.args?.command ?? "");
      tdebug("before.fired", { rawTool: tool, toolName, hasCmd: !!command });

      if (BASH_TOOLS.has(toolName) && isBashWriteToOpArtifact(command)) {
        throw new Error(
          "[pypto-pro-op-lint] 禁止通过 bash/shell 写入算子产物文件 (_golden*.py / test_*.py)。\n"
          + "原因: bash 写入不会触发 PostToolUse lint hook，违规不会以 in-band block 反馈给当前 agent，可能拖到 Verifier 阶段才暴露。\n"
          + "处理方式: 请使用 Write / Edit / MultiEdit 工具直接写入；这些工具会经 tool.execute.after 触发 post-edit hook，并把 S0 violation 作为 decision: block 返回。\n"
          + "如果是查看文件 (cat/grep/diff/stat), 请改用对应只读命令; 这条规则只拦截写入。"
        );
      }
      if (BASH_TOOLS.has(toolName) && isStateWriteCommand(command)) {
        throw new Error(
          "[pypto-pro-op-lint] 禁止通过 bash/shell 直接写入 .orchestrator_state.json。"
          + "请使用 state_transition 工具更新状态文件。",
        );
      }

      if (!WRITE_TOOLS.has(toolName)) return;

      const filePath = extractFilePath(output.args ?? {});
      const absPath = resolvePath(baseDir, filePath);
      if (!absPath) return;

      if (absPath.endsWith(`${path.sep}.orchestrator_state.json`)) {
        throw new Error(
          "[pypto-pro-op-lint] 禁止直接修改 .orchestrator_state.json。"
          + "请使用 state_transition 工具进行阶段状态迁移。",
        );
      }
    },
  };
};
