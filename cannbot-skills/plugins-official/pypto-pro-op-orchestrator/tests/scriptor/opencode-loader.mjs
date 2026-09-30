// Test-only resolution for legacy extensionless TypeScript imports (OpenCode uses Bun).
import fs from "node:fs";
export async function resolve(specifier, context, nextResolve) {
  try { return await nextResolve(specifier, context); }
  catch (error) {
    if (specifier.startsWith(".") && context.parentURL) {
      const candidate = new URL(specifier + ".ts", context.parentURL);
      if (candidate.protocol === "file:" && fs.existsSync(candidate)) return nextResolve(candidate.href, context);
    }
    throw error;
  }
}
