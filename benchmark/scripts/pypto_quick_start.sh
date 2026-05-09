#!/usr/bin/env bash
# One-command quick start for the curated PyPTO benchmark set.

set -euo pipefail

if [ "$#" -ne 0 ]; then
  echo "ERROR: pypto_quick_start.sh does not accept arguments." >&2
  echo "Usage: bash benchmark/scripts/pypto_quick_start.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: python3/python not found in PATH" >&2
  exit 1
fi

echo "==> Download or update PyPTO"
bash benchmark/scripts/download_pypto.sh

echo "==> Download or update KernelBench"
bash benchmark/scripts/download_kernelbench.sh

echo "==> Run curated PyPTO benchmark"
exec "${PYTHON_BIN}" -m benchmark run --config configs/pypto.yaml
