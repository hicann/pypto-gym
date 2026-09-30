#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Refresh or verify the self-contained Ascriptor source snapshot in this plugin."""
import argparse
import json
from pathlib import Path
import sys

RUNTIME = Path(__file__).resolve().parent / "scriptor-runtime"
sys.path.insert(0, str(RUNTIME))
from scriptorlib.common import ContractError
from scriptorlib.sources import refresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path,
                        default=Path(__file__).resolve().parents[1] / "resources/ascriptor")
    parser.add_argument("--check", action="store_true",
                        help="Verify indexed source bytes and references; write nothing")
    args = parser.parse_args()
    try:
        result = refresh(args.destination, check=args.check)
    except (ContractError, OSError, ValueError) as exc:
        parser.exit(1, f"ascriptor snapshot verification failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.check and not result["synchronized"]:
        parser.exit(1, "snapshot index differs from the source files; refresh it\n")


if __name__ == "__main__":
    main()
