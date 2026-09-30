#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Print the authored compact kernel startup guide without source-hash dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
PACK = AGENT_ROOT / "context/kernel-authoring.zh-CN.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", required=True, help="Print the compact guide")
    parser.parse_args()
    try:
        pack = PACK.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        parser.exit(1, f"Cannot read compact kernel context: {exc}\n")
    print(pack, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
