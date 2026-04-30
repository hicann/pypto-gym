#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
修复 transformers 模型代码导入语句
将相对导入改为绝对导入，适用于从 transformers 包复制到本地目录的场景

用法: python3 fix_imports.py <file_path>
例如: python3 fix_imports.py /data/models/Nanbeige4.1-3B/core/modeling_llama.py
"""

import logging
import re
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')


def fix_imports(file_path):
    """修复导入语句"""

    with open(file_path, "r") as f:
        content = f.read()

    # 处理单行导入: from ...xxx import yyy 或 from ...xxx.yyy import zzz
    # 转换为: from transformers.xxx import yyy 或 from transformers.xxx.yyy import zzz
    content = re.sub(
        r"from\s+\.\.\.([\w\.]+)\s+import", r"from transformers.\1 import", content
    )

    # 处理多行导入块中的 from ...xxx ( 或 from ...xxx.yyy (
    # 例如: from ...modeling_outputs import (
    # 转换为: from transformers.modeling_outputs import (
    content = re.sub(
        r"from\s+\.\.\.([\w\.]+)\s+import\s*\(",
        r"from transformers.\1 import (",
        content,
    )

    # 处理 from ..xxx 或 from ..xxx.yyy (两层相对导入)
    content = re.sub(
        r"from\s+\.\.([\w\.]+)\s+import", r"from transformers.\1 import", content
    )

    # 处理单独的 import 行（非 from ... 导入）
    # 例如: ...modeling_outputs import (
    # 这些是上一行 from ... 的续行，需要修复
    # 匹配模式: 行首有空格，然后是模块名 import
    lines = content.split("\n")
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 检查是否是残缺的导入行（没有 from 关键字）
        # 例如: cache_utils import Cache, DynamicCache
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("from")
            and not stripped.startswith("import")
        ):
            # 检查是否是 transformers 内部模块的导入
            # 模式: word import something
            match = re.match(r"^(\w+)\s+import\s+", stripped)
            if match:
                module_name = match.group(1)
                # 检查是否是 transformers 内部模块
                internal_modules = [
                    "activations",
                    "cache_utils",
                    "configuration_utils",
                    "generation",
                    "integrations",
                    "masking_utils",
                    "modeling_layers",
                    "modeling_outputs",
                    "modeling_rope_utils",
                    "modeling_utils",
                    "processing_utils",
                    "utils",
                ]
                if module_name in internal_modules or module_name.startswith("utils."):
                    line = line.replace(
                        module_name + " import", f"transformers.{module_name} import"
                    )

        fixed_lines.append(line)
        i += 1

    content = "\n".join(fixed_lines)

    # 保持同一目录下的相对导入不变
    # from .configuration_xxx 保持原样
    # 这行不需要修改，已经是正确的

    with open(file_path, "w") as f:
        f.write(content)

    logging.info(f"已修复导入: {file_path}")


def main():
    if len(sys.argv) < 2:
        logging.info(__doc__)
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        logging.error(f"文件不存在: {file_path}")
        sys.exit(1)

    # 先备份
    backup_path = file_path + ".bak"
    import shutil

    shutil.copy(file_path, backup_path)
    logging.info(f"已创建备份: {backup_path}")

    fix_imports(file_path)

    # 验证修复后的文件可以导入
    logging.info(f"\n验证修复结果:")
    logging.info(f"  请检查文件内容，确认导入语句正确")
    logging.info(f"  如有问题，可从备份恢复: cp {backup_path} {file_path}")


if __name__ == "__main__":
    main()
