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

import io
import logging
import os
import shutil
import sys
import token
import tokenize

logging.basicConfig(level=logging.INFO, format='%(message)s')

_TRANSFORMERS_INTERNAL_MODULES = {
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
}


def _absolute_offset(line_offsets, position):
    row, column = position
    return line_offsets[row - 1] + column


def _token_context(content):
    lines = content.splitlines(keepends=True)
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    if not lines or not content.endswith(("\n", "\r")):
        line_offsets.append(offset)

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
    except (IndentationError, tokenize.TokenError):
        return None, None
    return tokens, line_offsets


def _apply_replacements(content, replacements):
    for start, end, replacement in reversed(replacements):
        content = content[:start] + replacement + content[end:]
    return content


def _rewrite_relative_imports(content):
    """仅改写 Python 代码 token 中两层及以上的相对导入前缀。"""
    tokens, line_offsets = _token_context(content)
    if tokens is None:
        return content

    replacements = []
    for index, current in enumerate(tokens):
        if current.type != token.NAME or current.string != "from":
            continue
        next_index = index + 1
        dot_count = 0
        first_dot = None
        last_dot = None
        while next_index < len(tokens):
            candidate = tokens[next_index]
            if candidate.type != token.OP or set(candidate.string) != {"."}:
                break
            first_dot = first_dot or candidate
            last_dot = candidate
            dot_count += len(candidate.string)
            next_index += 1
        if dot_count < 2 or next_index >= len(tokens):
            continue
        if tokens[next_index].type != token.NAME:
            continue
        start = _absolute_offset(line_offsets, first_dot.start)
        end = _absolute_offset(line_offsets, last_dot.end)
        replacements.append((start, end, "transformers."))

    return _apply_replacements(content, replacements)


def _is_statement_start(tokens, index):
    if index == 0:
        return True
    return tokens[index - 1].type in {
        token.INDENT, token.DEDENT, token.NEWLINE, tokenize.NL,
    }


def _residual_module(tokens, index):
    parts = [tokens[index].string]
    last_token = tokens[index]
    cursor = index + 1
    while cursor + 1 < len(tokens):
        dot = tokens[cursor]
        name = tokens[cursor + 1]
        if dot.type != token.OP or dot.string != "." or name.type != token.NAME:
            break
        parts.append(name.string)
        last_token = name
        cursor += 2
    if cursor >= len(tokens) or tokens[cursor].type != token.NAME:
        return None
    if tokens[cursor].string != "import":
        return None
    return ".".join(parts), last_token


def _rewrite_residual_imports(content):
    """仅修复代码 token 中缺少 ``from`` 的内部模块导入。"""
    tokens, line_offsets = _token_context(content)
    if tokens is None:
        return content
    replacements = []
    for index, current in enumerate(tokens):
        if current.type != token.NAME or not _is_statement_start(tokens, index):
            continue
        residual = _residual_module(tokens, index)
        if residual is None:
            continue
        module_name, last_token = residual
        if not (
            module_name in _TRANSFORMERS_INTERNAL_MODULES
            or module_name.startswith("utils.")
        ):
            continue
        start = _absolute_offset(line_offsets, current.start)
        end = _absolute_offset(line_offsets, last_token.end)
        replacements.append((start, end, f"from transformers.{module_name}"))
    return _apply_replacements(content, replacements)


def _rewrite_imports(content):
    """将离开 transformers 包后失效的内部导入改为绝对导入。"""
    content = _rewrite_relative_imports(content)
    content = _rewrite_residual_imports(content)

    return content


def fix_imports(file_path):
    """修复导入语句"""
    with open(file_path, "r") as f:
        content = _rewrite_imports(f.read())

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
    if os.path.exists(backup_path):
        logging.info(f"已有备份，保留原文件: {backup_path}")
    else:
        shutil.copy(file_path, backup_path)
        logging.info(f"已创建备份: {backup_path}")

    fix_imports(file_path)

    # 验证修复后的文件可以导入
    logging.info("\n验证修复结果:")
    logging.info("  请检查文件内容，确认导入语句正确")
    logging.info(f"  如有问题，可从备份恢复: cp {backup_path} {file_path}")


if __name__ == "__main__":
    main()
