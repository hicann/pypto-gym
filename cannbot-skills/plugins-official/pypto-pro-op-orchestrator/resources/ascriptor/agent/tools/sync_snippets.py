# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Synchronize small workflow anchors with their exact library-owned functions."""
import argparse
import ast
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]

def excerpt(path, function):
    text=path.read_text()
    tree=ast.parse(text)
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==function]
    if len(functions)!=1:
        raise ValueError(f'Expected one owner function: {function}')
    node=functions[0]
    start=min([node.lineno]+[item.lineno for item in node.decorator_list])
    return '\n'.join(text.splitlines()[start-1:node.end_lineno])

def render(text, name, code):
    start=f'<!-- code-anchor:{name}:start -->'
    end=f'<!-- code-anchor:{name}:end -->'
    if text.count(start)!=1 or text.count(end)!=1:
        raise ValueError(f'Missing or duplicate code anchor: {name}')
    return re.sub(re.escape(start)+r'.*?'+re.escape(end),
                  lambda _:start+'\n```python\n'+code+'\n```\n'+end,text,flags=re.S)

def synchronize(root, library_root, check=True):
    errors=[]
    for row in json.loads((root/'docs/code-anchors.json').read_text())['anchors']:
        code=excerpt(library_root/row['source'],row['function'])
        for language in ('en','zh-CN'):
            path=root/language/row['target']
            text=path.read_text()
            expected=render(text,row['id'],code)
            if check:
                if text!=expected:
                    errors.append(f'Stale code anchor: {language}/{row["target"]}: {row["id"]}')
            else:
                path.write_text(expected)
    return errors

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library-root',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    errors=synchronize(ROOT,args.library_root,args.check)
    print(json.dumps({'errors':errors,'checked':args.check}))
    return bool(errors)

if __name__=='__main__':
    raise SystemExit(main())
