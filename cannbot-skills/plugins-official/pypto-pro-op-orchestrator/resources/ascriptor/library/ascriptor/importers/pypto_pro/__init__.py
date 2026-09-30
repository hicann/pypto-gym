# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Experimental Pro transport; importing this module does not import Pro or torch.

Export/import preparation preserves the source graph. Instruction conversion is
separately admitted and returns only fully verified Lowered IR.
"""

from .schema import Bundle, ProImportError, dumps, loads
from .session import ImportPlan, prepare_import


def export_kernel(kernel, **options) -> Bundle:
    """Export a specialized Pro kernel using the optional pinned native adapter."""
    from .export import export_kernel as export

    return export(kernel, **options)


def import_module(bundle: Bundle):
    """Convert an admitted source graph to verified Lowered IR."""
    from .lower import import_module as convert

    return convert(bundle)


def import_kernel(bundle: Bundle):
    """Return a verified Lowered runtime entry with a specialization-safe executor."""
    from .entry import ImportedKernel

    module = import_module(bundle)
    graph = bundle.document["targets"][0]
    nodes = {node["id"]: node for node in graph["nodes"]}
    functions = [nodes[ref["$ref"]] for ref in nodes[graph["root"]]["fields"]["functions"]]
    kernel = next(f for f in functions if f["fields"]["func_type"]["name"] == "Opaque")  # import_module admitted one
    return ImportedKernel(module, kernel["location"])


__all__ = ["Bundle", "ImportPlan", "ProImportError", "dumps", "loads", "prepare_import", "export_kernel", "import_module", "import_kernel"]
