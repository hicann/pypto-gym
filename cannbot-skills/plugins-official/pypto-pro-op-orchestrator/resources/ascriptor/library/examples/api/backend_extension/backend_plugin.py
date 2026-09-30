# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A registered backend that preserves CCE semantics and adds artifact metadata.

The example is intentionally a transparent adapter. A new instruction lowering
would need its own semantic and vendor compilation gates.
"""

from dataclasses import replace

from ascriptor.backends.base import Artifacts, Backend, Capabilities, ResourceLimits
from ascriptor.backends.cce import CceBackend


class TeachingCceBackend:
    name = "teaching_cce"

    def __init__(self):
        self._delegate = CceBackend()

    def capabilities(self) -> Capabilities:
        return self._delegate.capabilities()

    def resources(self, device) -> ResourceLimits:
        return self._delegate.resources(device)

    def compile(self, module, options=None) -> Artifacts:
        if module.attrs.get("ir") != "lowered/1":
            raise ValueError("TeachingCceBackend consumes lowered/1; compile source through compile_kernel")
        result = self._delegate.compile(module, options)
        return replace(result, metadata={**result.metadata, "teaching_adapter": self.name})


def check_protocol():
    backend = TeachingCceBackend()
    assert isinstance(backend, Backend)
    assert isinstance(backend.capabilities(), Capabilities)
    return backend
