# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime entry for a specialized, already Lowered Pro import."""
from dataclasses import dataclass, field

from ...ir import Module


@dataclass(frozen=True)
class ImportedKernel:
    module: Module
    location: dict | None = field(default=None, compare=False)  # the Pro kernel definition, for located refusals

    @property
    def name(self):
        return self.module.name

    @property
    def device(self):
        return self.module.device

    @property
    def mode(self):
        return self.module.attrs['mode']

    def ir(self):
        return self.module

    def executor(self, **options):
        """Create OpExec while preserving the exported launch and inout state."""
        from ...devices import load
        from ...runtime import OpExec
        from ...runtime.launch_config import launch_block_dim

        block_dim = launch_block_dim(self.module, options.get('block_dim'), 'the executor', self.location)
        device = options.get('device')
        if device is not None and load(device).device_type != load(self.device).device_type:
            raise ValueError('Imported device is fixed by the export specialization')
        inout = 'inout' in self.module.attrs['meta']['directions'].values()
        seed = options.get('seed_outputs', inout)
        if type(seed) is not bool:
            raise TypeError('seed_outputs must be a bool')
        if inout and not seed:
            raise ValueError('Imported inout parameters require seed_outputs=True')
        options['seed_outputs'] = seed
        options['block_dim'] = block_dim
        return OpExec(self, **options)
