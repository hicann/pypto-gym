# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Export failures are recorded without pretending to execute a device kernel."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/scriptor-runtime"))
from scriptorlib.common import ContractError,digest
from scriptorlib.runner import check

class ExportPreflightTests(unittest.TestCase):
    def test_failed_export_has_bound_error_and_no_device_claim(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);op=root/'custom/probe';(op/'.scriptor').mkdir(parents=True)
            (op/'SPEC.md').write_text('fixture only')
            (op/'.scriptor/state.json').write_text('{"delivery_sync_mode":"auto_mutex"}')
            with patch('scriptorlib.runner.activate'), patch('scriptorlib.runner.runtime_options',return_value={}), \
                 patch('scriptorlib.runner.read_spec',return_value={'op_name':'probe'}), \
                 patch('scriptorlib.runner._precision_engine',return_value=(None,{'policy':'legacy_tolerance'})), \
                 patch('scriptorlib.runner.verify_export',side_effect=ContractError('stale incomplete export')), \
                 patch('scriptorlib.runner._execute') as execute:
                result=check(root,op,'candidate')
            execute.assert_not_called()
            report=json.loads((op/result['report']).read_text())
            self.assertEqual(report['verdict'],'FAIL')
            self.assertEqual(report['sync_mode'],'auto_mutex')
            self.assertEqual(report['checks']['export'],'FAIL')
            for field in ['pypto_hardware','standalone','latency']:
                self.assertEqual(report['checks'][field],'NOT_RUN')
            self.assertEqual(report['metrics'],[])
            evidence=report['evidence'][0]
            self.assertEqual(digest(op/evidence['path']),evidence['sha256'])
            error=json.loads((op/evidence['path']).read_text())
            self.assertEqual(error['device_execution'],'NOT_RUN')
            self.assertIn('stale incomplete export',error['message'])

if __name__=='__main__':unittest.main()
