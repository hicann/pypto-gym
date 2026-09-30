# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Never present skipped work or ambiguous profiler data as a passing check."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts/scriptor-runtime"))
from scriptorlib.common import ContractError
from scriptorlib.runner import _profile, summarize_checks


class ReportingTests(unittest.TestCase):
    def test_early_functional_failure_leaves_hardware_not_run(self):
        checks=summarize_checks({"p0":{"checks":{"functional":"FAIL"}}},
                                ("functional","pipesim","pypto_hardware","standalone","latency"))
        self.assertEqual(checks["functional"],"FAIL")
        self.assertEqual({checks[k] for k in checks if k!="functional"},{"NOT_RUN"})

    def test_every_case_must_have_executed(self):
        checks=summarize_checks({"p0":{"checks":{"latency":"PASS"}},"p1":{}},("latency",))
        self.assertEqual(checks["latency"],"NOT_RUN")

    def test_kernel_latency_requires_unambiguous_launch_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"summary.csv"
            for count in (32,64):
                with path.open("w") as output:
                    writer=csv.DictWriter(output,fieldnames=["Op Name","Task Duration(us)","aiv_vec_ratio"])
                    writer.writeheader()
                    for i in range(count):
                        writer.writerow({"Op Name":"probe_kernel","Task Duration(us)":10 if i<5 else 2,"aiv_vec_ratio":0.3})
                if count==32:
                    result=_profile(path,"probe_kernel")
                    self.assertEqual(result["latency_us"],2)
                    self.assertEqual(result["kept"],27)
                    self.assertAlmostEqual(result["vector_pipe_utilization_pct"],30)
                else:
                    with self.assertRaises(ContractError): _profile(path,"probe_kernel")


if __name__ == "__main__":
    unittest.main()
