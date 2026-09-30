# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Place inside a copy of the accepted examples/api/axpb; run with ascriptor[sim].

It imports that folder's own `kernel` module, so it belongs in the folder rather than beside it.
"""

import argparse
import json
from pathlib import Path

import ascriptor
import torch
from kernel import axpb
from ascriptor.backends.sim.pipesim import simulate
from ascriptor.passes import PIPELINE, PassManager
from ascriptor.passes.autosync import check_balance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/pipe-axpb"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lowered = PassManager(PIPELINE).run(axpb.ir())
    balance = check_balance(lowered)
    assert balance == [], balance
    cases = []
    for seed in (7101, 7102):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randint(-64, 65, (1, 64), generator=generator).float()
        y = torch.randint(-64, 65, (1, 64), generator=generator).float()
        original_x, original_y = x.clone(), y.clone()
        output = torch.full_like(x, float("nan"))
        expected = 2 * x + y
        result = simulate(lowered, (x, y, output), block_dim=1, timeout=30,
                          seed_outputs=True, check_gm=True, processes=False)
        result.write_trace(args.output / f"trace-{seed}.json")
        (args.output / f"schedule-{seed}.json").write_text(json.dumps(result.report, indent=2) + "\n")
        assert result.hazards == [], result.hazards
        assert result.report.get("deadlock") is None, result.report
        assert len(result.outputs) == 1, "Output count changed"
        actual = result.outputs[0]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert bool(torch.isfinite(actual).all()), "Unwritten or non-finite output"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(x, original_x, rtol=0, atol=0)
        torch.testing.assert_close(y, original_y, rtol=0, atol=0)
        corrupt = actual.clone()
        corrupt[0, -1] += 1
        rejected = 0
        for wrong in (torch.zeros_like(actual), corrupt):
            try:
                torch.testing.assert_close(wrong, expected, rtol=0, atol=0)
            except AssertionError:
                rejected += 1
            else:
                raise AssertionError("The comparator accepted a bad output")
        cases.append({"seed": seed, "comparison": "exact", "negative_controls": rejected,
                      "event_balance": balance, "hazards": result.hazards,
                      "deadlock": result.report.get("deadlock"), "model_cycles": result.cycles})
    print(json.dumps({"stage": "pipesim", "ascriptor": ascriptor.__version__,
                      "import_origin": ascriptor.__file__, "check_gm": True, "cases": cases}))


if __name__ == "__main__":
    main()
