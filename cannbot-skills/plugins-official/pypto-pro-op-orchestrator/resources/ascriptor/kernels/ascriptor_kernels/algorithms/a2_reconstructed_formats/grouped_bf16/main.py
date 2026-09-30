# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the eleven reconstructed BF16 group formats through OpExec and check the references.

    python main.py                          # every case, functional simulator
    python main.py --list                   # the case ids, with their purpose
    python main.py --case hifx4_hierarchy   # one case
    python main.py --launcher aclnn         # cce backend, on this machine's card

Eleven entries share one kernel signature -- (x, y, rows, cols) -- and a case selects one by
`variant`. Three of them are the MX family (per-32 E8M0/E2M1, the two-level MBS macro factor,
and MXFP8 E5M2), six are a BF16 absmax scale over a 16, 32 or 64 group with either the E2M1
grid or a signed integer grid, and two are the three-level HiFX hierarchy at 4 and 5 bits.

Every output is a reconstructed floating value, never a packed payload byte: the scales and
codes these kernels build are internal numbers, so nothing here is a conversion ABI. The
filename `e1m2` in the source denotes a signed integer grid, not FP4 E1M2, which is why the
variants are named `signed_int4_*`.

This folder is self-contained: it imports the installed `ascriptor` and nothing from the
repository around it, so it runs from wherever you copy it to.
"""

import argparse

import torch
from ascriptor.runtime import LAUNCHERS, OpExec

from kernel import build_kernel
from reference import make_inputs, reference

# The contract declares these bodies on both A2 and A3, and `build_kernel` binds the selected
# entry to one facade; change this to "a3" to run the identical source against A3.
DEVICE = "a2"

# variant -> the kernel entry it launches. The names are the source's, kept so a reader can
# find the body; `ENTRIES` in kernel.py records which launch mode each one was written for.
ENTRY = {"e2m1_g16": "group16_bf16_fp4_e2m1_kernel",
         "signed_int4_g16": "group16_bf16_fp4_e1m2_kernel",
         "e2m1_g32": "group32_bf16_fp4_e2m1_kernel",
         "signed_int4_g32": "group32_bf16_fp4_e1m2_kernel",
         "e2m1_g64": "group64_bf16_fp4_e2m1_kernel",
         "signed_int4_g64": "group64_bf16_fp4_e1m2_kernel",
         "plain_mxfp4": "mxfp4_kernel_bf16",
         "mbs_mxfp4": "mbs_mxfp4_kernel",
         "mxfp8_e5m2": "mxfp8e5m2_kernel_bf16",
         "hifx4": "hifx4",
         "hifx5": "hifx5"}

# 2e-3 is one BF16 quantum near 1, which is the storage the output lands in; the references
# reproduce each variant's scale policy and rounding direction exactly rather than
# approximating it, so this is a statement about the final store and the FP32 operation order,
# not about the quantization. The relative L2 ceiling is what refuses a vacuous answer: at the
# 2^-64 and 2^-126 magnitudes these cases reach, an output of zeros passes atol on its own.
TOLERANCE = {"atol": 0.002, "rtol": 0.002, "max_relative_l2": 0.002}

CASES = [
    {"id": "e2m1_g16_minimum", "seed": 9000, "block_dim": 1,
     "purpose": "The smallest legal shape: one 16-element group on one core",
     "parameters": {"variant": "e2m1_g16", "rows": 1, "cols": 16, "pattern": "random",
                    "scale": 1.0}},
    {"id": "e2m1_g16_zero", "seed": 9001, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "e2m1_g16", "rows": 1, "cols": 16, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "e2m1_g16_boundaries", "seed": 9002, "block_dim": 1,
     "purpose": "Every E2M1 threshold, with an exact absmax anchor of 6 in each 16-group, so each "
                "midpoint is hit head on rather than approached",
     "parameters": {"variant": "e2m1_g16", "rows": 2, "cols": 16, "pattern": "boundaries",
                    "scale": 1.0}},
    {"id": "e2m1_g16_source_tail", "seed": 9003, "block_dim": 1,
     "purpose": "65 groups over one core: the last tile carries 1 of 16 groups",
     "parameters": {"variant": "e2m1_g16", "rows": 13, "cols": 80, "pattern": "random",
                    "scale": 0.2}},
    {"id": "e2m1_g16_reuse", "seed": 9004, "block_dim": 1,
     "purpose": "85 groups as 6 tiles over one core, 6 per core: every on-chip buffer and scale slot "
                "is reused from one tile to the next",
     "parameters": {"variant": "e2m1_g16", "rows": 17, "cols": 80, "pattern": "random",
                    "scale": 8.0}},
    {"id": "e2m1_g16_idle", "seed": 9005, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "e2m1_g16", "rows": 1, "cols": 16, "pattern": "random",
                    "scale": 0.02}},

    {"id": "signed_int4_g16_minimum", "seed": 9020, "block_dim": 1,
     "purpose": "The smallest legal shape: one 16-element group on one core",
     "parameters": {"variant": "signed_int4_g16", "rows": 1, "cols": 16, "pattern": "random",
                    "scale": 1.0}},
    {"id": "signed_int4_g16_zero", "seed": 9021, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "signed_int4_g16", "rows": 1, "cols": 16, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "signed_int4_g16_boundaries", "seed": 9022, "block_dim": 1,
     "purpose": "Every half-integer of the signed grid, with an exact absmax anchor of 7 in each "
                "16-group: ties round away from zero and the result clamps to [-8, 7]",
     "parameters": {"variant": "signed_int4_g16", "rows": 2, "cols": 16, "pattern": "boundaries",
                    "scale": 1.0}},
    {"id": "signed_int4_g16_source_tail", "seed": 9023, "block_dim": 1,
     "purpose": "65 groups over one core: the last tile carries 1 of 16 groups",
     "parameters": {"variant": "signed_int4_g16", "rows": 13, "cols": 80, "pattern": "random",
                    "scale": 0.2}},
    {"id": "signed_int4_g16_reuse", "seed": 9024, "block_dim": 1,
     "purpose": "85 groups as 6 tiles over one core, 6 per core: every on-chip buffer and scale slot "
                "is reused from one tile to the next",
     "parameters": {"variant": "signed_int4_g16", "rows": 17, "cols": 80, "pattern": "random",
                    "scale": 8.0}},
    {"id": "signed_int4_g16_idle", "seed": 9025, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "signed_int4_g16", "rows": 1, "cols": 16, "pattern": "random",
                    "scale": 0.02}},

    {"id": "e2m1_g32_minimum", "seed": 9040, "block_dim": 1,
     "purpose": "The smallest legal shape: one 32-element group on one core",
     "parameters": {"variant": "e2m1_g32", "rows": 1, "cols": 32, "pattern": "random",
                    "scale": 1.0}},
    {"id": "e2m1_g32_zero", "seed": 9041, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "e2m1_g32", "rows": 1, "cols": 32, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "e2m1_g32_boundaries", "seed": 9042, "block_dim": 1,
     "purpose": "Every E2M1 threshold, with an exact absmax anchor of 6 in each 32-group, so each "
                "midpoint is hit head on rather than approached",
     "parameters": {"variant": "e2m1_g32", "rows": 2, "cols": 32, "pattern": "boundaries",
                    "scale": 1.0}},
    {"id": "e2m1_g32_source_tail", "seed": 9043, "block_dim": 1,
     "purpose": "65 groups over one core: the last tile carries 1 of 32 groups",
     "parameters": {"variant": "e2m1_g32", "rows": 13, "cols": 160, "pattern": "random",
                    "scale": 0.2}},
    {"id": "e2m1_g32_reuse", "seed": 9044, "block_dim": 1,
     "purpose": "165 groups as 6 tiles over one core, 6 per core: every on-chip buffer and scale "
                "slot is reused from one tile to the next",
     "parameters": {"variant": "e2m1_g32", "rows": 33, "cols": 160, "pattern": "random",
                    "scale": 8.0}},
    {"id": "e2m1_g32_idle", "seed": 9045, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "e2m1_g32", "rows": 1, "cols": 32, "pattern": "random",
                    "scale": 0.02}},

    {"id": "signed_int4_g32_minimum", "seed": 9060, "block_dim": 1,
     "purpose": "The smallest legal shape: one 32-element group on one core",
     "parameters": {"variant": "signed_int4_g32", "rows": 1, "cols": 32, "pattern": "random",
                    "scale": 1.0}},
    {"id": "signed_int4_g32_zero", "seed": 9061, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "signed_int4_g32", "rows": 1, "cols": 32, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "signed_int4_g32_boundaries", "seed": 9062, "block_dim": 1,
     "purpose": "Every half-integer of the signed grid, with an exact absmax anchor of 7 in each "
                "32-group: ties round away from zero and the result clamps to [-8, 7]",
     "parameters": {"variant": "signed_int4_g32", "rows": 2, "cols": 32, "pattern": "boundaries",
                    "scale": 1.0}},
    {"id": "signed_int4_g32_source_tail", "seed": 9063, "block_dim": 1,
     "purpose": "65 groups over one core: the last tile carries 1 of 32 groups",
     "parameters": {"variant": "signed_int4_g32", "rows": 13, "cols": 160, "pattern": "random",
                    "scale": 0.2}},
    {"id": "signed_int4_g32_reuse", "seed": 9064, "block_dim": 1,
     "purpose": "165 groups as 6 tiles over one core, 6 per core: every on-chip buffer and scale "
                "slot is reused from one tile to the next",
     "parameters": {"variant": "signed_int4_g32", "rows": 33, "cols": 160, "pattern": "random",
                    "scale": 8.0}},
    {"id": "signed_int4_g32_idle", "seed": 9065, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "signed_int4_g32", "rows": 1, "cols": 32, "pattern": "random",
                    "scale": 0.02}},

    {"id": "e2m1_g64_minimum", "seed": 9080, "block_dim": 1,
     "purpose": "The smallest legal shape: one 64-element group on one core",
     "parameters": {"variant": "e2m1_g64", "rows": 1, "cols": 64, "pattern": "random",
                    "scale": 1.0}},
    {"id": "e2m1_g64_zero", "seed": 9081, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "e2m1_g64", "rows": 1, "cols": 64, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "e2m1_g64_boundaries", "seed": 9082, "block_dim": 1,
     "purpose": "Every E2M1 threshold, with an exact absmax anchor of 6 in each 64-group, so each "
                "midpoint is hit head on rather than approached",
     "parameters": {"variant": "e2m1_g64", "rows": 2, "cols": 64, "pattern": "boundaries",
                    "scale": 1.0}},
    {"id": "e2m1_g64_source_tail", "seed": 9083, "block_dim": 1,
     "purpose": "65 groups over one core: the last tile carries 1 of 64 groups",
     "parameters": {"variant": "e2m1_g64", "rows": 13, "cols": 320, "pattern": "random",
                    "scale": 0.2}},
    {"id": "e2m1_g64_reuse", "seed": 9084, "block_dim": 1,
     "purpose": "325 groups as 6 tiles over one core, 6 per core: every on-chip buffer and scale "
                "slot is reused from one tile to the next",
     "parameters": {"variant": "e2m1_g64", "rows": 65, "cols": 320, "pattern": "random",
                    "scale": 8.0}},
    {"id": "e2m1_g64_idle", "seed": 9085, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "e2m1_g64", "rows": 1, "cols": 64, "pattern": "random",
                    "scale": 0.02}},

    {"id": "signed_int4_g64_minimum", "seed": 9100, "block_dim": 1,
     "purpose": "The smallest legal shape: one 64-element group on one core",
     "parameters": {"variant": "signed_int4_g64", "rows": 1, "cols": 64, "pattern": "random",
                    "scale": 1.0}},
    {"id": "signed_int4_g64_zero", "seed": 9101, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "signed_int4_g64", "rows": 1, "cols": 64, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "signed_int4_g64_boundaries", "seed": 9102, "block_dim": 1,
     "purpose": "Every half-integer of the signed grid, with an exact absmax anchor of 7 in each "
                "64-group: ties round away from zero and the result clamps to [-8, 7]",
     "parameters": {"variant": "signed_int4_g64", "rows": 2, "cols": 64, "pattern": "boundaries",
                    "scale": 1.0}},
    {"id": "signed_int4_g64_source_tail", "seed": 9103, "block_dim": 1,
     "purpose": "65 groups over one core: the last tile carries 1 of 64 groups",
     "parameters": {"variant": "signed_int4_g64", "rows": 13, "cols": 320, "pattern": "random",
                    "scale": 0.2}},
    {"id": "signed_int4_g64_reuse", "seed": 9104, "block_dim": 1,
     "purpose": "325 groups as 6 tiles over one core, 6 per core: every on-chip buffer and scale "
                "slot is reused from one tile to the next",
     "parameters": {"variant": "signed_int4_g64", "rows": 65, "cols": 320, "pattern": "random",
                    "scale": 8.0}},
    {"id": "signed_int4_g64_idle", "seed": 9105, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "signed_int4_g64", "rows": 1, "cols": 64, "pattern": "random",
                    "scale": 0.02}},

    {"id": "plain_mxfp4_minimum", "seed": 9120, "block_dim": 1,
     "purpose": "The smallest legal shape: one 32-element group on one core",
     "parameters": {"variant": "plain_mxfp4", "rows": 1, "cols": 32, "pattern": "random",
                    "scale": 1.0}},
    {"id": "plain_mxfp4_zero", "seed": 9121, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "plain_mxfp4", "rows": 1, "cols": 32, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "plain_mxfp4_boundaries", "seed": 9122, "block_dim": 1,
     "purpose": "Every E2M1 threshold under a floor-to-power-of-two E8M0 scale instead of a BF16 "
                "absmax/6 scale: the same grid, a different scale policy",
     "parameters": {"variant": "plain_mxfp4", "rows": 3, "cols": 256, "pattern": "boundaries",
                    "scale": 1.0}},
    {"id": "plain_mxfp4_source_reuse", "seed": 9123, "block_dim": 2,
     "purpose": "3852 groups as 121 tiles over 2 cores, 61 per core: every on-chip buffer and scale "
                "slot is reused from one tile to the next",
     "parameters": {"variant": "plain_mxfp4", "rows": 321, "cols": 384, "pattern": "random",
                    "scale": 0.2}},
    {"id": "plain_mxfp4_tail", "seed": 9124, "block_dim": 1,
     "purpose": "72 groups over one core: the last tile carries 8 of 32 groups",
     "parameters": {"variant": "plain_mxfp4", "rows": 9, "cols": 256, "pattern": "random",
                    "scale": 8.0}},
    {"id": "plain_mxfp4_idle", "seed": 9125, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "plain_mxfp4", "rows": 1, "cols": 32, "pattern": "random",
                    "scale": 0.05}},
    {"id": "plain_mxfp4_tiny", "seed": 9126, "block_dim": 1,
     "purpose": "Every element at 2^-126, the smallest FP32 normal: the E8M0 scale sits on its "
                "2^-127 floor and the value must survive the round trip unchanged",
     "parameters": {"variant": "plain_mxfp4", "rows": 1, "cols": 32, "pattern": "tiny",
                    "scale": 1.0}},

    {"id": "mbs_mxfp4_minimum", "seed": 9140, "block_dim": 1,
     "purpose": "The smallest legal shape: one 128-element macro on one core",
     "parameters": {"variant": "mbs_mxfp4", "rows": 1, "cols": 128, "pattern": "random",
                    "scale": 1.0}},
    {"id": "mbs_mxfp4_zero", "seed": 9141, "block_dim": 1,
     "purpose": "One macro of zeros: the macro-factor guard returns 1.0 rather than dividing 6.0 by "
                "an amax of zero, and the inner per-32 guard fires in the same launch",
     "parameters": {"variant": "mbs_mxfp4", "rows": 1, "cols": 128, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "mbs_mxfp4_macro_boundary", "seed": 9142, "block_dim": 1,
     "purpose": "Three rows at the macro-factor boundary: absmax exactly 6.0 (factor 1), absmax at "
                "2^-126, and a 1.5/1.75/2/3 pattern that exercises the top-8 truncation",
     "parameters": {"variant": "mbs_mxfp4", "rows": 3, "cols": 256, "pattern": "macro_boundary",
                    "scale": 1.0}},
    {"id": "mbs_mxfp4_tiny", "seed": 9143, "block_dim": 1,
     "purpose": "Every element at 2^-126: 6/absmax overflows FP32, so the macro factor is built from "
                "the top-8 mantissa bits of an infinity -- the source's behaviour, preserved rather "
                "than repaired",
     "parameters": {"variant": "mbs_mxfp4", "rows": 1, "cols": 128, "pattern": "tiny",
                    "scale": 1.0}},
    {"id": "mbs_mxfp4_source_reuse", "seed": 9144, "block_dim": 2,
     "purpose": "963 macros as 31 tiles over 2 cores, 16 per core: every on-chip buffer and scale "
                "slot is reused from one tile to the next",
     "parameters": {"variant": "mbs_mxfp4", "rows": 321, "cols": 384, "pattern": "random",
                    "scale": 0.2}},
    {"id": "mbs_mxfp4_factor_reuse", "seed": 9145, "block_dim": 1,
     "purpose": "195 macros as 7 tiles on one core: consecutive macros carry different factors "
                "through the same broadcast rows, so a factor held one tile too long shows",
     "parameters": {"variant": "mbs_mxfp4", "rows": 65, "cols": 384, "pattern": "random",
                    "scale": 8.0}},
    {"id": "mbs_mxfp4_idle", "seed": 9146, "block_dim": 3,
     "purpose": "One macro on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "mbs_mxfp4", "rows": 1, "cols": 128, "pattern": "random",
                    "scale": 0.05}},

    {"id": "mxfp8_e5m2_minimum", "seed": 9160, "block_dim": 1,
     "purpose": "The smallest legal shape: one 32-element group on one core",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 1, "cols": 32, "pattern": "random",
                    "scale": 1.0}},
    {"id": "mxfp8_e5m2_zero", "seed": 9161, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 1, "cols": 32, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "mxfp8_e5m2_epsilon", "seed": 9162, "block_dim": 1,
     "purpose": "Eleven E2M1-shaped levels scaled by 2^-64: at that magnitude the preserved "
                "5.421011e-20 source epsilon is the entire step, which an idealized E5M2 would not "
                "do",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 3, "cols": 256, "pattern": "epsilon",
                    "scale": 1.0}},
    {"id": "mxfp8_e5m2_source_reuse", "seed": 9163, "block_dim": 2,
     "purpose": "3852 groups as 121 tiles over 2 cores, 61 per core: every on-chip buffer and scale "
                "slot is reused from one tile to the next",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 321, "cols": 384, "pattern": "random",
                    "scale": 0.2}},
    {"id": "mxfp8_e5m2_tail", "seed": 9164, "block_dim": 1,
     "purpose": "72 groups over one core: the last tile carries 8 of 32 groups, built from the "
                "logspace pattern",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 9, "cols": 256, "pattern": "logspace",
                    "scale": 8.0}},
    {"id": "mxfp8_e5m2_idle", "seed": 9165, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "mxfp8_e5m2", "rows": 1, "cols": 32, "pattern": "random",
                    "scale": 0.05}},

    {"id": "hifx4_minimum", "seed": 9180, "block_dim": 1,
     "purpose": "The smallest legal shape: one 64-element group on one core",
     "parameters": {"variant": "hifx4", "rows": 1, "cols": 64, "pattern": "random",
                    "scale": 1.0}},
    {"id": "hifx4_zero", "seed": 9181, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "hifx4", "rows": 1, "cols": 64, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "hifx4_hierarchy", "seed": 9182, "block_dim": 1,
     "purpose": "A repeating 1,1,1,1,7,7,7,7 pattern with both signs: inside one 64-group the per-8 "
                "and per-4 exponent decisions must disagree, which is exactly what separates the "
                "full three-level HiFX from the host-only level 1 and 2 controls",
     "parameters": {"variant": "hifx4", "rows": 3, "cols": 64, "pattern": "hierarchy",
                    "scale": 1.0}},
    {"id": "hifx4_source_reuse", "seed": 9183, "block_dim": 2,
     "purpose": "384 groups as 16 tiles over 2 cores, 8 per core: every on-chip buffer and scale "
                "slot is reused from one tile to the next",
     "parameters": {"variant": "hifx4", "rows": 96, "cols": 256, "pattern": "random",
                    "scale": 1.0}},
    {"id": "hifx4_source_tail", "seed": 9184, "block_dim": 1,
     "purpose": "185 groups over one core: the last tile carries 17 of 24 groups",
     "parameters": {"variant": "hifx4", "rows": 37, "cols": 320, "pattern": "random",
                    "scale": 0.02}},
    {"id": "hifx4_large_scale", "seed": 9185, "block_dim": 1,
     "purpose": "Generated scale 400, the largest the domain allows, over 125 groups: the per-64 "
                "E6M2 scale runs into its 49152 clamp",
     "parameters": {"variant": "hifx4", "rows": 25, "cols": 320, "pattern": "random",
                    "scale": 400.0}},
    {"id": "hifx4_idle", "seed": 9186, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "hifx4", "rows": 1, "cols": 64, "pattern": "random",
                    "scale": 0.05}},

    {"id": "hifx5_minimum", "seed": 9200, "block_dim": 1,
     "purpose": "The smallest legal shape: one 64-element group on one core",
     "parameters": {"variant": "hifx5", "rows": 1, "cols": 64, "pattern": "random",
                    "scale": 1.0}},
    {"id": "hifx5_zero", "seed": 9201, "block_dim": 1,
     "purpose": "A single group of zeros: the scale guard must publish zeros instead of dividing by "
                "an amax of zero",
     "parameters": {"variant": "hifx5", "rows": 1, "cols": 64, "pattern": "zero",
                    "scale": 1.0}},
    {"id": "hifx5_hierarchy", "seed": 9202, "block_dim": 1,
     "purpose": "A repeating 1,1,1,1,7,7,7,7 pattern with both signs: inside one 64-group the per-8 "
                "and per-4 exponent decisions must disagree, which is exactly what separates the "
                "full three-level HiFX from the host-only level 1 and 2 controls",
     "parameters": {"variant": "hifx5", "rows": 3, "cols": 64, "pattern": "hierarchy",
                    "scale": 1.0}},
    {"id": "hifx5_source_reuse", "seed": 9203, "block_dim": 2,
     "purpose": "384 groups as 16 tiles over 2 cores, 8 per core: every on-chip buffer and scale "
                "slot is reused from one tile to the next",
     "parameters": {"variant": "hifx5", "rows": 96, "cols": 256, "pattern": "random",
                    "scale": 1.0}},
    {"id": "hifx5_source_tail", "seed": 9204, "block_dim": 1,
     "purpose": "185 groups over one core: the last tile carries 17 of 24 groups",
     "parameters": {"variant": "hifx5", "rows": 37, "cols": 320, "pattern": "random",
                    "scale": 0.02}},
    {"id": "hifx5_large_scale", "seed": 9205, "block_dim": 1,
     "purpose": "Generated scale 400, the largest the domain allows, over 125 groups: the per-64 "
                "E6M2 scale runs into its 49152 clamp",
     "parameters": {"variant": "hifx5", "rows": 25, "cols": 320, "pattern": "random",
                    "scale": 400.0}},
    {"id": "hifx5_idle", "seed": 9206, "block_dim": 3,
     "purpose": "One group on three cores: two cores are handed an empty tile range and must publish "
                "nothing",
     "parameters": {"variant": "hifx5", "rows": 1, "cols": 64, "pattern": "random",
                    "scale": 0.05}},
]


def execute(case, inputs, launcher, backend):
    """Build the entry this case names, then launch it. The destination is handed in poisoned
    with NaN and seeded into the launch, so a padded UB column, a guard row, or a group no core
    reached reads back as NaN instead of as a plausible zero."""
    op = OpExec(build_kernel(DEVICE, ENTRY[inputs["variant"]]), launcher=launcher,
                backend=backend, device=DEVICE, block_dim=case["block_dim"],
                out_dir=f"tmp/{launcher}", seed_outputs=True)
    out = torch.full_like(inputs["x"], float("nan"))
    return {"out": op(inputs["x"], out, inputs["rows"], inputs["cols"])}


def compare(name, got, want, tolerance):
    got, want = got.cpu().float(), want.cpu().float()
    if tolerance is None:
        ok, detail = torch.equal(got, want), ""
    else:
        bounds = {k: v for k, v in tolerance.items() if k in ("rtol", "atol")}
        # `atol` alone is not a bound on max_abs_diff: allclose allows atol + rtol*|want| for
        # each element, so report the worst one as a fraction of its own allowance.
        room = (bounds.get("atol", 0.0) + bounds.get("rtol", 0.0) * want.float().abs()).clamp(min=1e-30)
        margin = ((got.float() - want.float()).abs() / room).max().item()
        ok, detail = torch.allclose(got, want, **bounds), f"  allclose={margin:.2f}x ({bounds})"
        if "max_relative_l2" in tolerance:
            norm = torch.linalg.vector_norm(want.flatten())
            residual = torch.linalg.vector_norm((got - want).flatten())
            relative = (residual / norm).item() if norm > 0 else residual.item()
            ok = ok and relative <= tolerance["max_relative_l2"]
            detail += f"  rel_l2={relative:.3e}/{tolerance['max_relative_l2']}"
    worst = (got - want).abs().max().item()
    print(f"    {name:4s} {'ok  ' if ok else 'FAIL'}  max_abs_diff={worst:.3e}{detail}")
    if not ok:
        # Destinations arrive NaN-poisoned, so an element still NaN was never written.
        outside = (got != want) if tolerance is None else ~torch.isclose(got.float(), want.float(), **bounds)
        if outside.any():
            idx = outside.nonzero()
            poison = int((outside & torch.isnan(got.float())).sum())
            note = f", {poison} still NaN-poisoned (never written)" if poison else ""
            print(f"      {len(idx)}/{outside.numel()} elements outside{note}; "
                  f"first {' '.join(str(tuple(i.tolist())) for i in idx[:3])}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launcher", default="sim", choices=LAUNCHERS)
    parser.add_argument("--backend", default="cce", choices=("cce",))
    parser.add_argument("--case", default="all", help="a case id, or 'all'")
    parser.add_argument("--variant", default="all", help="a variant name, or 'all'")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(f"{case['id']:26s} {case['purpose']}")
        return 0

    selected = [case for case in CASES
                if args.case in ("all", case["id"])
                and args.variant in ("all", case["parameters"]["variant"])]
    if not selected:
        parser.error(f"no case named {args.case!r}; --list prints them")
    failed = []
    for case in selected:
        p = case["parameters"]
        print(f"{case['id']}  ({p['rows']}x{p['cols']}, pattern={p['pattern']}, "
              f"launcher={args.launcher}, block_dim={case['block_dim']})")
        inputs = make_inputs(case)
        expected = reference(inputs)
        actual = execute(case, inputs, args.launcher, args.backend)
        for name in expected:
            if not compare(name, actual[name], expected[name], TOLERANCE):
                failed.append(f"{case['id']}/{name}")
    print(f"\n{len(selected) - len({f.split('/')[0] for f in failed})}/{len(selected)} cases passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
