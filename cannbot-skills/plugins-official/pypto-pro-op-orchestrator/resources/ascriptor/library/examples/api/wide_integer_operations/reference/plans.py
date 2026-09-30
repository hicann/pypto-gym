# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pure metadata for typed entry families and separately observable result names."""

PLANS = [
    ('native', 'int64_safe_kernel', ['add', 'sub', 'mul', 'and', 'or', 'xor', 'not'], ['native_a', 'native_b'], 'i64'),
    ('native', 'int64_shift_kernel', ['shift_left', 'shift_right'], ['native_a'], 'i64'),
    ('native', 'int64_dup_kernel', ['dup'], ['native_a'], 'i64'),
    ('native', 'int64_cmpsel_kernel', ['select_reg', 'select_scalar'], ['native_a', 'native_b'], 'i64'),
    ('native', 'int64_fused_kernel', ['abssub', 'square_plus_b'], ['native_a', 'native_b'], 'i64'),
    ('extended', 'int64_elt_kernel', ['min', 'max', 'neg', 'abs', 'adds', 'muls', 'maxs', 'mins', 'copy', 'axpy'], ['extended_a', 'extended_b'], 'i64'),
    ('extended', 'int64_reduce_kernel', ['cadd', 'cmax', 'cmin'], ['extended_a'], 'i64'),
    ('unsigned', 'u64_core_kernel', ['add', 'sub', 'mul', 'and', 'or', 'xor', 'not', 'shift_left', 'shift_right', 'dup', 'square_plus_b'], ['unsigned_a', 'unsigned_b'], 'u64'),
    ('unsigned', 'u64_absolute_kernel', ['abssub'], ['unsigned_a', 'unsigned_b'], 'u64'),
    ('unsigned', 'u64_cmpsel_kernel', ['select_reg', 'select_scalar'], ['unsigned_a', 'unsigned_b'], 'u64'),
    ('shifts', 'shiftl_i32_kernel', ['left'], ['shift32_data', 'shift32_count'], 'i32'),
    ('shifts', 'shiftr_i32_kernel', ['right'], ['shift32_data', 'shift32_count'], 'i32'),
    ('shifts', 'shiftl_i64_kernel', ['left'], ['shift64_data', 'shift64_count'], 'i64'),
    ('shifts', 'shiftr_i64_kernel', ['right'], ['shift64_data', 'shift64_count'], 'i64'),
]


def output_name(module, dtype, name):
    return ('variable_' + dtype if module == 'shifts' else module) + '_' + name
