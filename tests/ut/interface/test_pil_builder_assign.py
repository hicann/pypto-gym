#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 CANN community contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from test_pil_builder_utils import TestParser, Expr


def test_pil_builder_expr():

    with TestParser():

        # --- expr statement: binop (result discarded) ---

        @TestParser.test
        def expr_binop():
            Expr.int(0) + Expr.int(1)

        # --- expr statement: named expr (side-effectful, target assigned) ---

        @TestParser.test
        def expr_named_expr():
            (var_a := Expr.int(0))
            Expr.str(var_a)



def test_pil_builder_named_expr():

    with TestParser():

        # --- named expr in binop ---

        @TestParser.test
        def named_expr_binop():
            var_x = (var_a := Expr.int(0)) + (var_b := Expr.int(1))
            Expr.str(var_x)
            Expr.str(var_a)
            Expr.str(var_b)

        # --- named expr in unary op ---

        @TestParser.test
        def named_expr_unary():
            var_x = -(var_a := Expr.int(5))
            Expr.str(var_x)
            Expr.str(var_a)

        # --- named expr as if test ---

        @TestParser.test
        def named_expr_if_true():
            if var_a := Expr.true(0):
                Expr.str(1)
            else:
                Expr.str(2)
            Expr.str(var_a)

        @TestParser.test
        def named_expr_if_false():
            if var_a := Expr.false(0):
                Expr.str(1)
            else:
                Expr.str(2)
            Expr.str(var_a)

        # --- named expr as for iter ---

        @TestParser.test
        def named_expr_for_iter():
            for var_x in (var_it := [Expr.int(0), Expr.int(1)]):
                Expr.str(var_x)
            Expr.str(len(var_it))

        # --- named expr as while test ---

        @TestParser.test
        def named_expr_while():
            var_items = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_i = [0]
            while var_n := (var_i[0] < len(var_items)):
                Expr.str(var_items[var_i[0]])
                var_i[0] = var_i[0] + 1
            Expr.str(var_n)

        # --- named expr as call positional arg ---

        @TestParser.test
        def named_expr_call_pos_arg():
            Expr.str(var_a := Expr.int(0))
            Expr.str(var_a)

        # --- named expr as call keyword arg ---

        @TestParser.test
        def named_expr_call_kw_arg():

            def func(x):
                Expr.str(x)
            func(x=(var_a := Expr.int(0)))
            Expr.str(var_a)

        # --- named expr in tuple literal ---

        @TestParser.test
        def named_expr_in_tuple():
            var_t = ((var_a := Expr.int(0)), (var_b := Expr.int(1)))
            Expr.str(var_t[0])
            Expr.str(var_t[1])
            Expr.str(var_a)
            Expr.str(var_b)

        # --- named expr in list literal ---

        @TestParser.test
        def named_expr_in_list():
            var_l = [(var_a := Expr.int(0)), (var_b := Expr.int(1))]
            Expr.str(var_l[0])
            Expr.str(var_l[1])
            Expr.str(var_a)
            Expr.str(var_b)

        # --- named expr as dict key and value ---

        @TestParser.test
        def named_expr_dict_key():
            var_d = {(var_k := Expr.int(0)): Expr.int(1)}
            Expr.str(var_d[0])
            Expr.str(var_k)

        @TestParser.test
        def named_expr_dict_value():
            var_d = {Expr.int(0): (var_v := Expr.int(99))}
            Expr.str(var_d[0])
            Expr.str(var_v)

        # --- named expr in set literal ---

        @TestParser.test
        def named_expr_in_set():
            var_s = {(var_a := Expr.int(1)), (var_b := Expr.int(2))}
            Expr.str(1 in var_s)
            Expr.str(var_a)
            Expr.str(var_b)

        # --- named expr as subscript index ---

        @TestParser.test
        def named_expr_subscript_index():
            var_arr = Expr(0)
            var_arr[0] = Expr.int(99)
            var_x = var_arr[(var_i := Expr.int(0))]
            Expr.str(var_x)
            Expr.str(var_i)

        # --- named expr as slice bound ---

        @TestParser.test
        def named_expr_slice_bound():
            var_l = [Expr.int(0), Expr.int(1), Expr.int(2)]
            var_s = var_l[(var_lo := Expr.int(1)):(var_hi := Expr.int(3))]
            Expr.str(var_s[0])
            Expr.str(var_lo)
            Expr.str(var_hi)

        # --- named expr as type annotation value ---

        @TestParser.test
        def named_expr_annotation_value():
            var_x: int = (var_a := Expr.int(0))
            Expr.str(var_x)
            Expr.str(var_a)

        # --- named expr as type annotation's annotation expression ---

        @TestParser.test
        def named_expr_as_annotation():
            var_x: (var_ann := Expr.str(0))
            Expr.str(var_ann)

        @TestParser.test
        def named_expr_as_annotation_with_value():
            var_x: (var_ann := Expr.str(0)) = Expr.int(1)
            Expr.str(var_x)
            Expr.str(var_ann)


def test_pil_builder_assign():

    with TestParser():

        # --- name target ---

        @TestParser.test
        def assign_name():
            var_x = Expr.int(0)

        @TestParser.test
        def assign_name_rhs_call():
            var_x = Expr.int(0) + Expr.int(1)

        @TestParser.test
        def assign_multi_target():
            # e.g. a = b = expr - both names get the same value
            var_x = var_y = Expr.int(0)

        # --- attribute target ---

        @TestParser.test
        def assign_attr():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)

        @TestParser.test
        def assign_attr_rhs_call():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0) + Expr.int(1)

        @TestParser.test
        def assign_attr_chain():
            # e.g. obj.val.val += rhs - chain of attribute loads
            var_obj = Expr(0)
            var_obj.val = Expr(1)
            var_obj.val.val = Expr.int(0)

        # --- subscript target ---

        @TestParser.test
        def assign_subscript_const_index():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(1)

        @TestParser.test
        def assign_subscript_expr_index():
            var_obj = Expr(0)
            var_obj[Expr.int(0)] = Expr.int(1)

        @TestParser.test
        def assign_subscript_attr_index():
            # e.g. obj[other.val] = rhs - index is an attribute load
            var_obj = Expr(0)
            var_idx = Expr(1)
            var_idx.val = Expr.int(0)
            var_obj[var_idx.val] = Expr.int(1)

        @TestParser.test
        def assign_subscript_subscript_index():
            # e.g. obj[idx[k]] = rhs - index is itself a subscript
            var_obj = Expr(0)
            var_idx = Expr(1)
            var_idx[0] = Expr.int(0)
            var_obj[var_idx[0]] = Expr.int(1)

        @TestParser.test
        def assign_subscript_binop_index():
            # e.g. obj[a + b] = rhs - index is a binop
            var_obj = Expr(0)
            var_obj[Expr.int(0) + Expr.int(1)] = Expr.int(2)

        @TestParser.test
        def assign_subscript_slice():
            var_obj = Expr(0)
            var_obj[0:2] = Expr.int(1)

        @TestParser.test
        def assign_subscript_slice_with_step():
            var_obj = Expr(0)
            var_obj[0:4:2] = Expr.int(1)

        @TestParser.test
        def assign_subscript_expr_slice():
            # slice bounds are side-effectful expressions
            var_obj = Expr(0)
            var_obj[Expr.int(0):Expr.int(1)] = Expr.int(2)

        @TestParser.test
        def assign_subscript_attr_slice():
            # e.g. obj[a.val:b.val] = rhs - slice bounds are attribute loads
            var_obj = Expr(0)
            var_lo = Expr(1)
            var_lo.val = Expr.int(0)
            var_hi = Expr(2)
            var_hi.val = Expr.int(2)
            var_obj[var_lo.val:var_hi.val] = Expr.int(3)

        @TestParser.test
        def assign_subscript_subscript_slice():
            # e.g. obj[lo[0]:hi[0]] = rhs - slice bounds are subscripts
            var_obj = Expr(0)
            var_lo = Expr(1)
            var_lo[0] = Expr.int(0)
            var_hi = Expr(2)
            var_hi[0] = Expr.int(2)
            var_obj[var_lo[0]:var_hi[0]] = Expr.int(3)

        @TestParser.test
        def assign_subscript_binop_slice():
            # e.g. obj[a+1 : b*2] = rhs - slice bounds are binops
            var_obj = Expr(0)
            var_obj[Expr.int(0) + 1: Expr.int(1) * 2] = Expr.int(2)

        # --- nested subscript / attr chains ---

        @TestParser.test
        def assign_attr_subscript():
            # e.g. obj.val[k] = rhs - subscript index is an attribute load
            var_obj = Expr(0)
            var_obj.val = Expr(1)
            var_obj.val[0] = Expr.int(1)

        @TestParser.test
        def assign_subscript_attr():
            # e.g. obj[k].val = rhs - subscript index is a subscript
            var_obj = Expr(0)
            var_obj[0] = Expr(1)
            var_obj[0].val = Expr.int(1)

        @TestParser.test
        def assign_subscript_attr_subscript_attr():
            # e.g. obj[k].val[k].val = rhs - assignment through a four-level access chain
            var_obj = Expr(0)
            var_obj[0] = Expr(1)
            var_obj[0].val = Expr(2)
            var_obj[0].val[0] = Expr(3)
            var_obj[0].val[0].val = Expr.int(1)

        # --- tuple / list unpack ---

        @TestParser.test
        def assign_tuple_unpack():
            var_x, var_y = Expr.int(0), Expr.int(1)

        @TestParser.test
        def assign_list_unpack():
            [var_x, var_y] = [Expr.int(0), Expr.int(1)]

        @TestParser.test
        def assign_starred_unpack():
            var_x, *var_y, var_z = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]

        @TestParser.test
        def assign_nested_tuple_unpack():
            (var_x, (var_y, var_z)) = (Expr.int(0), (Expr.int(1), Expr.int(2)))

        @TestParser.test
        def assign_unpack_to_attr_subscript():
            # lhs elements can be attribute / subscript targets
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_arr = Expr(1)
            var_arr[0] = Expr.int(0)
            var_obj.val, var_arr[0] = Expr.int(1), Expr.int(2)

        # --- chained assignment ---


        @TestParser.test
        def assign_chain_name_name():
            # e.g. x = y = expr - both names bound to same value
            var_x = var_y = Expr.int(0)

        @TestParser.test
        def assign_chain_name_attr():
            # e.g. x = obj.val = expr - name bound to attribute load of object
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_x = var_obj.val = Expr.int(1)

        @TestParser.test
        def assign_chain_name_subscript():
            # e.g. x = obj[k] = expr - name bound to subscript of object
            var_obj = Expr(0)
            var_obj[0] = Expr.int(0)
            var_x = var_obj[0] = Expr.int(1)

        @TestParser.test
        def assign_chain_attr_subscript():
            # e.g. obj.val = arr[k] = expr - attribute load of object bound to subscript of array target
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_arr = Expr(1)
            var_arr[0] = Expr.int(0)
            var_obj.val = var_arr[0] = Expr.int(1)

        @TestParser.test
        def assign_chain_three():
            # e.g. x = obj.val = arr[k] = expr - three targets bound to same value
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_arr = Expr(1)
            var_arr[0] = Expr.int(0)
            var_x = var_obj.val = var_arr[0] = Expr.int(1)

        @TestParser.test
        def assign_chain_tuple_name():
            # e.g. (a, b) = x = expr - tuple elements bound to names
            var_x = var_a, var_b = Expr.int(0), Expr.int(1)

        @TestParser.test
        def assign_chain_tuple_tuple():
            # e.g. (a, b) = (c, d) = expr - two tuple lhs targets
            var_a, var_b = var_c, var_d = Expr.int(0), Expr.int(1)

        @TestParser.test
        def assign_chain_list_list():
            # e.g. [a, b] = [c, d] = expr: list elements bound to names
            [var_a, var_b] = [var_c, var_d] = [Expr.int(0), Expr.int(1)]

        @TestParser.test
        def assign_chain_tuple_nested_2():
            # e.g. (a, (b, c)) = x = expr - 2-level nested tuple on first target
            var_x = var_a, (var_b, var_c) = Expr.int(0), (Expr.int(1), Expr.int(2))

        @TestParser.test
        def assign_chain_tuple_nested_3():
            # e.g. x = (a, (b, (c, d))) = expr - 3-level nested tuple
            var_x = var_a, (var_b, (var_c, var_d)) = \
                Expr.int(0), (Expr.int(1), (Expr.int(2), Expr.int(3)))

        @TestParser.test
        def assign_chain_list_nested_3():
            # e.g. x = [a, [b, [c, d]]] = expr - 3-level nested list
            var_x = [var_a, [var_b, [var_c, var_d]]] = \
                [Expr.int(0), [Expr.int(1), [Expr.int(2), Expr.int(3)]]]

        @TestParser.test
        def assign_chain_mixed_nested_3():
            # e.g. x = (a, [b, (c, d)]) = expr - mixed tuple/list 3-level
            var_x = var_a, [var_b, (var_c, var_d)] = \
                Expr.int(0), [Expr.int(1), (Expr.int(2), Expr.int(3))]

        @TestParser.test
        def assign_chain_starred_nested():
            # e.g. (a, *b, c) = x = expr - chained assignment with a starred unpack target
            var_x = [var_a, var_b, var_c, var_d] = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]
            var_a, *var_rest, var_z = var_x = [Expr.int(0), Expr.int(1), Expr.int(2), Expr.int(3)]

        @TestParser.test
        def assign_chain_three_nested():
            # e.g. (a, b) = [c, d] = x = expr - three targets, two of them are nested
            var_x = [var_c, var_d] = var_a, var_b = [Expr.int(0), Expr.int(1)]


def test_pil_builder_aug_assign():

    with TestParser():

        @TestParser.test
        def aug_assign_name():
            var_x = Expr.int(0)
            var_x += Expr.int(1)

        @TestParser.test
        def aug_assign_name_rhs_call():
            var_x = Expr.int(0)
            var_x += Expr.int(1) * Expr.int(2)

        @TestParser.test
        def aug_assign_attr():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_obj.val += Expr.int(1)

        @TestParser.test
        def aug_assign_attr_rhs_call():
            var_obj = Expr(0)
            var_obj.val = Expr.int(0)
            var_obj.val += Expr.int(1) + Expr.int(2)

        @TestParser.test
        def aug_assign_subscript():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(1)
            var_obj[0] += Expr.int(2)

        @TestParser.test
        def aug_assign_subscript_rhs_call():
            var_obj = Expr(0)
            var_obj[0] = Expr.int(1)
            var_obj[0] += Expr.int(2) * Expr.int(3)

        @TestParser.test
        def aug_assign_subscript_expr_index():
            var_obj = Expr(0)
            var_obj[Expr.int(0)] = Expr.int(1)
            var_obj[Expr.int(0)] += Expr.int(2)

        @TestParser.test
        def aug_assign_nested_attr_subscript():
            var_obj = Expr(0)
            var_obj.val = Expr(1)
            var_obj.val[0] = Expr.int(1)
            var_obj.val[0] += Expr.int(2) + Expr.int(3)

        @TestParser.test
        def aug_assign_subscript_attr_chain():
            # e.g. obj[k].val += rhs - subscript then attribute cases bound
            var_obj = Expr(0)
            var_obj[0] = Expr(1)
            var_obj[0].val = Expr.int(1)
            var_obj[0].val += Expr.int(2)

        @TestParser.test
        def aug_assign_attr_subscript_attr_subscript():
            # e.g. obj.val[k].val[k] += rhs - assignment through a four-level access chain
            var_obj = Expr(0)
            var_obj.val = Expr(1)
            var_obj.val[0] = Expr(2)
            var_obj.val[0].val = Expr(3)
            var_obj.val[0].val[0] = Expr.int(1)
            var_obj.val[0].val[0] += Expr.int(2)

        @TestParser.test
        def aug_assign_subscript_slice():
            # e.g. obj[a:b] += rhs - augmented assignment on a basic slice
            var_obj = Expr(0)
            var_obj[0:1] = Expr.int(1)
            var_obj[0:1] += Expr.int(2)

        @TestParser.test
        def aug_assign_subscript_slice_with_step():
            # e.g. obj[a:b:c] += rhs - augmented assignment on a slice with step
            var_obj = Expr(0)
            var_obj[0:4:2] = Expr.int(1)
            var_obj[0:4:2] += Expr.int(2)


def test_pil_builder_ann_assign():

    with TestParser():

        # --- annotation only (no value): annotation is evaluated for side effects ---

        @TestParser.test
        def ann_assign_only_call_annotation():
            var_x: Expr.str(0)

        @TestParser.test
        def ann_assign_only_const_annotation():
            # constant annotation: no side effect, trace stays empty
            var_x: int

        # --- annotation + name target ---

        @TestParser.test
        def ann_assign_name_call_annotation():
            var_x: Expr.str(0) = Expr.int(1)

        @TestParser.test
        def ann_assign_name_const_annotation():
            var_x: int = Expr.int(0)

        # --- annotation + attribute target ---

        @TestParser.test
        def ann_assign_attr_target():
            var_obj = Expr(0)
            var_obj.val: Expr.str(0) = Expr.int(1)

        # --- annotation + subscript target ---

        @TestParser.test
        def ann_assign_subscript_target():
            var_obj = Expr(0)
            var_obj[0]: Expr.str(0) = Expr.int(1)

        # --- annotation + subscript target: object comes from a call ---

        @TestParser.test
        def ann_assign_subscript_target_obj_from_call():

            def make():
                return Expr(0)
            make()[0]: Expr.str(0) = Expr.int(1)

        # --- annotation + subscript target: slice index comes from a call ---

        @TestParser.test
        def ann_assign_subscript_target_slice_from_call():
            var_obj = Expr(0)
            var_obj[Expr.int(0)]: Expr.str(1) = Expr.int(2)

        # --- annotation + subscript target: both object and slice from calls ---

        @TestParser.test
        def ann_assign_subscript_target_obj_and_slice_from_calls():

            def make():
                return Expr(0)
            make()[Expr.int(0)]: Expr.str(1) = Expr.int(2)

        # --- annotation + subscript target: slice is a range (lower:upper from calls) ---

        @TestParser.test
        def ann_assign_subscript_target_slice_range_from_calls():
            var_obj = Expr(0)
            var_obj[Expr.int(0):Expr.int(1)]: Expr.str(2) = Expr.int(3)

        # --- annotation + subscript target: object from call, slice is a range ---

        @TestParser.test
        def ann_assign_subscript_target_obj_call_slice_range():

            def make():
                return Expr(0)
            make()[Expr.int(0):Expr.int(1)]: Expr.str(2) = Expr.int(3)

        # --- annotation expression is a complex call (binop result) ---

        @TestParser.test
        def ann_assign_binop_annotation():
            var_x: Expr.int(0) + Expr.int(1) = Expr.int(2)


def test_pil_builder_delete():

    with TestParser():

        # --- delete name ---

        @TestParser.test
        def delete_name():
            var_x = Expr.int(0)
            del var_x

        # --- delete attribute ---

        @TestParser.test
        def delete_attribute():
            var_obj = Expr(0)
            var_obj.val = Expr.str(1)
            del var_obj.val

        # --- delete subscript ---

        @TestParser.test
        def delete_subscript():
            var_obj = Expr(0)
            var_obj[Expr.str(1)] = Expr.str(2)
            del var_obj[Expr.str(1)]

        @TestParser.test
        def delete_subscript_slice():
            var_obj = Expr(0)
            var_obj[Expr.int(1):Expr.int(2)] = Expr.str(2)
            del var_obj[Expr.int(1):Expr.int(2)]

        @TestParser.test
        def delete_subscript_tuple():
            var_obj = Expr(0)
            var_obj[Expr.int(1), Expr.int(2)] = Expr.str(2)
            del var_obj[Expr.int(1), Expr.int(2)]

        @TestParser.test
        def delete_subscript_tuple_slice():
            var_obj = Expr(0)
            var_obj[Expr.int(1), Expr.int(2):Expr.int(3)] = Expr.str(2)
            del var_obj[Expr.int(1), Expr.int(2):Expr.int(3)]

        # --- delete tuple (multiple targets in one del) ---

        @TestParser.test
        def delete_tuple():
            var_a = Expr.int(0)
            var_b = Expr.int(1)
            del var_a, var_b

        # --- delete nested tuple/list syntax ---

        @TestParser.test
        def delete_nested_tuple():
            var_a = Expr.int(0)
            var_b = Expr.int(1)
            del (var_a, var_b)

        @TestParser.test
        def delete_nested_list():
            a = Expr.int(0)
            b = Expr.int(1)
            del [a, b]

        # --- delete mixed: name, attribute, subscript in one statement ---

        @TestParser.test
        def delete_mixed():
            var_obj = Expr(0)
            var_obj.val = Expr.str(1)
            var_obj[Expr.str(2)] = Expr.str(3)
            var_x = Expr.int(4)
            del var_x, var_obj.val, var_obj[Expr.str(2)]
