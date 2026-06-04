#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
"""
import abc
import os
import logging
from collections.abc import Iterable
import inspect
import torch
import torch_npu
import numpy as np
from numpy.testing import assert_allclose
import pypto


class TestBuilder(abc.ABC):
    def __init__(self, params: tuple, kernel, kernel_golden, tiling: int):
        super().__init__()

        self.params = params
        self.kernel = kernel
        self.kernel_golden = kernel_golden
        self.tiling = tiling

        self.input_pto_list = []
        self.output_pto_list = []
        self.input_data_list = []
        self.output_data_list = []
        self.tensor_list = []

        self.input_dyn_axes = []
        self.output_dyn_axes = []

        self.rtol_value = 1e-3
        self.atol_value = 1e-3

        self.device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))

    def __call__(self, on_board: bool = True, jit: bool = False):
        self.device_id = int(os.environ.get('TILE_FWK_DEVICE_ID', 0))
        self.run(on_board, jit)

    def set_tol(self, rtol=1e-3, atol=1e-3):
        self.rtol_value = rtol
        self.atol_value = atol


    def set_dyn_axes(self, input_dyn_axes, output_dyn_axes):
        self.input_dyn_axes = input_dyn_axes
        self.output_dyn_axes = output_dyn_axes

    def setup_inputs(self, *args):
        for idx, item in enumerate(args):
            dtype = self.dtype_conversion(str(item.dtype))
            pto_tensor = pypto.tensor(item.shape, dtype, f"PTO_TENSOR_{idx}")
            self.input_pto_list.append(pto_tensor)
            self.input_data_list.append(item)

    def setup_inputs_jit(self, *args):
        input_list = []
        for item in args:
            if isinstance(item, tuple):
                input_list.extend(item)
            else:
                input_list.append(item)
        for item in input_list:
            self.input_pto_list.append(item)

    def get_input_list(self):
        return self.input_pto_list

    def get_output_list(self):
        return self.output_pto_list

    def get_input_data_list(self):
        return self.input_data_list

    def get_output_data_list(self):
        return self.output_data_list

    def init_output(self, goldens):
        for idx, golden in enumerate(goldens):
            dtype = self.dtype_conversion(str(golden.dtype))
            pto_tensor = pypto.tensor(golden.shape, dtype, f"PTO_TENSOR_out_{idx}")
            output_data = torch.zeros(golden.shape, dtype=golden.dtype)
            self.output_pto_list.append(pto_tensor)
            self.output_data_list.append(output_data)

    def init_output_jit(self, goldens):
        for golden in goldens:
            pto_tensor = torch.zeros_like(golden, dtype=golden.dtype, device=f'npu:{self.device_id}')
            self.output_pto_list.append(pto_tensor)

    @abc.abstractmethod
    def get_input_from_param(self):
        pass

    def run_pto(self, kernel, tiling, on_board: bool = True):
        if on_board:
            torch.npu.set_device(self.device_id)

        logging.info("Function compile ...")
        pypto.set_vec_tile_shapes(tiling, tiling)
        with pypto.function("MAIN", *self.input_pto_list, *self.output_pto_list) as rlf:
            for _ in rlf:
                kernel(self.params, *self.input_pto_list, *self.output_pto_list)
            del rlf
        assert all(isinstance(x, pypto.tensor) for x in self.output_pto_list)
        logging.info("Function compile done.")

        if on_board:
            logging.info("Kernel Launch ...")
            pto_input_data = [pypto.from_torch(tensor, f"IN_{idx}")
                                for idx, tensor in enumerate(self.input_data_list)]
            pto_output_data = [pypto.from_torch(tensor, f"OUT_{idx}")
                                for idx, tensor in enumerate(self.output_data_list)]
            pypto.runtime._device_run_once_data_from_host(*pto_input_data, *pto_output_data)
            logging.info("Kernel run finish.")

            result_len = len(self.golden_output)
            for idx in range(result_len):
                assert_allclose(self.golden_output[idx].cpu().flatten().tolist(),
                                self.output_data_list[idx].cpu().flatten().tolist(),
                                rtol=self.rtol_value, atol=self.atol_value)

    def run_pto_jit(self):
        torch.npu.set_device(self.device_id)
        self.inputs = self.get_input_from_param()
        output_count = len(inspect.signature(self.kernel_golden).parameters) - 1 - len(self.inputs)
        goldens = self.kernel_golden(self.params, *self.inputs, *[None] * output_count)
        self.init_output_jit(goldens)
        pto_inputs = self._convert_torch_to_pto(self.input_pto_list, self.input_dyn_axes)
        pto_outputs = self._convert_torch_to_pto(self.output_pto_list, self.output_dyn_axes)
        self.kernel(*pto_inputs, *pto_outputs, self.params)
        torch_npu.npu.synchronize()
        result_len = len(goldens)
        for idx in range(result_len):
            assert_allclose(np.array(self.output_pto_list[idx].cpu().flatten().tolist()),
                            np.array(goldens[idx].flatten().tolist()),
                            rtol=self.rtol_value, atol=self.atol_value)

    def _convert_torch_to_pto(self, tensors, dynamic_axes):
        if len(tensors) == len(dynamic_axes):
            pto_tensors = [pypto.from_torch(
                tensor, dynamic_axis=axis) for tensor, axis in zip(tensors, dynamic_axes)]
        elif len(dynamic_axes) == 0:
            pto_tensors = [pypto.from_torch(tensor) for tensor in tensors]
        else:
            raise RuntimeError("The lengths of tensors and dynamic_axes must be identical.")
        return pto_tensors

    def run(self, on_board: bool = True, jit: bool = False):
        if jit:
            self.run_pto_jit()
            return
        if on_board:
            pypto.runtime._device_init()
        self.inputs = self.get_input_from_param()
        output_count = len(inspect.signature(self.kernel_golden).parameters) - 1 - len(self.inputs)
        self.golden_output = self.torch_convert(self.kernel_golden(self.params, *self.inputs, *[None] * output_count))
        self.init_output(self.golden_output)
        logging.info("PyPTO run is called.")
        self.run_pto(self.kernel, self.tiling, on_board)
        logging.info("PyPTO run finished.")
        if on_board:
            pypto.runtime._device_fini()

    def torch_convert(self, data_tuple: tuple):
        if not isinstance(data_tuple, tuple):
            data_tuple = (data_tuple, )

        def _convert(item):
            if isinstance(item, np.ndarray):
                return torch.from_numpy(item)
            elif torch.is_tensor(item):
                return item
            elif isinstance(item, (int, float, bool)):
                return torch.tensor(item)
            elif isinstance(item, Iterable) and not isinstance(item, str):
                return type(item)(_convert(subitem) for subitem in item)
            else:
                return item

        result = tuple(_convert(item) for item in data_tuple)
        if not isinstance(data_tuple, tuple) and len(result) == 1:
            return result[0]
        return result

    def dtype_conversion(self, str_dtype):
        if str_dtype in ['int4']:
            return pypto.DT_INT4
        elif str_dtype in ['int8', 'torch.int8', 'np.int8']:
            return pypto.DT_INT8
        elif str_dtype in ['int16', 'torch.int16', 'np.int16']:
            return pypto.DT_INT16
        elif str_dtype in ['int32', 'torch.int32', 'np.int32']:
            return pypto.DT_INT32
        elif str_dtype in ['int64', 'torch.int64', 'np.int64']:
            return pypto.DT_INT64
        elif str_dtype in ['float8', 'torch.float8', 'np.float8']:
            return pypto.DT_FP8
        elif str_dtype in ['float16', 'half', 'torch.float16', 'np.float16']:
            return pypto.DT_FP16
        elif str_dtype in ['float32', 'torch.float32', 'np.float32']:
            return pypto.DT_FP32
        elif str_dtype in ['bfloat16', 'torch.bfloat16']:
            return pypto.DT_BF16
        elif str_dtype in ['uint8', 'torch.uint8', 'np.uint8']:
            return pypto.DT_UINT8
        elif str_dtype in ['uint16', 'torch.uint16', 'np.uint16']:
            return pypto.DT_UINT16
        elif str_dtype in ['uint32', 'torch.uint32', 'np.uint32']:
            return pypto.DT_UINT32
        elif str_dtype in ['uint64', 'torch.uint64', 'np.uint64']:
            return pypto.DT_UINT64
        elif str_dtype in ['bool', 'torch.bool', 'np.bool_']:
            return pypto.DT_BOOL
        elif str_dtype in ['torch.double', 'np.double']:
            return pypto.DT_DOUBLE
        else:
            raise ValueError("undefined dtype")
