#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DistributedConfig
"""
import os

import torch
import torch.distributed as dist
import torch_npu


class CustomProcessGroup(dist.ProcessGroup):
    """安全访问内部方法"""

    def get_hccl_comm_name(self, rank: int) -> str:
        """获取HCCL通信名称的公共方法"""
        backend = self._get_backend(torch.device('npu'))
        return backend.get_hccl_comm_name(rank)


class DistributedConfig:
    """通信测试配置类"""

    def __init__(
        self,
        world_size: int = 0,
        master_ip: str = "127.0.0.1",
    ):
        """
        初始化多卡测试配置
        Args:
            world_size: 需要的卡数 0表示从环境变量解析
            master_ip: 主节点IP地址
            master_port: 主节点端口
            logical_ranks: hccl申请资源需要逻辑卡数
            physical_device_ids: 实际使用的物理卡
        """
        self.master_ip = master_ip
        self.world_size = world_size
        self._parse_device_list()
        self.logical_ranks = list(range(self.world_size))
        self.master_port = self._calculate_port()

    def init_hccl_comm(self, logical_rank_id: int) -> list[str]:
        physical_device_id = self.get_physical_device_id(logical_rank_id)
        torch_npu.npu.set_device(physical_device_id)
        os.environ['TILE_FWK_DEVICE_ID'] = str(physical_device_id)
        dist.init_process_group(
            backend='hccl',
            rank=logical_rank_id,
            world_size=self.world_size,
            init_method=f'tcp://{self.master_ip}:{self.master_port}',
        )
        group_name = self._get_hccl_comm_name(logical_rank_id)
        return [group_name]

    def get_physical_device_id(self, logical_rank_id: int) -> int:
        """
        根据逻辑rank获取物理设备ID
        如果没有环境变量 返回0-n的映射
        Args:
            logical_rank_id: 逻辑rank
        Returns:
            int: 物理设备ID
        """
        if logical_rank_id >= len(self.physical_device_ids):
            raise ValueError(
                f"Logical rank {logical_rank_id} out of range. "
                f"Available physical devices: {self.physical_device_ids}"
            )
        return self.physical_device_ids[logical_rank_id]

    def _get_hccl_comm_name(self, logical_rank_id: int) -> str:
        """获取HCCL通信名称"""
        group_handle = dist.new_group(backend='hccl', ranks=self.logical_ranks)
        get_backend_method = getattr(group_handle, '_get_backend')
        backend = get_backend_method(torch.device('npu'))
        return backend.get_hccl_comm_name(logical_rank_id)

    def _parse_device_list(self):
        """解析设备列表"""
        device_list_str = os.environ.get("TILE_FWK_DEVICE_ID_LIST", "")

        if device_list_str:
            self.physical_device_ids = [
                int(d.strip()) for d in device_list_str.split(",") if d.strip()
            ]
            self.world_size = len(self.physical_device_ids)
        else:
            self.physical_device_ids = list(range(self.world_size))

    def _calculate_port(self) -> int:
        """
        计算端口号
        策略: 5000 + 物理设备ID列表的第一个设备ID
        例如: 设备[4,5] -> 端口 5004
             设备[0,1] -> 端口 5000
             设备[8,9] -> 端口 5008
        返回:
            int: 计算的端口号
        """
        if not self.physical_device_ids:
            return 50001

        first_device_id = self.physical_device_ids[0]
        port = 5000 + first_device_id
        if port < 1024 or port > 65535:
            return 50001

        return port
