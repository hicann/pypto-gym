#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn

FORMULA = "w_new, m_new, v_new = adamw_step(weight, grad, m, v, beta1, beta2, lr, wd, eps, step)"
DYNAMIC_AXIS = ["M"]


class Model(nn.Module):
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        step: int = 1,
    ):
        super().__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.step = step

    def forward(
        self,
        weight: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
    ):
        weight_dtype = weight.dtype
        w_f32 = weight.to(torch.float32)
        g_f32 = grad.to(torch.float32)

        bc1 = 1.0 - (self.beta1 ** self.step)
        bc2 = 1.0 - (self.beta2 ** self.step)

        m_new = self.beta1 * m + (1.0 - self.beta1) * g_f32
        v_new = self.beta2 * v + (1.0 - self.beta2) * (g_f32 * g_f32)

        m_hat = m_new / bc1
        v_hat = v_new / bc2

        update = m_hat / (torch.sqrt(v_hat) + self.eps) + self.weight_decay * w_f32
        w_new_f32 = w_f32 - self.lr * update

        return w_new_f32.to(weight_dtype), m_new, v_new


def get_inputs():
    M, K = 7168, 2048
    weight = torch.randn(M, K, dtype=torch.bfloat16) * 0.02
    grad = torch.randn(M, K, dtype=torch.bfloat16) * 0.01
    m = torch.randn(M, K, dtype=torch.float32) * 1e-3
    v = torch.randn(M, K, dtype=torch.float32).abs() * 1e-6
    return [weight, grad, m, v]


def get_init_inputs():
    return [0.9, 0.999, 1e-3, 0.01, 1e-8, 1]
