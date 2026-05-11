import torch
import torch.nn as nn


FORMULA = (
    "z[b, n] = x[b, k] @ W^T[k, n] + bias[n]; "
    "z_grouped[b, g, c] = z[b, g*C_g + c] with C_g = out_features / num_groups; "
    "mu[b, g] = mean_c(z_grouped[b, g, c]); var[b, g] = var_c(z_grouped[b, g, c]); "
    "n_norm[b, g, c] = (z_grouped[b, g, c] - mu[b, g]) / sqrt(var[b, g] + eps) * gamma[g*C_g + c] + beta[g*C_g + c]; "
    "s1[b, n] = sigmoid(n_norm_flat[b, n]) * n_norm_flat[b, n]; "
    "m[b, n] = s1[b, n] * multiply_weight[n]; "
    "out[b, n] = sigmoid(m[b, n]) * m[b, n]"
)
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape)) 

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.gemm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.group_norm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * self.multiply_weight
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        return x

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 16
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]