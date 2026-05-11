import torch
import torch.nn as nn


FORMULA = (
    "z[b, n] = x[b, k] @ W^T[k, n] + bias[n]; "
    "z_grouped[b, g, c] = z[b, g*C_g + c] with C_g = hidden_size / num_groups; "
    "mu[b, g] = mean_c(z_grouped[b, g, c]); var[b, g] = var_c(z_grouped[b, g, c]); "
    "n_norm[b, g, c] = (z_grouped[b, g, c] - mu[b, g]) / sqrt(var[b, g] + eps) * gamma[g*C_g + c] + beta[g*C_g + c]; "
    "a[b, n] = LeakyReLU(n_norm_flat[b, n], negative_slope); "
    "out[b, n] = a[b, n] + a[b, n]"
)
DYNAMIC_AXIS = ["B"]


class Model(nn.Module):
    """
    A model that performs a matrix multiplication, group normalization, leaky ReLU activation, and element-wise sum.
    """
    def __init__(self, input_size, hidden_size, num_groups, eps=1e-5, negative_slope=0.01):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_size, eps=eps)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, hidden_size).
        """
        x = self.fc(x)
        x = self.gn(x)
        x = self.leaky_relu(x)
        x = x + x
        return x


batch_size = 128
input_size = 512
hidden_size = 256
num_groups = 8

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_groups]