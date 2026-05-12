import torch
import torch.nn as nn

FORMULA = "out = mean_b(1 - cosine_sim(predictions[b, :], targets[b, :]))"
DYNAMIC_AXIS = ["B"]

class Model(nn.Module):
    """
    A model that computes Cosine Similarity Loss for comparing vectors.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        cosine_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
        return torch.mean(1 - cosine_sim)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []
