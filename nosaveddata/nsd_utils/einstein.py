import torch
import torch.nn as nn
import einops

class Rearrange(nn.Module):
    def __init__(self, pattern):
        super(Rearrange, self).__init__()
        self.pattern = pattern

    def forward(self, x):
        return einops.rearrange(x, self.pattern)