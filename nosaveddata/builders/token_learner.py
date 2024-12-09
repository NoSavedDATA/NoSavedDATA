# REFERENCES

# https://github.com/google-research/scenic/blob/main/scenic/projects/token_learner/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nsd_utils.networks import params_count
from .weight_init import *


class TokenLearner(nn.Module):
    def __init__(self, in_channels, S, layers=3) -> None:
        super().__init__()
        
        self.attention = nn.Sequential(nn.Conv2d(in_channels, S, 3, 1, 1, bias=False),
                                       *[nn.SiLU(), nn.Conv2d(S, S, 3, 1, 1, bias=False)]*(layers),
                                       nn.Sigmoid())
        
        self.attention.apply(init_xavier)
        params_count(self, 'Token Learner')
        
        
    def forward(self, X):
        # Input shape:  (B, C, H, W)
        # Output shape: (B, S, C)
        B, C, _, _ = X.shape
        
        attn_w = self.attention(X).flatten(-2,-1)[...,None]
        
        X = X.view(B, C, -1).transpose(-2,-1)[:,None]
        print(f"attn out: {attn_w.shape}, X {X.shape}")
        X = (X*attn_w).mean(2)
        
        return X



class TokenLearner1D(nn.Module):
    def __init__(self, in_channels, S, layers=3) -> None:
        super().__init__()
        
        self.attention = nn.Sequential(nn.Conv1d(in_channels, S, 5, 1, 2, bias=False),
                                       *[nn.SiLU(), nn.Conv1d(S, S, 5, 1, 2, bias=False)]*(layers),
                                       nn.Sigmoid())
        
        self.attention.apply(init_xavier)
        params_count(self, 'Token Learner')
        
        
    def forward(self, X):
        # Input shape:  (B, T, C)
        # Output shape: (B, S, C)
        B, C, _, _ = X.shape
        
        attn_w = self.attention(X.transpose(-2,-1))[...,None]

        print(f"post attention {attn_w.shape}")
        
        X = X[:,None]
        X = (X*attn_w).mean(2)
        print(f"token learner 1d: {X.shape }")
        return X