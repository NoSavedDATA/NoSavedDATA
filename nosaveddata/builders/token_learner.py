# REFERENCES
# https://github.com/rish-16/tokenlearner-pytorch/blob/main/tokenlearner_pytorch/tokenlearner_pytorch.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_init import init_cnn

class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1,1), stride=1),
            nn.BatchNorm2d(1),
            nn.GELU()
        ).to('cuda')
        
        self.sgap = nn.AvgPool2d(2).to('cuda')
        self.conv.apply(init_cnn)
        
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.contiguous().view(B, C, H, W)
        
        mx = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        combined = torch.cat([mx, avg], dim=1)
        fmap = self.conv(combined)
        weight_map = torch.sigmoid(fmap) * x
        out = (weight_map).mean(dim=(-2, -1))
        
        return out, weight_map


class TokenLearner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)]).to('cuda')
        
    def forward(self, x):
        B, _, _, C = x.shape
        Z = torch.Tensor(B, self.S, C).to(x.device)
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x) # [B, C]
            Z[:, i, :] = Ai
        return Z


class TokenFuser(nn.Module):
    def __init__(self, H, W, C, S) -> None:
        super().__init__()
        self.projection = nn.Linear(S, S, bias=False)
        self.Bi = nn.Linear(C, S)
        self.spatial_attn = SpatialAttention()
        self.S = S
        
    def forward(self, y, x):
        B, S, C = y.shape
        B, H, W, C = x.shape
        
        Y = self.projection(y.view(B, C, S)).view(B, S, C)
        Bw = torch.sigmoid(self.Bi(x)).view(B, H*W, S) # [B, HW, S]
        BwY = torch.matmul(Bw, Y)
        
        _, xj = self.spatial_attn(x)
        xj = xj.view(B, H*W, C)
        
        out = (BwY + xj).view(B, H, W, C)
        
        return out 