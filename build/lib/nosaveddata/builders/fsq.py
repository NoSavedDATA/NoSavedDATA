import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import nosaveddata as nsd

from ..nsd_utils.save_hypers import nsd_Module

class FSQ(nsd_Module):
    def __init__(self, L):
        super().__init__()
        self.eps = 1e-3

        self.L = torch.tensor(L).cuda()
        self.basis = torch.cat((torch.tensor([1],device=self.L.device), (self.L[:-1]).cumprod(-1)),-1).long()
        self.compute_Lvalues()


        self.codebooks_size = self.L.prod(-1)
        self.implicit_codebooks = self.indexes_to_codes(torch.arange(self.codebooks_size,device=self.L.device))
        print(f"Codebooks size: {self.codebooks_size}")

    def compute_Lvalues(self):
        self.half_l = (self.L-1) * (1-self.eps) / 2
        self.half_width = self.L // 2
        self.offset = torch.where(self.L%2==1,0.0,0.5)
        self.shift = torch.tan(self.offset / self.half_l)

    def round_ste(self, x):
        rounded = x.round()
        return x + (rounded-x).detach()

    def bound(self,x):
        return F.tanh(x+self.shift) * self.half_l - self.offset

    def _scale_and_shift(self, x):
        return (x*self.half_width) + self.half_width
    
    def _scale_and_shift_inverse(self, x):
        return (x - self.half_width) / self.half_width

    def codes_to_indexes(self, x):
        x = self._scale_and_shift(x)
        # print(f"codes to indexes {x, self.basis, x*self.basis}")
        
        return (x*self.basis).sum(dim=-1).round().long()

    def indexes_to_codes(self, indices):

        indices = indices[...,None]
        # print(f"{indices.shape, self.basis.shape, self.L.shape}")
        codes_non_centered = torch.remainder(torch.floor_divide(indices,self.basis), self.L)
        return self._scale_and_shift_inverse(codes_non_centered)

    def forward(self, x):
        quantized = self.round_ste(self.bound(x)) / self.half_width
        return quantized, self.codes_to_indexes(quantized), self.indexes_to_codes(self.codes_to_indexes(quantized)) 



    
