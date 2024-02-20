# REFERENCES
# https://github.com/explainingai-code/VQVAE-Pytorch/blob/main/model/quantizer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


# TAP Encoder is a Transformer, so the encoder and decoder can be anything. Therefore, I've left only the quantizer here.

class Quantizer1d(nn.Module):
    def __init__(self,
                 num_embeddings=256,
                 dim=512
                 ):
        super(Quantizer1d, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, dim)
    
    def forward(self, x):
        B, T, C = x.shape
        #x = x.permute(0, 2, 3, 1)
        #x = x.reshape(x.size(0), -1, x.size(-1))
        
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1)).view(B,T,C)
        
        
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss' : codebook_loss,
            'commitment_loss' : commmitment_loss
        }
        quant_out = x + (quant_out - x).detach()
        min_encoding_indices = min_encoding_indices.contiguous().view((B,-1))
        return quant_out, quantize_losses, min_encoding_indices
    

class Quantizer2d(nn.Module):
    def __init__(self,
                 num_embeddings=256,
                 dim=512
                 ):
        super(Quantizer2d, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        #x = x.permute(0, 2, 3, 1)
        #x = x.reshape(x.size(0), -1, x.size(-1))
        x = x.contiguous().view(B, C, H*W).transpose(-2,-1)
        
        dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
        min_encoding_indices = torch.argmin(dist, dim=-1)
        
        #
        quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
        x = x.contiguous().view((-1, x.size(-1)))
        commmitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        quantize_losses = {
            'codebook_loss' : codebook_loss,
            'commitment_loss' : commmitment_loss
        }
        quant_out = x + (quant_out - x).detach()
        #quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
        quant_out = quant_out.transpose(-2,-1).contiguous().view(B, C, H, W)
        min_encoding_indices = min_encoding_indices.contiguous().view((-1, quant_out.size(-2), quant_out.size(-1)))
        return quant_out, quantize_losses, min_encoding_indices
    
    def quantize_indices(self, indices):
        return einsum(indices, self.embedding.weight, 'b n h w, n d -> b d h w')