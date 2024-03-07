import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_init import *
from .mlp import MLP
from .transformer import  GPT_Transformer
from ..nsd_utils.networks import params_count


import numpy as np
import random


class ViT(nn.Module):
    def __init__(self, d_model, num_blks, nhead, patches=(16,16), img_size=(96,72), first_channel=3,
                 dropout = 0.1, bias=False, report_params_count=True,
                 ffn_mult=4):
        super().__init__()
        self.num_hiddens = d_model
        self.first_channel=first_channel
        
        self.patches = np.prod(patches)
        self.N = int(np.prod(img_size)/self.patches)
        
        self.in_proj = MLP(first_channel*self.patches, out_hiddens=d_model, last_init=init_xavier)
        
        self.transformer = GPT_Transformer(d_model, num_blks, nhead, seq_len=self.N,
                 dropout = dropout, bias=bias, report_params_count=report_params_count,
                 ffn_mult=ffn_mult)
        
        
        params_count(self, 'ViT Encoder')
    
    def patchify(self, X):
        X = X.view(-1, self.patches*self.first_channel, self.N).transpose(-2,-1)
        return X
    
    def proj(self, X):
        X = self.patchify(X)
        return self.in_proj(X)
        
    def forward(self, X):
        X = self.patchify(X)
        X = self.in_proj(X)
        
        X = self.transformer(X, is_causal=False)
        
        return X
    
    
    
class ViT_Temporal(nn.Module):
    def __init__(self, d_model, num_blks, temporal_aggr_num_blks, nhead,
                 patches=(16,16), img_size=(96,72), first_channel=3, stacked_frames=4,
                 dropout = 0.1, bias=False, report_params_count=True,
                 ffn_mult=4):
        super().__init__()
        self.num_hiddens = d_model
        self.first_channel=first_channel
        self.stacked_frames=stacked_frames
        
        self.patches = np.prod(patches)
        self.N = int(np.prod(img_size)/self.patches)
        
        self.in_proj = MLP(first_channel*self.patches, out_hiddens=d_model, last_init=init_xavier)
        
        self.transformer = GPT_Transformer(d_model, num_blks, nhead, seq_len=self.N,
                 dropout = dropout, bias=bias, report_params_count=report_params_count,
                 ffn_mult=ffn_mult)
        
        self.temporal_aggr = GPT_Transformer(d_model, temporal_aggr_num_blks, nhead, seq_len=self.N*stacked_frames,
                 dropout = dropout, bias=bias, report_params_count=report_params_count,
                 ffn_mult=ffn_mult)
        
        
        params_count(self, 'ViT Temporal')
    
    def patchify(self, X):
        X = X.view(X.shape[0]*self.stacked_frames, -1, *X.shape[-2:])
        X = X.view(-1, self.patches*self.first_channel, self.N).transpose(-2,-1)
        return X
    
    def proj(self, X):
        X = self.patchify(X)
        return self.in_proj(X)
    
    def transformers(self, X):
        
        X = self.transformer(X, is_causal=False).view(-1, self.stacked_frames*self.N, self.num_hiddens)
        X = self.temporal_aggr(X, is_causal=False)
        
        return X[:,-self.N:]
    
    def forward(self, X):
        X = self.proj(X)
        X = self.transformers(X)
        
        return X

    
    
class ViT_IWM(nn.Module):
    def __init__(self, encoder, d_encoder,
                 d_predictor, num_blks_predictor, nhead_predictor,
                 out_dim=2048,
                 patches=(16,16), img_size=(96,72),
                 stacked_frames=4,
                 masked_tokens=4,
                 num_augmentations=2,
                 dropout = 0.1, bias=False, report_params_count=True,
                 ffn_mult=4):
        super().__init__()
        self.d_encoder = d_encoder
        self.stacked_frames=stacked_frames
        
        self.patches = np.prod(patches)
        self.N = int(np.prod(img_size)/self.patches)
        self.masked_tokens=self.N//masked_tokens
        
        self.encoder = encoder
        
        self.predictor_proj = MLP(d_encoder, out_hiddens=d_predictor, last_init=init_xavier) \
                              if d_predictor!=d_encoder else nn.Identity()
        
        self.predictor = GPT_Transformer(d_predictor, num_blks_predictor, nhead_predictor, seq_len=self.N,
                 dropout = dropout, bias=bias, report_params_count=report_params_count,
                 ffn_mult=ffn_mult)
        
        self.mask = MLP(1, out_hiddens=d_encoder, last_init=init_xavier)
        self.mask_pos_encoding = nn.Embedding(self.N, d_encoder)
        self.mask_mlp = MLP(d_encoder+num_augmentations, d_encoder, d_encoder, layers=4, in_act=nn.ReLU(), out_act=nn.ReLU(),
                            init=init_relu, last_init=init_relu)
        
        
        params_count(self, 'IWM')
    
    def get_mask(self, X, augmentations):
        B, T, D = X.shape
        B = B//self.stacked_frames
        m_rand = random.randint(0,self.masked_tokens*2)
        
        mask_pos = torch.randint(0, T, (B,self.masked_tokens+m_rand), device='cuda')
        mask_pos_repeat = mask_pos.repeat_interleave(self.stacked_frames,0)
        
        X_mask_pos = (mask_pos_repeat + torch.arange(B, device='cuda').repeat_interleave(self.stacked_frames,0)[:,None]*B).view(-1)
        
        
        mask = self.mask(torch.ones(B*self.stacked_frames,self.masked_tokens+m_rand,1, device='cuda'))
        
        mask = mask + self.mask_pos_encoding(mask_pos_repeat)
        augmentations = augmentations.repeat_interleave(self.stacked_frames,0)[:,None].expand(-1,mask.shape[1],-1)
        
        mask = self.mask_mlp(torch.cat((mask,augmentations),-1))
        
        
        X.view(-1,D)[X_mask_pos]=X.view(-1,D)[X_mask_pos]*0+mask.view(-1,D)
        
        mask_pos = mask_pos[:,:self.masked_tokens,None].expand(-1,-1,X.shape[-1])
        
        
        return X, mask_pos
    
    def encode(self, X):
        return self.encoder(X)

    
    def forward(self, X, y, augmentations):
        X = self.encoder.proj(X)
        X_masked, mask_pos = self.get_mask(X, augmentations)
        X = self.encoder.transformers(X_masked)
        
        X = self.predictor_proj(X)
        
        X = self.predictor(X)
        mask_pos = mask_pos.contiguous().view(X.shape[0], -1, X.shape[-1])
        
        return X.gather(1,mask_pos), y.gather(1,mask_pos)