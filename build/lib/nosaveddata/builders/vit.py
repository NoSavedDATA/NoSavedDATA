import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_init import *
from .mlp import MLP, MLP_NoDATA
from .transformer import  GPT_Transformer, Transformer_NoDATA
from ..nsd_utils.networks import params_count
from ..nsd_utils.bbf import network_ema
from ..nsd_utils.save_hypers import nsd_Module

import math
import numpy as np
import random



class ViT(nsd_Module):
    def __init__(self, d_model, num_blks, nhead, patches=(16,16), img_size=(96,72), first_channel=3,
                 dropout = 0, bias=True, report_params_count=True,
                 ffn_mult=4, stochastic_depth=1.0):
        super().__init__()

        self.patches = np.prod(patches)
        self.N = int(np.prod(img_size)/self.patches)

        self.in_proj = MLP(first_channel*self.patches, out_hiddens=d_model, last_init=init_gpt)

        self.cls = nn.Embedding(1,d_model)
        self.transformer = Transformer_NoDATA(d_model, num_blks, nhead, seq_len=self.N,
                 dropout = dropout, bias=bias, report_params_count=False,
                 ffn_mult=ffn_mult, stochastic_depth=stochastic_depth)

        self.cls.apply(init_gpt)

        if report_params_count:
            params_count(self, 'ViT')

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

    def masked(self, X, mask):
        
        X = self.transformer.masked(X, mask, is_causal=False)

        return X
    
    
    
class ViT_Temporal(nsd_Module): # Processes X stacked_frames separately, then process the result by a temporal transformer
    def __init__(self, d_model, num_blks, temporal_aggr_num_blks, nhead,
                 patches=(16,16), img_size=(96,72), first_channel=3, stacked_frames=4,
                 dropout = 0., bias=True, report_params_count=True,
                 ffn_mult=4, stochastic_depth=1.0):
        super().__init__()
        
        
        self.patches = np.prod(patches)
        self.N = int(np.prod(img_size)/self.patches)
        
        self.in_proj = MLP(first_channel*self.patches, out_hiddens=d_model, last_init=init_xavier)
        
        self.transformer = Transformer_NoDATA(d_model, num_blks, nhead, seq_len=self.N,
                 dropout = dropout, bias=bias, report_params_count=False,
                 ffn_mult=ffn_mult, scale_init=12, stochastic_depth=stochastic_depth)
        
        self.temporal_aggr = Transformer_NoDATA(d_model, temporal_aggr_num_blks, nhead, seq_len=self.N*stacked_frames,
                 dropout = dropout, bias=bias, report_params_count=False,
                 ffn_mult=ffn_mult, scale_init=12, stochastic_depth=stochastic_depth)
        
        if report_params_count:
            params_count(self, 'ViT Temporal')
    
    def patchify(self, X):
        X = X.contiguous().view(X.shape[0]*self.stacked_frames, -1, *X.shape[-2:])
        
        X = X.view(-1, self.patches*self.first_channel, self.N).transpose(-2,-1)
        return X
    
    def proj(self, X):
        X = self.patchify(X)
        return self.in_proj(X)
    
    def transformers(self, X):
        
        X = self.transformer(X, is_causal=False).view(-1, self.stacked_frames*self.N, self.d_model)
        X = self.temporal_aggr(X, is_causal=False)
        
        return X[:,-self.N:]

    def masked(self, X, mask):
        
        X = self.transformer.masked(X, mask, is_causal=False).view(-1, self.stacked_frames*mask.shape[1], self.d_model)
        X = self.temporal_aggr(X, is_causal=False)
        
        return X[:,-mask.shape[1]:]
    
    def forward(self, X):
        X = self.proj(X)
        X = self.transformers(X)
        
        return X




class ViT_IWM(nsd_Module):
    def __init__(self, encoder,
                 d_predictor, num_blks_predictor, nhead_predictor,
                 stacked_frames=4,
                 mask_samples=4,
                 masked_tokens=4,
                 num_augmentations=3,
                 first_channel=3,
                 dropout = 0, bias=True, report_params_count=True,
                 ffn_mult=4, stochastic_depth=1.0):
        super().__init__()
        
        self.d_encoder = encoder.d_model
        
        
        self.first_channel = encoder.first_channel*stacked_frames
        self.img_size = encoder.img_size
        self.patches = encoder.patches
        self.N = encoder.N
        self.masked_tokens=self.N//masked_tokens

        # Mask
        self.mask = MLP(1, out_hiddens=d_predictor, last_init=init_xavier)
        self.mask_pos_encoding = nn.Embedding(self.N, d_predictor)
        self.mask_mlp = MLP(d_predictor+num_augmentations, d_predictor, d_predictor, layers=4, in_act=nn.ReLU(), out_act=nn.ReLU(),
                            init=init_relu, last_init=init_gpt)
        self.mask_pos_encoding.apply(init_gpt)

        # Encoder
        self.encoder = encoder

        # Predictor
        self.predictor_proj = MLP(self.d_encoder, out_hiddens=d_predictor, last_init=init_gpt) \
                              if d_predictor!=self.d_encoder else nn.Identity()

        self.predictor = Transformer_NoDATA(d_predictor, num_blks_predictor, nhead_predictor, seq_len=self.N+1,
                 dropout = dropout, bias=bias, report_params_count=False,
                 ffn_mult=ffn_mult, scale_init=num_blks_predictor, stochastic_depth=stochastic_depth)


        self.predictor_out_proj = MLP(d_predictor, out_hiddens=self.d_encoder, last_init=init_gpt) \
                              if d_predictor!=self.d_encoder else nn.Identity()

        if report_params_count:
            params_count(self, 'IWM')

    def hard_reset(self, new_network, alpha):
        network_ema(self.encoder, new_network.encoder, alpha)

        network_ema(self.predictor_proj, new_network.predictor_proj, alpha)
        network_ema(self.predictor, new_network.predictor, alpha)

        network_ema(self.mask, new_network.mask, alpha)
        network_ema(self.mask_pos_encoding, new_network.mask_pos_encoding, alpha)
        network_ema(self.mask_mlp, new_network.mask_mlp, alpha)

    def get_random_mask(self, X, augmentations):
        B, T, D = X.shape
        B = B//self.stacked_frames
        m_rand = self.mask_samples*random.randint(0,int(self.masked_tokens*2//self.mask_samples)-1)
        
        
        # Get non-overlapping mask
        mask_pos = torch.arange(T, device='cuda')[None,:].repeat_interleave(B,0).float()
        mask_pos = torch.multinomial(mask_pos, num_samples=self.masked_tokens+m_rand, replacement=False)
        
        mask_pos_repeat = mask_pos.repeat_interleave(self.stacked_frames,0)

        # Get the mask complement
        full_range = torch.arange(T,device='cuda')[None,:].repeat_interleave(B,0)

        complement = torch.zeros_like(full_range, dtype=torch.bool)
        complement.scatter_(1, mask_pos, 1)

        complement = full_range[~complement].view(mask_pos.shape[0], -1)
        

        # Mask mlp for geometric + augmentation informations
        mask = self.mask(torch.ones(B*self.stacked_frames,self.masked_tokens+m_rand,1, device='cuda'))

        mask = mask + self.mask_pos_encoding(mask_pos_repeat)

        augmentations = augmentations.repeat_interleave(self.stacked_frames,0)[:,None].expand(-1,mask.shape[1],-1)

        mask = self.mask_mlp(torch.cat((mask,augmentations),-1))

        # Expand to allow gather
        mask_pos = mask_pos[:,:,None].expand(-1,-1,X.shape[-1])
        complement = complement[:,:,None].expand(-1,-1,X.shape[-1])

        return X, mask_pos, complement, mask
        
    def patchify(self, X):
        X = X.view(-1, self.patches*self.first_channel, self.N).transpose(-2,-1)
        return X
    def get_block_mask(self, batch_size):
        
        all_wins = torch.zeros(self.first_channel,*self.img_size).long()
        
        b_mask, b_complement = [], []
        min_c_len = 999 # for trunked collate
        #min_m=999
        
        for b in range(batch_size):
            wins, complements = [], []
            for m in range(self.mask_samples):
                w,h = self.img_size


                min_ar, max_ar = (0.75, 1.5)
                aspect_ratio = min_ar + random.random() * (max_ar - min_ar)

                h_sample_size = int( (h*(torch.tensor(random.random())*0.05+0.15)) * aspect_ratio)

                w_wins, h_wins = torch.randint(0,h-h_sample_size,(2,)).split(1,0)
                win=all_wins.clone()


                for w_win, h_win in zip(w_wins, h_wins):
                    win[...,w_win:w_win+h_sample_size, h_win:h_win+h_sample_size]=1

                
                win = self.patchify(win.float()).mean(-1)
                
                values, idx = win.sort(descending=True)

                idx = idx[:,:self.N//4]
                
                #min_m = min(min_m, len(values[0].nonzero()))
                wins.append(idx)


            wins = torch.stack(wins).squeeze()


            full_range = torch.arange(win.shape[1])

            complement = torch.zeros_like(full_range, dtype=torch.bool)
            complement.scatter_(0, wins.view(-1).unique(), 1)

            complement = full_range[~complement]
            min_c_len = min(min_c_len, len(complement))
            
            
            b_mask.append(wins)
            b_complement.append(complement)
            
            
        for i in range(len(b_complement)):
            b_complement[i] = b_complement[i][:min_c_len]
        
        b_mask = torch.stack(b_mask).cuda()
        b_complement = torch.stack(b_complement).cuda()
        #print(min_m)
        
        return b_mask, b_complement
    
    def get_mask(self, X, augmentations):
        B = X.shape[0]//self.stacked_frames

        
        mask_pos, complement = self.get_block_mask(B)
        mask_pos = mask_pos.view(B*self.mask_samples,-1)
        
        
        
        mask = self.mask(torch.ones(B*self.mask_samples,1,1, device='cuda'))
        
        mask = mask + self.mask_pos_encoding(mask_pos)
        #augmentations = augmentations.repeat_interleave(self.stacked_frames*self.mask_samples,0)[:,None].expand(-1,mask.shape[1],-1)
        #mask = self.mask_mlp(torch.cat((mask,augmentations),-1))


        mask_pos = mask_pos[...,None].expand(-1,-1,self.d_encoder)
        complement = complement[...,None].expand(-1,-1,self.d_encoder).repeat_interleave(self.stacked_frames,0)
        
        return mask_pos, mask, complement
    
    def encode(self, X):
        return self.encoder(X)


    def forward(self, X, y, augmentations):
        X = self.encoder.proj(X)
        
        mask_pos, mask, complement = self.get_mask(X, augmentations)
        
        X = self.encoder.masked(X, complement)
        X = self.predictor_proj(X)

        X = torch.cat((X.repeat_interleave(4,0),mask),1)
        
        X = self.predictor.no_pos(X)[:,-mask.shape[1]:]
        X = self.predictor_out_proj(X)
        
        return X, y.repeat_interleave(4,0).gather(1,mask_pos)