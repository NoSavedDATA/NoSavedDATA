import torch
from torch import nn
import torch.nn.functional as F

import math

from ..nsd_utils.networks import params_count
from ..nsd_utils.save_hypers import nsd_Module
from .weight_init import *
from .mlp import MLP
from .transformer import Attention, FFN, LayerNormNoBias, GPT_Block






class SoftMoE(nsd_Module):
    def __init__(self, hiddens, num_experts=8, num_slots=1, act=nn.SiLU()):
        super().__init__()

        self.slots = MLP(hiddens, out_hiddens=num_experts*num_slots, last_init=init_lecun)

        self.experts = nn.ModuleList([MLP(num_slots*hiddens, out_hiddens=num_slots*hiddens, out_act=act, last_init=init_relu)]*num_experts)

        params_count(self, 'Soft MoE')
    
    def forward(self, X):
        # Input shape:  (B, T, D)
        # Output shape: (B, T, D)
        B, T, D = X.shape
        
        logits = self.slots(X)
        
        dispatch_w = F.softmax(logits, 1)
        combine_w = F.softmax(logits, 2)
        
        slots = (dispatch_w.transpose(-2,-1)@X).contiguous().view(B, self.num_experts,-1)

        y = torch.stack([f_i(slots[:,i]) for i, f_i in enumerate(self.experts)],1)
        y = y.view(B, -1, D)
        y = combine_w@y

        return y








class SoftMoE_Projection(nsd_Module):
    def __init__(self, hiddens, projected_dim, num_experts=8, num_slots=1, act=nn.SiLU()):
        super().__init__()
        
        self.slots = MLP(hiddens, out_hiddens=num_experts*num_slots, last_init=init_lecun)

        self.experts = nn.ModuleList([MLP(num_slots*hiddens, out_hiddens=num_slots*hiddens, out_act=act, last_init=init_relu)]*num_experts)
        self.expert_projection = MLP(hiddens, out_hiddens=projected_dim, last_init=init_xavier)

    
        params_count(self, 'Soft MoE+Projection')
        
    def forward(self, X):
        # Input shape:  (B, T, D)
        # Output shape: (B, T, D)
        B, T, D = X.shape
        
        logits = self.slots(X)
        
        dispatch_w = F.softmax(logits, 1)
        combine_w = F.softmax(logits, 2)
        
        slots = (dispatch_w.transpose(-2,-1)@X).contiguous().view(B, self.num_experts,-1)

        y = torch.stack([f_i(slots[:,i]) for i, f_i in enumerate(self.experts)],1)
        y = y.view(B, -1, D)
        y = self.expert_projection(y)
        y = combine_w@y

        return y


'''
class SoftMoE_Combine_Output(nn.Module):
    def __init__(self, hiddens, projected_dim, num_experts=8, num_slots=1, act=nn.SiLU()):
        super().__init__()
        self.num_experts = num_experts
        self.num_slots = num_slots

        self.slots = MLP(hiddens, out_hiddens=num_experts*num_slots, last_init=init_lecun)

        self.experts = nn.ModuleList([MLP(num_slots*hiddens, out_hiddens=num_slots*hiddens, out_act=act, last_init=init_relu)]*num_experts)
        
        self.expert_projection = MLP(hiddens, out_hiddens=projected_dim, last_init=init_relu)
        self.out_act = act

        params_count(self, 'Soft MoE')
    
    def forward(self, X):
        # Input shape:  (B, T, D)
        # Output shape: (B, D)
        B, T, D = X.shape
        
        logits = self.slots(X)
        # (B, T, num_experts*num_slots)
        
        dispatch_w = F.softmax(logits, 1)
        combine_w = F.softmax(logits.mean(1), -1)[:,None]
        
        slots = (dispatch_w.transpose(-2,-1)@X).contiguous().view(B, self.num_experts,-1)

        y = torch.stack([f_i(slots[:,i]) for i, f_i in enumerate(self.experts)])
        
        y = y.view(B, -1, D)
        
        y = self.expert_projection(y)
        y = combine_w@y
        
        return self.out_act(y.squeeze())
'''

class SoftMoE_Combine_Output(nsd_Module):
    def __init__(self, hiddens, projected_dim, num_experts=8, num_slots=1, act=nn.SiLU()):
        super().__init__()
        

        self.dispatch_attn_w = MLP(hiddens, out_hiddens=num_experts*num_slots, last_init=init_xavier)

        self.dispatch_w_remove_grads = MLP(hiddens, out_hiddens=num_experts*num_slots)
        #self.dispatch_w_remove_grads.turn_off_grads()
        

        self.experts = nn.ModuleList([MLP(num_slots*hiddens, out_hiddens=num_slots*hiddens, out_act=act, last_init=init_gpt)]*num_experts)
        self.expert_projection = MLP(hiddens, out_hiddens=projected_dim, last_init=init_gpt)

        params_count(self, 'Soft MoE')
    
    def forward(self, X):
        # Input shape:  (B, T, D)
        # Output shape: (B, D)
        B, T, D = X.shape
        
        logits_dispatch = self.dispatch_attn_w(X)
        # (B, T, num_experts*num_slots)
        
        dispatch_w = F.softmax(logits_dispatch, 1)
        combine_w = F.softmax(logits_dispatch.mean(1), -1)[:,None]
        
        slots = (dispatch_w.transpose(-2,-1)@X).contiguous().view(B,self.num_experts,-1)
        
        y = torch.stack([f_i(slots[:,i]) for i, f_i in enumerate(self.experts)],1)
        
        y = y.view(B, -1, D)
        #y = y.view(B, -1, self.projected_dim)
        
        y = self.expert_projection(y)
        y = combine_w@y
        
        return y.squeeze(), combine_w.squeeze()

    def no_weight_grads(self, X):
        # Input shape:  (B, T, D)
        # Output shape: (B, D)
        B, T, D = X.shape

        self.dispatch_w_remove_grads.load_state_dict(self.dispatch_attn_w.state_dict())
        logits_dispatch = self.dispatch_w_remove_grads(X)
        # (B, T, num_experts*num_slots)
        
        dispatch_w = F.softmax(logits_dispatch, 1)
        combine_w = F.softmax(logits_dispatch.mean(1), -1)[:,None]
        
        slots = (dispatch_w.transpose(-2,-1)@X).contiguous().view(B,self.num_experts,-1)

        y = torch.stack([f_i(slots[:,i]) for i, f_i in enumerate(self.experts)],1)

        y = y.view(B, -1, D)
        #y = y.view(B, -1, self.projected_dim)
        
        y = self.expert_projection(y)
        y = combine_w@y
        
        return y.squeeze(), combine_w.squeeze()




# This code is here and not in transformer.py due to circular imports.
class SoftMoE_FFN(nn.Module):
    def __init__(self, hiddens, dropout, bias, ffn_mult, num_experts=8, num_slots=1, act=nn.SiLU()):
        super().__init__()
        self.num_experts = num_experts
        self.num_slots = num_slots

        self.slots = MLP(hiddens, out_hiddens=num_experts*num_slots, last_init=init_lecun)

        self.experts = nn.ModuleList([FFN(num_slots*hiddens, dropout, bias, ffn_mult)]*num_experts)

        params_count(self, 'Soft MoE')
    
    def forward(self, X):
        # Input shape:  (B, T, D)
        # Output shape: (B, T, D)
        B, T, D = X.shape
        
        logits = self.slots(X)
        
        dispatch_w = F.softmax(logits, 1)
        combine_w = F.softmax(logits, 2)
        
        slots = (dispatch_w.transpose(-2,-1)@X).contiguous().view(B, self.num_experts,-1)

        y = torch.stack([f_i(slots[:,i]) for i, f_i in enumerate(self.experts)],1)
        y = y.view(B, -1, D)
        y = combine_w@y

        return y

class GPT_SoftMoE_Block(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, bias=False, ffn_mult=4, num_experts=4, num_slots=1):
        super().__init__()
        self.ln_1 = LayerNormNoBias(d_model, bias=bias)
        self.attn = Attention(d_model, num_heads, bias, dropout)
        self.ln_2 = LayerNormNoBias(d_model, bias=bias)
        self.mlp = SoftMoE_FFN(d_model, dropout, bias, ffn_mult, num_experts=4, num_slots=1, act=nn.GELU())

    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, is_causal=True)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT_SoftMoE(nn.Module):
    def __init__(self, d_model, num_blks, nhead, seq_len,
                 dropout = 0.1, use_bias=False, bias=False, report_params_count=True,
                 ffn_mult=4, num_experts=4, num_slots=1):
        super().__init__()
        self.num_hiddens = d_model

        self.pos_encoding = nn.Sequential(nn.Linear(seq_len, d_model, bias=False),
                                          LayerNormNoBias(d_model)) #Stable Embedding Layer
        
        self.final_ln = LayerNormNoBias(d_model)
        self.start_dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i),
                                 GPT_Block(d_model, nhead, dropout, bias=False, ffn_mult=ffn_mult) if i < (num_blks//2) else \
                                 GPT_SoftMoE_Block(d_model, nhead, dropout, bias=False, ffn_mult=ffn_mult, num_experts=4, num_slots=1))
            
        
        #nn.init.xavier_uniform_(self.pos_encoding[0].weight)
        
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blks))
        
        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'GPT Transformer Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)

        
    def forward(self, X, is_causal=True):

        pos = torch.arange(0, self.seq_len, dtype=torch.float32, device='cuda')
        pos_emb = self.pos_encoding(pos)
        X = self.start_dropout(X+pos_emb)

        for i, blk in enumerate(self.blks):
            X = blk(X)
            
        return self.final_ln(X)