import torch
from torch import nn
import torch.nn.functional as F

import math

from ..nsd_utils.networks import params_count
from ..nsd_utils.save_hypers import nsd_Module
from .weight_init import *
from .mlp import MLP






class SoftMoE(nsd_Module):
    def __init__(self, hiddens, num_experts=8, num_slots=1, act=nn.SiLU()):
        super().__init__()

        self.slots = MLP(hiddens, out_hiddens=num_experts*num_slots)

        self.experts = nn.ModuleList([MLP(num_slots*hiddens, out_hiddens=num_slots*hiddens, out_act=act) for _ in range(num_experts)])
        self.apply(init_gpt)

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

        self.experts = nn.ModuleList([MLP(num_slots*hiddens, out_hiddens=num_slots*hiddens, out_act=act, last_init=init_relu) for _ in range(num_experts)])
        self.expert_projection = MLP(hiddens, out_hiddens=projected_dim)
        self.apply(init_gpt)

    
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
        

        self.experts = nn.ModuleList([MLP(num_slots*hiddens, out_hiddens=num_slots*hiddens, out_act=act, last_init=init_gpt) for _ in range(num_experts)])
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





