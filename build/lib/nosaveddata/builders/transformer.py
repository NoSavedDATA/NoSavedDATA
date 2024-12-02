# REFERENCES
# https://github.com/karpathy/nanoGPT
# https://github.com/JegZheng/truncated-diffusion-probabilistic-models
# https://github.com/facebookresearch/DiT/blob/main/models.py

import torch
from torch import nn
import torch.nn.functional as F
import math

from .weight_init import *
from ..nsd_utils.save_hypers import nsd_Module


@torch.jit.script # JIT decorator
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

class FusedGELU(nn.Module):
    def forward(self, x):
        return fused_gelu(x)


class LayerNormNoBias(nn.Module):
    """ LayerNormNoBias but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, d_model, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


    
class Attention(nsd_Module):
    def __init__(self, d_model=512, nhead=8, bias=False, dropout=0.1):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        # output projection
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, is_causal):
        B, T, C = q.size()
        
        q = self.W_k(q)
        k = self.W_k(k)
        v = self.W_v(v)
        
        q = q.view(B, T, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, -1, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, -1, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        
        # efficient attention using Flash Attention CUDA kernels
        
        with torch.backends.cuda.sdp_kernel():
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.proj(y))
        return y


class MemoryAttention(nsd_Module):
    def __init__(self, d_model=512, nhead=8, bias=False, dropout=0.1):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.W_kv = nn.Linear(d_model, 2 * d_model, bias=bias)
        # output projection
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, q):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k, v  = self.W_kv(x).split(self.n_embd, dim=2)
        
        
        # FoT LongLlama contrastive style (data pipeline constrastive for self attention enrichment)
        
        shifted_k=[]
        shifted_v=[]
        for i in range(7): # 7 is d-1 for d=8
            shifted_k.append(torch.roll(k[:,:T//2],i,0))
            shifted_v.append(torch.roll(v[:,:T//2],i,0))
        shifted_k=torch.stack(shifted_k).view(B,-1,C)
        shifted_v=torch.stack(shifted_v).view(B,-1,C)
        
        k=torch.concat((shifted_k,k),1)
        v=torch.concat((shifted_v,v),1)
        
        
        
        q = q.view(B, T, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, -1, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, -1, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        
        
        
        L = q.shape[2]
        S = k.shape[2]
        attn_mask = torch.ones(L, S, dtype=torch.bool, device='cuda').tril(diagonal=S-L)
        attn_mask[:T//2,:S-L]=False
        
        
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        
        # efficient attention using Flash Attention CUDA kernels
        with torch.backends.cuda.sdp_kernel():
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
            #y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.proj(y))
        return y

    def forward_memory(self, x, q, k_read, v_read):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        k, v = self.W_kv(x).split(self.n_embd, dim=2)
        write_k, write_v = k.detach(), v.detach()
        
        k=torch.cat((k_read, k), 1)
        v=torch.cat((v_read, v), 1)
        
        #shifted_k=[]
        #shifted_v=[]
        #for i in range(7): # 7 is d-1 for d=8
        #    shifted_k.append(torch.roll(k[:,:T//2],i,0))
        #    shifted_v.append(torch.roll(v[:,:T//2],i,0))
        #shifted_k=torch.stack(shifted_k).view(B,-1,C)
        #shifted_v=torch.stack(shifted_v).view(B,-1,C)
        
        #k=torch.cat((shifted_k, k), 1)
        #v=torch.cat((shifted_v, v), 1)
        
        q = q.view(B, T, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, -1, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, -1, self.nhead, C // self.nhead).transpose(1, 2) # (B, nh, T, hs)
        k_read = k_read.view(B, -1, self.nhead, C // self.nhead).transpose(1, 2)
        v_read = v_read.view(B, -1, self.nhead, C // self.nhead).transpose(1, 2)
          
        
        # Causal Mask
        L = q.shape[2]
        S = k.shape[2]-q.shape[2]
        causal_mask = torch.ones(L, L, dtype=torch.bool, device='cuda').tril(diagonal=0)
        eye_mask=torch.eye(L, dtype=torch.bool, device='cuda')
        read_attnmask=torch.ones(L, L*3, dtype=torch.bool, device='cuda')
        aux=torch.arange(L).repeat_interleave(3)
        
        #new_attnmask=causal_mask[:,aux]
        read_attnmask=eye_mask[:,aux]
        
        attn_mask=torch.cat((read_attnmask,causal_mask),1)
        
        #shift_mask = torch.ones(L, int(L*3.5), dtype=torch.bool, device='cuda')
        #shift_mask[:T//2,:]=False
        #attn_mask=torch.cat((shift_mask,attn_mask),1)
        
        
        # Memory Mask
        memory_mask = torch.ones(L*3, L, dtype=torch.bool, device='cuda')
        memory_mask=torch.concat((torch.eye(L*3, dtype=torch.bool, device='cuda'), memory_mask),1)
        #memory_mask=torch.concat((~torch.ones(L*3, int(L*3.5), dtype=torch.bool, device='cuda'), memory_mask),1)
        
        # Associative Learning
        std=0.5
        noise=torch.randn_like(k_read)*std
        k_read=F.normalize(k_read)
        k_read=k_read+noise
        
        
        with torch.backends.cuda.sdp_kernel():
            y = F.scaled_dot_product_attention(q,k,v,attn_mask=attn_mask,
                                                dropout_p=self.dropout)
            v_read = F.scaled_dot_product_attention(k_read,k,v, attn_mask=memory_mask,
                                                    dropout_p=0)
            k_read = F.scaled_dot_product_attention(k_read,k,k, attn_mask=memory_mask,
                                                    dropout_p=0)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        k_read = k_read.transpose(1, 2).contiguous().view(B, T, -1)
        v_read = v_read.transpose(1, 2).contiguous().view(B, T, -1)
        
        # output projection
        y = self.resid_dropout(self.proj(y))
        return y, write_k, write_v, k_read, v_read
        #return y, write_k, write_v, None,None

    
class FFN(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, bias=False, ffn_mult=4):
        super().__init__()
        self.fc    = nn.Linear(d_model, ffn_mult * d_model, bias=bias)
        self.gelu  = nn.GELU()
        self.proj  = nn.Linear(ffn_mult * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    

class GPT_Block(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, bias=False, ffn_mult=4):
        super().__init__()
        self.ln_1 = LayerNormNoBias(d_model, bias=bias)
        self.attn = Attention(d_model, nhead, bias, dropout)
        self.ln_2 = LayerNormNoBias(d_model, bias=bias)
        self.mlp = FFN(d_model, dropout, bias, ffn_mult)

    def forward(self, x, is_causal=True):
        x_ln = self.ln_1(x)
        x = x + self.attn(x_ln, x_ln, x_ln, is_causal=is_causal)
        
        x = x + self.mlp(self.ln_2(x))
        
        return x
    
    


class GPT_Transformer(nsd_Module):
    def __init__(self, d_model, num_blks, nhead, seq_len,
                 dropout = 0.1, bias=False, report_params_count=True,
                 ffn_mult=4):
        super().__init__()

        #self.pos_encoding = nn.Sequential(nn.Linear(seq_len, d_model, bias=False),
        #                                  LayerNormNoBias(d_model)) #Stable Embedding Layer # Requires One Hot
        self.pos_encoding = nn.Embedding(seq_len, d_model)
        
        self.final_ln = LayerNormNoBias(d_model)
        self.start_dropout = nn.Dropout(dropout)

        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), GPT_Block(
                                d_model, nhead, dropout, bias=False, ffn_mult=ffn_mult))
            
        
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
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

        
    def forward(self, X, is_causal=True):

        pos = torch.arange(0, self.seq_len, dtype=torch.long, device='cuda')
        pos_emb = self.pos_encoding(pos)[:X.shape[1]]
        X = self.start_dropout(X+pos_emb)

        for i, blk in enumerate(self.blks):
            X = blk(X, is_causal)
            
        return self.final_ln(X)
    


class GPT_NLP(nsd_Module):
    def __init__(self, hiddens, num_blks, nhead, seq_len, vocab_size=50257,
                 temperature=1.0, k=20, p=0.9, sampling='gpt', report_params_count=True, tied_weights=True):
        super().__init__()
        
        
        self.emb_vocab = nn.Embedding(vocab_size, hiddens)
        self.gpt = GPT_Transformer(hiddens, nhead=nhead, num_blks=num_blks)
        
        self.cls = nn.Linear(hiddens, vocab_size, bias=False)
        
        if tied_weights:
            self.emb_vocab.weight = self.cls.weight

        
        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'GPT NLP Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')

    def forward(self, X, is_causal=True):
        batch_size, seq_len = X.shape
        
        mask = X>self.vocab_size
        X[mask] = self.vocab_size-1
        
        X = self.emb_vocab(X)
        #cls = torch.autograd.Variable(torch.zeros(batch_size, 2, self.hiddens)).to('cuda')
        
        #X = torch.cat((X, cls), dim=1)
        X = self.gpt(X, is_causal=is_causal)

        return self.cls(X)



class Transformer_Block_NoLN(nsd_Module):
    def __init__(self, d_model, nhead, dropout=0.0, bias=False, ffn_mult=4, stochastic_depth=1):
        super().__init__()
        self.ln_1 = LayerNormNoBias(d_model, bias=bias)
        self.attn = Attention(d_model, nhead, bias, dropout)
        self.ln_2 = LayerNormNoBias(d_model, bias=bias)
        self.mlp = FFN(d_model, dropout, bias, ffn_mult)

    def forward(self, x, is_causal=True):
        #x = renormalize(x)
        keep_path = torch.ones(x.shape[0],device='cuda')*(self.stochastic_depth if self.training else 1)
        keep_path = torch.bernoulli(keep_path)[:,None,None]

        x_ln = self.ln_1(x)
        x = x + self.attn(x_ln, x_ln, x_ln, is_causal=is_causal)*keep_path
        
        x = x + self.mlp(self.ln_2(x))*keep_path
        
        return x

class Transformer_NoDATA(nn.Module):
    def __init__(self, d_model, num_blks, nhead, seq_len,
                 dropout = 0.1, bias=False, report_params_count=True,
                 ffn_mult=4, stochastic_depth=1.0, scale_init=1):
        super().__init__()
        self.num_hiddens = d_model
        self.scale_init=scale_init
        if scale_init==1:
            self.scale_init=num_blks


        self.pos_encoding = nn.Embedding(seq_len, d_model)

        self.final_ln = LayerNormNoBias(d_model)
        self.start_dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_blks=num_blks

        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), Transformer_Block_NoLN(
                                d_model, nhead, dropout, bias=False, ffn_mult=ffn_mult,
                                stochastic_depth=1-((1-stochastic_depth)*i/num_blks) ))


        # https://proceedings.mlr.press/v119/huang20f/huang20f.pdf

        #self.apply(init_gpt)
        #for pn, p in self.named_parameters():
        #    if pn.endswith('proj.weight'):
        #        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blks))

        self.apply(init_xavier)
        
        #for pn, p in self.named_parameters():
        #    if pn.endswith('proj.weight') or pn.endswith('W_v.weight') or pn.endswith('fc.weight') or pn.endswith('pos_encoding.weight'):
        #        torch.nn.init.xavier_uniform_(p, gain=(torch.tensor(4*self.scale_init,dtype=torch.float)).pow(-1/4))
        #self.apply(self._init_weights)
        

        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'GPT Transformer Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            #torch.nn.init.normal_(module.weight, mean=0.0, std=1/math.sqrt(self.num_hiddens))
            torch.nn.init.xavier_uniform_(module.weight, gain=(torch.tensor(4*self.scale_init,dtype=torch.float)).pow(-1/4))
        
        

    def forward(self, X, is_causal=True):

        pos = torch.arange(0, self.seq_len, dtype=torch.long, device='cuda')
        pos_emb = self.pos_encoding(pos)[:X.shape[1]]
        X = self.start_dropout(X+pos_emb)
        

        for i, blk in enumerate(self.blks):
            X = blk(X, is_causal)
            
        X = self.final_ln(X)
        
        return X
    
    def no_pos(self, X, is_causal=True):
        X = self.start_dropout(X)
        
        
        for i, blk in enumerate(self.blks):
            X = blk(X, is_causal)

        X = self.final_ln(X)
        
        return X
    
    def masked(self, X, mask, is_causal=True):

        pos = torch.arange(0, self.seq_len, dtype=torch.long, device='cuda')
        pos_emb = self.pos_encoding(pos)[:X.shape[1]]
        X = self.start_dropout(X+pos_emb)
        X = X.gather(1, mask)
        
        
        for i, blk in enumerate(self.blks):
            X = blk(X, is_causal)

        X = self.final_ln(X)
        
        return X




    
def modulate(x, shift, scale):
    # x (B, T, D)
    # shift (B, D)
    # scale (B, D)
    
    return x * (1 + scale[:,None]) + shift[:,None]


class DiT_Block(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, bias=False, ffn_mult=4):
        super().__init__()
        self.ln_1 = LayerNormNoBias(d_model, bias=bias)
        self.attn = Attention(d_model, nhead, bias, dropout)
        self.ln_2 = LayerNormNoBias(d_model, bias=bias)
        self.mlp = FFN(d_model, dropout, bias, ffn_mult)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )

        self.ln_1.apply(init_gpt)
        self.attn.apply(init_gpt)
        self.ln_2.apply(init_gpt)
        self.mlp.apply(init_gpt)
        self.adaLN_modulation.apply(init_zeros)
        
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_ln = modulate(self.ln_1(x), shift_msa, scale_msa)
        #x_ln = modulate(x, shift_msa, scale_msa)
        x = x + gate_msa[:,None] * self.attn(x_ln, x_ln, x_ln, is_causal=False)
        x = x + gate_mlp[:,None] * self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp))
        #x = x + gate_mlp[:,None] * self.mlp(modulate(x, shift_mlp, scale_mlp))
        return x

    def forward_no_dit(self, x):
        x_ln = self.ln_1(x)
        x = x + self.attn(x_ln, x_ln, x_ln, is_causal=False)
        return x + self.mlp(self.ln_2(x))
    
    
class DiT_Transformer(nsd_Module):
    def __init__(self, d_model, num_blks, nhead, seq_len,
                 dropout = 0.1, bias=False, report_params_count=True,
                 ffn_mult=4, scale_init=1):
        super().__init__()
        if scale_init==1:
            scale_init=num_blks

        self.pos_encoding = nn.Embedding(seq_len, d_model)
        
        self.final_ln = LayerNormNoBias(d_model)
        self.start_dropout = nn.Dropout(dropout)
        

        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), DiT_Block(
                                d_model, nhead, dropout, bias=False, ffn_mult=ffn_mult))
            
        
        #nn.init.xavier_uniform_(self.pos_encoding[0].weight)
        
        self.apply(init_gpt)
        self.init_weights()
        
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blks))

        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'GPT Transformer Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')
    
    def init_weights(self):
        
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blks:
            block.adaLN_modulation[-1].apply(init_zeros)
    
        
    def forward(self, X, c):
        # Input:
        # X e (B, T, D)
        # c e (B, D)
        
        pos = torch.arange(0, self.seq_len, dtype=torch.long, device='cuda')
        pos_emb = self.pos_encoding(pos)
        
        X = self.start_dropout(X+pos_emb)

        for i, blk in enumerate(self.blks):
            X = blk(X, c)
            
        return self.final_ln(X)
    

   def forward_no_dit(self, X):
        # Input:
        # X e (B, T, D)
        # c e (B, D)
        
        pos = torch.arange(0, self.seq_len, dtype=torch.long, device='cuda')
        pos_emb = self.pos_encoding(pos)
        
        X = self.start_dropout(X+pos_emb)

        for i, blk in enumerate(self.blks):
            X = blk.forward_no_dit(X)
            
        return self.final_ln(X)
    
     
    

class CrossAttention_Block(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = LayerNormNoBias(d_model, bias=bias)
        self.attn = Attention(d_model, nhead, bias, dropout)
        self.ln_2 = LayerNormNoBias(d_model, bias=bias)
        self.mlp = FFN(d_model, dropout, bias)

    def forward(self, q, k, v, is_causal=False):
        q = q + self.attn(self.ln_1(q),self.ln_1(k),self.ln_1(v), is_causal=is_causal)
        q = q + self.mlp(self.ln_2(q))
        return q
    


class CrossAttention_Transformer(nn.Module):
    def __init__(self, d_model, num_blks, nhead, seq_len, dim_feedforward=2048,  
                 dropout = 0.1, vocab_size = 0, bias=False):
        super().__init__()

        self.pos_encoding = nn.Embedding(seq_len, d_model)
        
        self.out_ln = LayerNormNoBias(d_model)
        self.start_dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), CrossAttention_Block(
                                d_model, nhead, dropout, bias=False))
            
        
        nn.init.xavier_uniform_(self.pos_encoding[0].weight)


        
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blks))
        self.apply(self._init_weights)
        
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
    
    def forward(self, q, k, v, is_causal=False):

        pos = torch.arange(0, self.seq_len, dtype=torch.long, device='cuda')
        pos_emb = self.pos_encoding(pos)
        q = self.start_dropout(q+pos_emb)
        k = self.start_dropout(k+pos_emb)
        v = self.start_dropout(v+pos_emb)

        for i, blk in enumerate(self.blks):
            q = blk.forward(q,k,v, is_causal)
        q = self.out_ln(q)
        return q


    
    







class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.FloatTensor, zq: torch.FloatTensor) -> torch.FloatTensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f
    
    
    
class ConvAttnBlock(nn.Module):
    def __init__(self, in_channels, t_emb_dim=512, dropout=0, nhead=8):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.nhead = in_channels//nhead
        
        self.norm = nn.GroupNorm(32, in_channels)
        
        #self.norm = SpatialNorm(in_channels, t_emb_dim)

        self.q = torch.nn.Linear(in_channels,
                                 in_channels)
        self.k = torch.nn.Linear(in_channels,
                                 in_channels)
        self.v = torch.nn.Linear(in_channels,
                                 in_channels)
        self.proj_out = torch.nn.Linear(in_channels,
                                        in_channels)
        self.q.apply(init_cnn)
        self.k.apply(init_cnn)
        self.v.apply(init_cnn)
        self.proj_out.apply(init_cnn)


    def forward(self, x, t_emb=None):
        b, c, h, w = x.shape

        h_ = x
        h_ = self.norm(h_).view(b, c, h*w).transpose(1,2)
        
        #h_ = self.norm(h_, t_emb)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        q = q.contiguous().view(b, h*w, self.nhead, c//self.nhead).transpose(1, 2)
        k = k.contiguous().view(b, h*w, self.nhead, c//self.nhead).transpose(1, 2)
        v = k.contiguous().view(b, h*w, self.nhead, c//self.nhead).transpose(1, 2)

        # compute attention

        with torch.backends.cuda.sdp_kernel():
            h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)

        h_ = h_.transpose(1, 2).view(b, h*w, c)
        h_ = self.proj_out(h_).transpose(1,2)

        h_ = h_.reshape(b, c, h, w)

        return x+h_

    """
    def forward(self, x, t_emb=None):
        h_ = x
        h_ = self.norm(h_)
        print(f"{h_.shape}")
        #h_ = self.norm(h_, t_emb)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.view(b, c, h*w).transpose(1,2)
        k = k.view(b, c, h*w).transpose(1,2)
        v = v.view(b, c, h*w).transpose(1,2)
        '''
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        '''
        with torch.backends.cuda.sdp_kernel():
            h_ = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)

        h_ = h_.transpose(1, 2)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_
    """