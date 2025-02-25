import math

import torch
import torch.nn.functional as F
from torch import nn

from .transformer_llama import *
from .weight_init import *
# from torch.nn.attention import SDPBackend, sdpa_kernel


'''
REFERENCES:
https://github.com/facebookresearch/llama/blob/main/llama/model.py
'''

'''
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
'''





'''XL'''


class Attention_Rotary_Embedding_XL(nn.Module):
    def __init__(self, d_model, num_heads, seq_len, bias=False, dropout=0.1):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_len = seq_len

        self.k_xl_positinal_emb = nn.Embedding(self.seq_len, d_model)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        # output projection
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = num_heads
        self.n_embd = d_model
        self.dropout = dropout

        self.k_xl = None
        self.v_xl = None

    @torch.no_grad()
    def reset_indices(self, reset_indices, bs):

        if self.k_xl==None or not isinstance(reset_indices, torch.Tensor):
            self.k_xl = torch.zeros(bs, self.seq_len, self.d_model, device='cuda')
            self.v_xl = torch.zeros(bs, self.seq_len, self.d_model, device='cuda')
        else:
            # print(f"RESET: {self.k_xl.shape, reset_indices.shape}")
            # print(f"{reset_indices}")
            reset_indices = reset_indices[:,None,None].cuda()
            self.k_xl = self.k_xl * reset_indices # 1 or 0
            self.v_xl = self.v_xl * reset_indices


    def forward(self, q, k, v, freqs_cis, is_causal, mask=None):
        B, T, C = q.size()
        
        q, k, v  = self.W_qkv(x).split(self.d_model, dim=-1)


        
        k_pre = k.detach()
        v_pre = v.detach()


        self.k_xl = self.k_xl + self.k_xl_positinal_emb(torch.arange(0,self.k_xl.shape[-2],device='cuda'))[None,:]


        
        
        q = q.view(B,  T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        k = k.view(B, -1, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, -1, self.n_head, C // self.n_head) # (B, T, nh, hs)
        
        
        q, k = apply_rotary_emb(q, k, *freqs_cis)
        q = q.transpose(1, 2) # (B, nh, T, hs)
        k = k.transpose(1, 2) # (B, nh, T, hs)
        v = v.transpose(1, 2) # (B, nh, T, hs)
        k_xl = self.k_xl.to(q.dtype).view(B, -1, self.n_head, C // self.n_head).transpose(1,2)
        v_xl = self.v_xl.to(q.dtype).view(B, -1, self.n_head, C // self.n_head).transpose(1,2)


        k = torch.cat((k_xl, k),-2)
        v = torch.cat((v_xl, v),-2)

        # print(f"q {q.shape} k {k.shape}")
        
        self.k_xl = k_pre.to(q.dtype)
        self.v_xl = v_pre.to(q.dtype)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels

        

        if mask is not None and k.shape[-2]>q.shape[-2]:
            xl_mask = torch.ones(mask.shape[0],1,k.shape[-2],device=q.device)
            # print(f"{mask[...,None].shape,xl_mask.shape}")
            mask = mask[...,None]*xl_mask
            # mask = torch.cat((xl_mask,mask),-1)
            mask = mask[:,None]
            # print(f"{mask.shape}")



        # print(f"{q.shape, k.shape, v.shape, mask.shape}")        

        
        with torch.backends.cuda.sdp_kernel():
            y = torch.nn.functional.scaled_dot_product_attention(q.contiguous(), k.to(q.dtype).contiguous(), v.to(q.dtype).contiguous(), attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)
        # with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        #     y = F.scaled_dot_product_attention(q, k.to(q.dtype).contiguous(), v.to(q.dtype).contiguous(), is_causal=True)
        

        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.proj(y))
        return y

    def no_xl(self, q, k, v, freqs_cis, is_causal, mask=None):
        B, T, C = q.size()
        
        q, k, v  = self.W_qkv(x).split(self.d_model, dim=-1)


        # print(f"k {k.shape}, k xl: {self.k_xl.shape}")
        k_pre = k.detach()
        v_pre = v.detach()

        k_xl = self.k_xl + self.k_xl_positinal_emb(torch.arange(0,self.k_xl.shape[-2],device='cuda'))[None,:]
        
        
        q = q.view(B,  T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        k = k.view(B, -1, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, -1, self.n_head, C // self.n_head) # (B, T, nh, hs)
        
        
        q, k = apply_rotary_emb(q, k, *freqs_cis)
        q = q.transpose(1, 2) # (B, nh, T, hs)
        k = k.transpose(1, 2) # (B, nh, T, hs)
        v = v.transpose(1, 2) # (B, nh, T, hs)
        k_xl = k_xl.to(q.dtype).view(B, -1, self.n_head, C // self.n_head).transpose(1,2)
        v_xl = self.v_xl.to(q.dtype).view(B, -1, self.n_head, C // self.n_head).transpose(1,2)

        k = torch.cat((k_xl, k),-2)
        v = torch.cat((v_xl, v),-2)

        # print(f"q {q.shape} k {k.shape}")
        

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        
        
        # if not mask:
        #     with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        #         y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)
        # else:
        #     print(f"Mask is not none. Disabling flash attention.")
        #     y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)
        with torch.backends.cuda.sdp_kernel():
            y = torch.nn.functional.scaled_dot_product_attention(q.contiguous(), k.to(q.dtype).contiguous(), v.to(q.dtype).contiguous(), attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.proj(y))
        return y



class LLaMa_Block_XL(nn.Module):
    def __init__(self, layer_id, d_model, ffn_hiddens, nhead, seq_len, bias=False, dropout=0.1, eps=1e-6, cross_attention=False):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        head_dim = d_model // nhead
        self.attention = Attention_Rotary_Embedding_XL(d_model, nhead, seq_len, bias=bias, dropout=dropout)
        self.feed_forward = FFN_LLaMa(
            dim=d_model,
            hidden_dim=ffn_hiddens
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.ffn_norm = RMSNorm(d_model, eps=eps)

        if cross_attention:
            self.forward = self.forward_cross_attention
        else:
            self.forward = self.forward_self_attention
    
    def forward_self_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):
        q=self.attention_norm(q)
        k=q.clone()
        v=q.clone()

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    
    def no_xl(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):
        q=self.attention_norm(q)
        k=q.clone()
        v=q.clone()

        h = q + self.attention.no_xl(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward_cross_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):

        q=self.attention_norm(q)
        k=self.attention_norm(k)
        v=self.attention_norm(v)

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out






class LLaMa_Transformer_XL(nn.Module):
    def __init__(self, d_model, ffn_hiddens, nhead, num_blks, seq_len, 
                  dropout = 0.1, bias=False, eps=1e-6, report_params_count=True, cross_attention=False):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
        """

        super().__init__()
        self.num_blks = num_blks


        self.layers = torch.nn.ModuleList()
        for layer_id in range(num_blks):
            self.layers.append(LLaMa_Block_XL(layer_id, d_model, ffn_hiddens, nhead, seq_len, bias, dropout, eps, cross_attention))

        self.norm = RMSNorm(d_model, eps=eps)

        freqs_cis_q = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        freqs_cis_k = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        self.freqs_cis = (freqs_cis_q, freqs_cis_k)

        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'LLaMa Transformer Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blks))
   
    def reset_indices(self, reset_indices, bs):

        for layer in self.layers:
            layer.attention.reset_indices(reset_indices, bs) 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
    
    def forward(self, q, k, v, causal, mask=None):


        _, seqlen, _ = q.shape
        
        self.freqs_cis = (self.freqs_cis[0].to(q.device),self.freqs_cis[1].to(q.device))
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]


        for layer in self.layers:
            q = layer(q, k, v, freqs_cis, causal, mask)
            # k=q and v=q if self attention, which is the default option.

        h = self.norm(q)
        
        

        return h
    def no_xl(self, q, k, v, causal, mask=None):


        _, seqlen, _ = q.shape
        
        self.freqs_cis = (self.freqs_cis[0].to(q.device),self.freqs_cis[1].to(q.device))
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]


        for layer in self.layers:
            q = layer.no_xl(q, k, v, freqs_cis, causal, mask)
            # k=q and v=q if self attention, which is the default option.

        h = self.norm(q)
        
        

        return h




# This code is here and not in transformer.py due to circular imports.
class SoftMoE_LLaMa_FFN(nn.Module):
    def __init__(self, hiddens, bias, ffn_hiddens, num_experts=8):
        super().__init__()
        self.num_experts = num_experts

        self.slots = nn.Linear(hiddens, num_experts, bias=bias)

        # self.experts = nn.ModuleList([FFN(num_slots*hiddens, dropout, bias, ffn_mult) for _ in range(num_experts)])
        self.experts = nn.ModuleList([FFN_LLaMa(dim=hiddens, hidden_dim=ffn_hiddens) for _ in range(num_experts)])
    
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



class LLaMa_Block_XL_SoftMoE(nn.Module):
    def __init__(self, layer_id, d_model, ffn_hiddens, nhead, seq_len, num_experts, bias=False, dropout=0.1, eps=1e-6, cross_attention=False):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        head_dim = d_model // nhead
        self.attention = Attention_Rotary_Embedding_XL(d_model, nhead, seq_len, bias=bias, dropout=dropout)
        self.feed_forward = SoftMoE_LLaMa_FFN(d_model, bias, ffn_hiddens=ffn_hiddens, num_experts=num_experts)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.ffn_norm = RMSNorm(d_model, eps=eps)

        if cross_attention:
            self.forward = self.forward_cross_attention
        else:
            self.forward = self.forward_self_attention
    
    def forward_self_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):
        q=self.attention_norm(q)
        k=q.clone()
        v=q.clone()

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def no_xl(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):
        q=self.attention_norm(q)
        k=q.clone()
        v=q.clone()

        h = q + self.attention.no_xl(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward_cross_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):

        q=self.attention_norm(q)
        k=self.attention_norm(k)
        v=self.attention_norm(v)

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out




class LLaMa_Transformer_XL_SoftMoE(nn.Module):
    def __init__(self, d_model, ffn_hiddens, nhead, num_blks, num_blks_moe, seq_len, num_experts,
                  dropout = 0.1, bias=False, eps=1e-6, report_params_count=True, cross_attention=False):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
        """

        super().__init__()
        self.num_blks = num_blks


        self.layers = torch.nn.ModuleList()
        for layer_id in range(num_blks):
            self.layers.append(LLaMa_Block_XL_SoftMoE(num_blks+layer_id, d_model, ffn_hiddens, nhead, seq_len, num_experts, bias, dropout, eps, cross_attention))
        for layer_id in range(num_blks_moe):
            self.layers.append(LLaMa_Block_XL(layer_id, d_model, ffn_hiddens, nhead, seq_len, bias, dropout, eps, cross_attention))

        self.norm = RMSNorm(d_model, eps=eps)

        freqs_cis_q = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        freqs_cis_k = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        self.freqs_cis = (freqs_cis_q, freqs_cis_k)

        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'LLaMa Transformer Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blks))
    

    def reset_indices(self, reset_indices, bs):

        for layer in self.layers:
            layer.attention.reset_indices(reset_indices, bs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
    
    def forward(self, q, k, v, causal, mask=None):


        _, seqlen, _ = q.shape
        
        self.freqs_cis = (self.freqs_cis[0].to(q.device),self.freqs_cis[1].to(q.device))
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]


        for layer in self.layers:
            q = layer(q, k, v, freqs_cis, causal, mask)
            # k=q and v=q if self attention, which is the default option.

        h = self.norm(q)
        
        

        return h

    
    def no_xl(self, q, k, v, causal, mask=None):


        _, seqlen, _ = q.shape
        
        self.freqs_cis = (self.freqs_cis[0].to(q.device),self.freqs_cis[1].to(q.device))
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]


        for layer in self.layers:
            q = layer.no_xl(q, k, v, freqs_cis, causal, mask)
            # k=q and v=q if self attention, which is the default option.

        h = self.norm(q)
        
        

        return h








class LLaMa_Block_SoftMoE(nn.Module):
    def __init__(self, layer_id, d_model, ffn_hiddens, nhead, num_experts, bias=False, dropout=0.1, eps=1e-6, cross_attention=False):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        head_dim = d_model // nhead
        self.attention = Attention_Rotary_Embedding(d_model, nhead, bias=bias, dropout=dropout)
        self.feed_forward = SoftMoE_LLaMa_FFN(d_model, bias, ffn_hiddens, num_experts=num_experts)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.ffn_norm = RMSNorm(d_model, eps=eps)

        if cross_attention:
            self.forward = self.forward_cross_attention
        else:
            self.forward = self.forward_self_attention
    
    def forward_self_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):
        q=self.attention_norm(q)
        k=q.clone()
        v=q.clone()

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward_cross_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):

        q=self.attention_norm(q)
        k=self.attention_norm(k)
        v=self.attention_norm(v)

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out




class LLaMa_Transformer_SoftMoE(nn.Module):
    def __init__(self, d_model, ffn_hiddens, nhead, num_blks, seq_len, num_experts,
                  dropout = 0.1, bias=False, eps=1e-6, report_params_count=True, cross_attention=False):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
        """

        super().__init__()
        self.num_blks = num_blks


        self.layers = torch.nn.ModuleList()
        for layer_id in range(num_blks//2):
            self.layers.append(LLaMa_Block(layer_id, d_model, ffn_hiddens, nhead, bias, dropout, eps, cross_attention))
        for layer_id in range(num_blks//2):
            self.layers.append(LLaMa_Block_SoftMoE(layer_id+num_blks//2, d_model, ffn_hiddens, nhead, num_experts, bias, dropout, eps, cross_attention))

        self.norm = RMSNorm(d_model, eps=eps)

        freqs_cis_q = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        freqs_cis_k = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        self.freqs_cis = (freqs_cis_q, freqs_cis_k)

        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'LLaMa Transformer Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
    
    def forward(self, q, k, v, causal, mask=None):


        _, seqlen, _ = q.shape
        
        self.freqs_cis = (self.freqs_cis[0].to(q.device),self.freqs_cis[1].to(q.device))
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]


        for layer in self.layers:
            q = layer(q, k, v, freqs_cis[0], causal, mask)
            # k=q and v=q if self attention, which is the default option.

        h = self.norm(q)
        
        

        return h






class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)
        self.saved_logits = logits

        #Noise logits
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)

        noisy_logits = logits + noise if self.training else logits


        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices


class SparseMoE(nn.Module):
    def __init__(self, hiddens, ffn_hiddens, num_experts, slots=2):
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.router = NoisyTopkRouter(hiddens, num_experts, slots)
        self.experts = nn.ModuleList([FFN_LLaMa(dim=hiddens, hidden_dim=ffn_hiddens) for _ in range(num_experts)])

        self.f_i = [0 for _ in range(num_experts)] # switch router loss
        self.P_i = [0 for _ in range(num_experts)]

    def get_load_balancing_loss(self):
        f_i = torch.stack(self.f_i)
        P_i = torch.stack(self.P_i)



        return self.num_experts * ( (f_i*P_i).sum() )

    def forward(self, x):
        n = x.shape[-2]
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)


            if flat_mask.any():
                self.f_i[i] = flat_mask.float().sum(-1)/n

                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                self.P_i[i] = gating_scores.sum(0).squeeze()/n

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output


class SparseMoE_ExpertParallelism(nn.Module):
    # Exchanges batch parallelism into expert parallelism
    def __init__(self, hiddens, bias, num_experts=4, slots=2, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.slots = slots
        

        self.gate = nn.Linear(hiddens, num_experts, bias=bias)


        self.experts = nn.ModuleList([FFN_LLaMa(dim=hiddens, hidden_dim=ffn_mult * hiddens) for _ in range(num_experts)])

    def get_load_balancing_loss(self):
      return (F.softmax(self.saved_logits,-1)+1e-7).log()
    
    def forward(self, X):
        # Input & Output shape:  (B, T, D)

        B, T, D = X.shape
        
        residue = X.clone()

        logits = self.gate(X)
        self.saved_logits = logits

        # print(f"logits {logits.shape}")
        topk_val, topk = logits.topk(self.slots)
        topk_val = F.softmax(topk_val, -1)[...,None]

        # print(f"{topk}")


        X = X[...,None,:]*topk_val
        # print(f"X: {X.shape}")



        
        for b in range(B):

            experts_inputs = [[] for _ in range(self.num_experts)]
            experts_to_seq = [[] for _ in range(self.num_experts)]

            for i in range(self.num_experts):
                mask = topk[b].view(-1) == i
                mask = torch.nonzero(mask)[...,-1]


                experts_inputs[i] = mask
                if mask.shape[-1]>0:
                    tokens_map = mask // self.slots
                    experts_to_seq[i] = tokens_map



            T_outputs = [[] for _ in range(T)]
            T_outputs_idx = [[] for _ in range(T)]
            
            for i in range(self.num_experts):
                if experts_inputs[i].shape[-1] == 0:
                    continue
                xb = X[b].view(-1,D)


                
                x = xb.gather(0,experts_inputs[i][:,None].repeat_interleave(D,-1))[None,:]
                

                x = self.experts[i](x)[0]

                for k, k_out in enumerate(experts_to_seq[i]):
                    T_outputs[k_out].append(x[k])
                    T_outputs_idx[k].append(k_out)




            for t in range(T):
                residue[b,t] += torch.stack(T_outputs[t]).sum(0)

            # print(f"residue {residue.shape}")
        
        return residue




class LLaMa_Block_SparseMoE(nn.Module):
    def __init__(self, layer_id, d_model, ffn_hiddens, nhead, num_experts, bias=False, dropout=0.1, eps=1e-6, cross_attention=False):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        head_dim = d_model // nhead
        self.attention = Attention_Rotary_Embedding(d_model, nhead, bias=bias, dropout=dropout)
        self.feed_forward = SparseMoE(d_model, ffn_hiddens, num_experts)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.ffn_norm = RMSNorm(d_model, eps=eps)

        if cross_attention:
            self.forward = self.forward_cross_attention
        else:
            self.forward = self.forward_self_attention
    
    def forward_self_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):
        q=self.attention_norm(q)
        k=q.clone()
        v=q.clone()

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward_cross_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):

        q=self.attention_norm(q)
        k=self.attention_norm(k)
        v=self.attention_norm(v)

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out




class LLaMa_Transformer_SparseMoE(nn.Module):
    def __init__(self, d_model, ffn_hiddens, nhead, num_blks, num_blks_moe, seq_len, num_experts,
                  dropout = 0.1, bias=False, eps=1e-6, report_params_count=True, cross_attention=False):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
        """

        super().__init__()
        self.num_blks = num_blks


        self.layers = torch.nn.ModuleList()
        for layer_id in range(num_blks):
            self.layers.append(LLaMa_Block(layer_id, d_model, ffn_hiddens, nhead, bias, dropout, eps, cross_attention))
        for layer_id in range(num_blks_moe):
            self.layers.append(LLaMa_Block_SparseMoE(layer_id+num_blks, d_model, ffn_hiddens, nhead, num_experts, bias, dropout, eps, cross_attention))

        self.norm = RMSNorm(d_model, eps=eps)

        freqs_cis_q = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        freqs_cis_k = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        self.freqs_cis = (freqs_cis_q, freqs_cis_k)

        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'LLaMa Transformer Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blks))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)

    def get_load_balancing_loss(self):
        loss = []
        for layer in self.layers[self.num_blks:]:
            loss.append(layer.feed_forward.get_load_balancing_loss())
        
        return torch.stack(loss).mean()


    def forward(self, q, k, v, causal, mask=None):


        _, seqlen, _ = q.shape
        
        self.freqs_cis = (self.freqs_cis[0].to(q.device),self.freqs_cis[1].to(q.device))
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]


        for layer in self.layers:
            q = layer(q, k, v, freqs_cis[0], causal, mask)
            # k=q and v=q if self attention, which is the default option.

        h = self.norm(q)
        
        

        return h






class LLaMa_Block_SparseMoE_XL(nn.Module):
    def __init__(self, layer_id, d_model, ffn_hiddens, nhead, seq_len, num_experts, bias=False, dropout=0.1, eps=1e-6, cross_attention=False):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        head_dim = d_model // nhead
        self.attention = Attention_Rotary_Embedding_XL(d_model, nhead, seq_len, bias=bias, dropout=dropout)
        self.feed_forward = SparseMoE(d_model, ffn_hiddens, num_experts)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.ffn_norm = RMSNorm(d_model, eps=eps)

        if cross_attention:
            self.forward = self.forward_cross_attention
        else:
            self.forward = self.forward_self_attention
    
    def forward_self_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):
        q=self.attention_norm(q)
        k=q.clone()
        v=q.clone()

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def no_xl(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):
        q=self.attention_norm(q)
        k=q.clone()
        v=q.clone()

        h = q + self.attention.no_xl(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward_cross_attention(
        self,
        q, k, v,
        freqs_cis,
        is_causal,
        mask=None
    ):

        q=self.attention_norm(q)
        k=self.attention_norm(k)
        v=self.attention_norm(v)

        h = q + self.attention(
            q, k, v, freqs_cis, is_causal, mask=mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out




class LLaMa_Transformer_SparseMoE_XL(nn.Module):
    def __init__(self, d_model, ffn_hiddens, nhead, num_blks, seq_len, num_experts,
                  dropout = 0.1, bias=False, eps=1e-6, report_params_count=True, cross_attention=False):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
        """

        super().__init__()
        self.num_blks = num_blks


        self.layers = torch.nn.ModuleList()

        for layer_id in range(num_blks):
            if layer_id%2==0:
                self.layers.append(LLaMa_Block_SparseMoE_XL(layer_id, d_model, ffn_hiddens, nhead, seq_len, num_experts, bias, dropout, eps, cross_attention))
            else:
                self.layers.append(LLaMa_Block_XL(layer_id, d_model, ffn_hiddens, nhead, seq_len, bias, dropout, eps, cross_attention))

        self.norm = RMSNorm(d_model, eps=eps)

        freqs_cis_q = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        freqs_cis_k = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

        self.freqs_cis = (freqs_cis_q, freqs_cis_k)

        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'LLaMa Transformer Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')


        self.apply(init_switch_t)
        # self.apply(self._init_weights)
    #     for pn, p in self.named_parameters():
    #         if pn.endswith('proj.weight'):
    #             torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_blks))

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def reset_indices(self, reset_indices, bs):

        for layer in self.layers:
            layer.attention.reset_indices(reset_indices, bs)

    def get_load_balancing_loss(self):
        loss = []
        for layer in self.layers[::2]:
            loss.append(layer.feed_forward.get_load_balancing_loss())
        
        return torch.stack(loss).mean()


    def forward(self, q, k, v, causal, mask=None):


        _, seqlen, _ = q.shape
        
        self.freqs_cis = (self.freqs_cis[0].to(q.device),self.freqs_cis[1].to(q.device))
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]


        for layer in self.layers:
            q = layer(q, k, v, freqs_cis, causal, mask)
            # k=q and v=q if self attention, which is the default option.

        h = self.norm(q)
        
        

        return h
    
    
    def no_xl(self, q, k, v, causal, mask=None):


        _, seqlen, _ = q.shape
        
        self.freqs_cis = (self.freqs_cis[0].to(q.device),self.freqs_cis[1].to(q.device))
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]


        for layer in self.layers:
            q = layer.no_xl(q, k, v, freqs_cis, causal, mask)
            # k=q and v=q if self attention, which is the default option.

        h = self.norm(q)
        
        

        return h

