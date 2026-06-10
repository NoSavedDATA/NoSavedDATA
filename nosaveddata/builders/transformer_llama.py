import math

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import Attention
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

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim, end, theta = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

"""
def apply_rotary_emb(
    xq,
    xk,
    freqs_cis_q,
    freqs_cis_k,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_q = reshape_for_broadcast(freqs_cis_q, xq_)
    freqs_cis_k = reshape_for_broadcast(freqs_cis_k, xk_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
"""
def apply_rotary(x, freqs_cis):
    x_ = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2)
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    out = torch.view_as_real(x_ * freqs_cis).flatten(-2)
    return out.type_as(x)

def repeat_kv(x, n_rep):
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )



class Attention_Rotary_Embedding(nn.Module):
    def __init__(self, d_model=512, num_heads=8, bias=False, dropout=0.1):
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
        self.n_head = num_heads
        self.n_embd = d_model
        self.dropout = dropout

    def forward(self, q, k, v, freqs_cis, is_causal, mask=None):
        B, T, C = q.size()
        
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        
        
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, nh, T, hs)
        k = k.view(B, -1, self.n_head, C // self.n_head) # (B, nh, hs, T)
        v = v.view(B, -1, self.n_head, C // self.n_head) # (B, nh, hs, T)
        
        #q, k = apply_rotary_emb(q, k, freqs_cis, freqs_cis)
        q = apply_rotary(q, freqs_cis[:q.shape[1]])
        k = apply_rotary(k, freqs_cis[:k.shape[1]])
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        
        # efficient attention using Flash Attention CUDA kernels
        
        with torch.backends.cuda.sdp_kernel():
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0, is_causal=is_causal)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.proj(y))
        return y




class FFN_LLaMa(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        multiple_of=256, # make SwiGLU hidden layer size multiple of large power of 2
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False,
        )
        self.w2_proj = nn.Linear(
            hidden_dim, dim, bias=False,
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False,
        )

    def forward(self, x):
        return self.w2_proj(F.silu(self.w1(x)) * self.w3(x))


class LLaMa_Block(nn.Module):
    def __init__(self, layer_id, d_model, ffn, nhead, bias=False, dropout=0.1, eps=1e-6, cross_attention=False):
        super().__init__()
        head_dim = d_model // nhead
        self.attention = Attention_Rotary_Embedding(d_model, nhead, bias=bias, dropout=dropout)
        self.feed_forward = FFN_LLaMa(
            dim=d_model,
            hidden_dim=ffn
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

        h = q + self.attention.forward(
            q, k, v, freqs_cis, is_causal
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
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

        h = q + self.attention.forward(
            q, k, v, freqs_cis, is_causal, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out






class LLaMa_Transformer(nn.Module):
    def __init__(self, d_model, ffn_dim, nhead, num_blks, seq_len, 
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
            self.layers.append(LLaMa_Block(layer_id, d_model, ffn_dim, nhead, bias, dropout, eps, cross_attention))

        self.norm = RMSNorm(d_model, eps=eps)

        self.freqs_cis = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

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
        
        self.freqs_cis = self.freqs_cis.to(q.device)
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]


        for layer in self.layers:
            q = layer(q, k, v, freqs_cis, causal, mask)
            # k=q and v=q if self attention, which is the default option.

        h = self.norm(q)
        
        

        return h







class LLaMa_NLP(nn.Module):
    def __init__(self, d_model, nhead, num_blks, seq_len, vocab_size,
                 dropout = 0.1, bias=False, eps=1e-6, report_params_count=True, tied_weights=False):
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
        self.vocab_size = vocab_size

        self.tok_embeddings = nn.Embedding(
            vocab_size, d_model
        )

        self.transformer = LLaMa_Transformer(d_model, nhead, num_blks, seq_len,
                                         dropout, bias, eps, report_params_count)

        self.output = nn.Linear(
            d_model, vocab_size, bias=bias
        )

        if tied_weights:
            self.tok_embeddings.weight = self.output.weight




        self.tok_embeddings.apply(self._init_weights)
        self.output.apply(self._init_weights)
        
        if report_params_count:
            params_to_count = [p for p in self.parameters() if p.requires_grad]
            print(f'LLaMa NLP Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x, causal, start_pos=0):
        
            
        mask = x>self.vocab_size
        x[mask] = self.vocab_size-1

        x = self.tok_embeddings(x)

        h = self.transformer(x, x, x, causal)
        
        output = self.output(h).float()

        return output

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half = self.dim // 2

        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(half, device=t.device) / half
        )

        args = t[:, None].float() * freqs[None]

        emb = torch.cat([
            torch.cos(args),
            torch.sin(args)
        ], dim=-1)

        return self.mlp(emb)


class DitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim, eps=1e-6):
        super().__init__()
        self.norm_final= RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def modulate(self, x, shift, scale):
        return x * (1 + scale[:,None]) + shift[:,None]

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)

        return x

class LLaMa_Block_DiT(nn.Module):
    def __init__(self, layer_id, d_model, ffn, nhead, bias=False, dropout=0.1, eps=1e-6, cross_attention=False):
        super().__init__()
        head_dim = d_model // nhead
        self.attention = Attention_Rotary_Embedding(d_model, nhead, bias=bias, dropout=dropout)
        self.feed_forward = FFN_LLaMa(
            dim=d_model,
            hidden_dim=ffn
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(d_model, eps=eps)
        self.ffn_norm = RMSNorm(d_model, eps=eps)

        if cross_attention:
            self.forward = self.forward_cross_attention
        else:
            self.forward = self.forward_self_attention

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        self.adaLN_modulation.apply(init_zeros)
        self.adaLN_modulation[1]._skip_init = True

    def modulate(self, x, shift, scale):
        return x * (1 + scale[:,None]) + shift[:,None]
    
    def forward_self_attention(
        self,
        q, k, v, c,
        freqs_cis,
        is_causal,
        mask=None
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        q_ln=self.attention_norm(q)
        q_ln = self.modulate(q_ln, shift_msa, scale_msa)
        k=q_ln
        v=q_ln

        h = q + gate_msa[:,None] * self.attention.forward(
            q_ln, k, v, freqs_cis, is_causal
        )
        out = h + gate_mlp[:,None] * self.feed_forward.forward(self.modulate(self.ffn_norm(h),shift_mlp,scale_mlp))
        return out

    def forward_cross_attention(
        self,
        q, k, v, c,
        freqs_cis,
        is_causal,
        mask=None
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        q_ln=self.attention_norm(q)
        k=self.attention_norm(k)
        v=self.attention_norm(v)
        q_ln = self.modulate(q_ln, shift_msa, scale_msa)
        k = self.modulate(k, shift_msa, scale_msa)
        v = self.modulate(v, shift_msa, scale_msa)

        h = q + gate_msa[:,None] * self.attention.forward(
            q_ln, k, v, freqs_cis, is_causal
        )
        out = h + gate_mlp[:,None] * self.feed_forward.forward(self.modulate(self.ffn_norm(h),shift_mlp,scale_mlp))
        return out

class LLaMa_DiT(nn.Module):
    def __init__(self, d_model, out_dim, ffn_dim, nhead, num_blks, seq_len, num_timesteps, 
                  dropout = 0.1, bias=False, eps=1e-6, report_params_count=True, cross_attention=False):
        super().__init__()
        self.num_blks = num_blks

        self.ts = TimestepEmbedding(d_model)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(num_blks):
            self.layers.append(LLaMa_Block_DiT(layer_id, d_model, ffn_dim, nhead, bias, dropout, eps, cross_attention))

        self.norm = RMSNorm(d_model, eps=eps)
        self.out = DitFinalLayer(d_model, out_dim, d_model)

        self.freqs_cis = precompute_freqs_cis(
            d_model // nhead, seq_len
        )

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
            if getattr(module, "_skip_init", False):
                return
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            #torch.nn.init.xavier_normal_(module.weight)
    
    def forward(self, q, k, v, t, causal, mask=None):


        _, seqlen, _ = q.shape
        
        self.freqs_cis = self.freqs_cis.to(q.device)
        freqs_cis = self.freqs_cis
        #freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        c = self.ts(t)

        for layer in self.layers:
            q = layer(q, k, v, c, freqs_cis, causal, mask)
            # k=q and v=q if self attention, which is the default option.

        # h = self.norm(q)
        h = self.out(q, c)

        return h


