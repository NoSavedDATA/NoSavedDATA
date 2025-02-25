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
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim, end, theta = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq,
    xk,
    freqs_cis_q,
    freqs_cis_k,
):
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_q = reshape_for_broadcast(freqs_cis_q, xq_)
    freqs_cis_k = reshape_for_broadcast(freqs_cis_k, xk_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_q).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
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
        
        q, k = apply_rotary_emb(q, k, freqs_cis, freqs_cis)
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
        """
        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        """
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

