# REFERENCES
# https://github.com/huggingface/diffusers
# https://github.com/JegZheng/truncated-diffusion-probabilistic-models

from .resnet import *
from .weight_init import *
from .transformer import  Attention
from .transformer_llama import *

import torch, torchvision
from torch import nn
import torch.nn.functional as F

import numpy as np
import math



class LLaMa_Block_Cross_Self_Attention(nn.Module):
    def __init__(self, d_model, nhead, seq_len, res, bias=False, dropout=0, eps=1e-6):
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
        nhead = d_model//64
        head_dim = d_model // nhead
        self.attention1 = Attention_Rotary_Embedding(d_model, nhead, bias=bias, dropout=dropout)
        self.attention2 = Attention_Rotary_Embedding(d_model, nhead, bias=bias, dropout=dropout)
        self.feed_forward = FFN_LLaMa(
            dim=d_model,
            hidden_dim=4 * d_model
        )
        self.attention_norm1 = RMSNorm(d_model, eps=eps)
        self.attention_norm2 = RMSNorm(d_model, eps=eps)
        self.ffn_norm = RMSNorm(d_model, eps=eps)

        self.freqs_cis1 = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            d_model // nhead, np.prod(res)
        ).cuda()
        self.freqs_cis2 = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            d_model // nhead, seq_len
        ).cuda()
        self.res=res

        self.attention1.apply(init_gpt)
        self.attention2.apply(init_gpt)
        self.feed_forward.apply(init_gpt)

    def forward(
        self,
        q, k, v,
        is_causal,
    ):

        bs, dim, cnn_shape = q.shape[0], q.shape[1], q.shape[2:]

        q=self.attention_norm1(q.view(bs, dim, -1).transpose(-1,-2))
        
        
        h = q + self.attention1.forward(
            q, q.clone(), q.clone(), self.freqs_cis1, is_causal
        )


        q=self.attention_norm2(q)
        k=self.attention_norm2(k)
        v=self.attention_norm2(v)

        h = q + self.attention2.forward(
            q, k, v, self.freqs_cis2, is_causal
        )

        out = h + self.feed_forward.forward(self.ffn_norm(h))
        out = out.transpose(-2,-1).view(bs, dim, *cnn_shape)
        return out



class CaptionProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, num_tokens=120):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        #self.register_buffer("y_embedding", nn.Parameter(torch.randn(num_tokens, in_features) / in_features**0.5))

        self.linear_1.apply(init_relu)
        self.linear_2.apply(init_orth)

    def forward(self, caption, force_drop_ids=None):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def sinusoidal_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    if len(timesteps.shape) != 1:
        print(timesteps.shape)
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb  

        
class Residual_Block_T_emb_1D(nn.Module):
    def __init__(self, in_channels, channels, stride=1, out_act=nn.Mish(), t_emb_dim=64, dropout=0):
        super().__init__()
        
        
        
        self.conv = nn.Sequential(nn.BatchNorm1d(in_channels),
                                  nn.Conv1d(in_channels, channels, kernel_size=3, padding=1, padding_mode='replicate',
                                            stride=stride),
                                  #nn.GroupNorm(channels//8, channels, eps=1e-6),
                                  nn.Mish())
        self.t_emb_proj = nn.Sequential(nn.Linear(t_emb_dim, channels))
        self.conv2 = nn.Sequential(nn.Dropout(p=dropout),
                                  nn.BatchNorm1d(channels),
                                  nn.Conv1d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate'),
                                  #nn.GroupNorm(channels//8, channels, eps=1e-6),
                                  out_act,
                                  )
        self.proj=nn.Identity()
        if stride>1 or in_channels!=channels:
            self.proj = nn.Conv1d(in_channels, channels, kernel_size=1,
                        stride=1, padding=0)
            self.proj.apply(init_proj2d)
        

        self.conv.apply(init_relu)
        if out_act==nn.Sigmoid() or out_act==nn.Identity():
            self.conv2.apply(init_orth)
        else:
            self.conv2.apply(init_relu)
            
        self.t_emb_proj.apply(init_orth)
        
    def forward(self, X, t_emb):
        Y = self.conv(X)
        t_emb = self.t_emb_proj(t_emb).view(Y.shape[0], 1, -1)
        
        Y = self.conv2((Y.transpose(-2,-1) + t_emb).transpose(-2,-1))

        return Y + self.proj(X)
        


class Down_ResNet_Blocks(nn.Module):
    def __init__(self, in_hiddens, out_hiddens, t_emb_dim=64, num_blocks=2, stride=2):
        super().__init__()

        in_hid = in_hiddens
        self.residuals = nn.ModuleList([])

        for i in range(num_blocks):
            self.residuals.append(Residual_Block_T_emb_1D(in_hid, out_hiddens, t_emb_dim=t_emb_dim))
            in_hid = out_hiddens


        
    def forward(self, X, t_emb, c_emb):
        residual = ()
        for blk in self.residuals:
            X = blk(X, t_emb)
            residual = residual + (X,)
        
        return X, residual




class Down_Attention_Blocks(nn.Module):
    def __init__(self, in_hiddens, out_hiddens, t_emb_dim=64, num_blocks=2, stride=2):
        super().__init__()

        in_hid = in_hiddens
        self.residuals = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        for i in range(num_blocks):
            self.residuals.append(Residual_Block_T_emb_1D(in_hid, out_hiddens, t_emb_dim=t_emb_dim))
            self.attentions.append(Attention(out_hiddens, num_heads=out_hiddens//64))
            in_hid = out_hiddens

        self.attentions.apply(init_gpt)

    def forward(self, X, t_emb, c_emb):
        residual = ()
        for blk, attn in zip(self.residuals, self.attentions):
            X = blk(X, t_emb)
            X = X.transpose(-1,-2)
            X = attn(X, X, X, is_causal=False)
            X = X.transpose(-1,-2)
            residual = residual + (X,)
        
        return X, residual

class Down_CrossAttention_Blocks(nn.Module):
    def __init__(self, in_hiddens, out_hiddens, res, t_emb_dim=64, c_emb_dim=64, seq_len=128, num_blocks=2, stride=2):
        super().__init__()

        self.condition_proj = nn.Linear(c_emb_dim, out_hiddens)
        in_hid = in_hiddens
        
        self.residuals = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(num_blocks):
            self.residuals.append(Residual_Block_T_emb_1D(in_hid, out_hiddens, t_emb_dim=t_emb_dim))
            self.attentions.append(LLaMa_Block_Cross_Self_Attention(out_hiddens, nhead=8, seq_len=seq_len, res=res))
            in_hid = out_hiddens

        self.condition_proj.apply(init_orth)
        
        

    def forward(self, X, t_emb, c_emb):
        c_emb = self.condition_proj(c_emb)
        residual = ()
        for blk, attn in zip(self.residuals, self.attentions):
            X = blk(X, t_emb)
            X = attn(X, c_emb, c_emb, is_causal=False)
            residual = residual + (X,)
        
        return X, residual



class Up_ResNet_Blocks(nn.Module):
    def __init__(self, in_hiddens, out_hiddens, prev_out_hiddens, t_emb_dim=64, num_blocks=2, stride=2):
        super().__init__()

        in_hid = in_hiddens
        self.residuals = nn.ModuleList([])
        
        for i in range(num_blocks):
            res_skip_channels = in_hiddens if (i == num_blocks - 1) else out_hiddens
            resnet_in_channels = prev_out_hiddens if i == 0 else out_hiddens


            self.residuals.append(Residual_Block_T_emb_1D(resnet_in_channels + res_skip_channels, out_hiddens, t_emb_dim=t_emb_dim))
            in_hid = out_hiddens


        
    def forward(self, X, t_emb, c_emb, res_sample):

        for blk in self.residuals:
            res_hidden = res_sample[-1]
            res_sample = res_sample[:-1]
            
            X = torch.cat((X, res_hidden), -2)
            X = blk(X, t_emb)
        return X


class Up_Attention_Blocks(nn.Module):
    def __init__(self, in_hiddens, out_hiddens, prev_out_hiddens, t_emb_dim=64, num_blocks=2, stride=2):
        super().__init__()

        self.residuals = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        for i in range(num_blocks):
            res_skip_channels = in_hiddens if (i == num_blocks - 1) else out_hiddens
            resnet_in_channels = prev_out_hiddens if i == 0 else out_hiddens
            
            self.residuals.append(Residual_Block_T_emb_1D(res_skip_channels + resnet_in_channels, out_hiddens, t_emb_dim=t_emb_dim))
            self.attentions.append(Attention(out_hiddens, num_heads=out_hiddens//64))
            
        self.attentions.apply(init_gpt)

    def forward(self, X, t_emb, c_emb, res_sample):

        for blk, attn in zip(self.residuals, self.attentions):
            res_hidden = res_sample[-1]
            res_sample = res_sample[:-1]
            
            X = torch.cat((X, res_hidden), -2)
            X = blk(X, t_emb)

            X = X.transpose(-1,-2)
            X = attn(X, X, X, is_causal=False)
            X = X.transpose(-1,-2)
        
        return X


class Up_CrossAttention_Blocks(nn.Module):
    def __init__(self, in_hiddens, out_hiddens, prev_out_hiddens, res, t_emb_dim=64, c_emb_dim=64, seq_len=128, num_blocks=2, stride=2):
        super().__init__()

        self.condition_proj = nn.Linear(c_emb_dim, out_hiddens)

        self.residuals = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        for i in range(num_blocks):
            res_skip_channels = in_hiddens if (i == num_blocks - 1) else out_hiddens
            resnet_in_channels = prev_out_hiddens if i == 0 else out_hiddens

            self.residuals.append(Residual_Block_T_emb_1D(res_skip_channels + resnet_in_channels, out_hiddens, t_emb_dim=t_emb_dim))
            self.attentions.append(LLaMa_Block_Cross_Self_Attention(out_hiddens, nhead=8, seq_len=seq_len, res=res))

        self.condition_proj.apply(init_orth)

    def forward(self, X, t_emb, c_emb, res_sample):
        c_emb = self.condition_proj(c_emb)

        for blk, attn in zip(self.residuals, self.attentions):
            res_hidden = res_sample[-1]
            res_sample = res_sample[:-1]
            
            X = torch.cat((X, res_hidden), -2)
            X = blk(X, t_emb)
            X = attn(X, c_emb, c_emb, is_causal=False)
        
        return X




    
class UNet_Middle(nn.Module):
    def __init__(self, in_hiddens, out_hiddens, res, t_emb_dim=64, c_emb_dim=64, seq_len=128, num_blocks=1, middle_cross_attn=True):
        super().__init__()
        self.condition_proj = nn.Linear(c_emb_dim, out_hiddens)
        self.middle_cross_attn = middle_cross_attn

        self.residual1 = Residual_Block_T_emb_1D(in_hiddens, out_hiddens, t_emb_dim=t_emb_dim)
        self.attentions = nn.ModuleList([])
        self.residuals = nn.ModuleList([])
        for i in range(num_blocks):
            if middle_cross_attn:
                self.attentions.append(LLaMa_Block_Cross_Self_Attention(out_hiddens, nhead=8, seq_len=seq_len, res=res))
                #self.attentions.append(Attention(out_hiddens, num_heads=out_hiddens//64))
            self.residuals.append(Residual_Block_T_emb_1D(out_hiddens, in_hiddens, t_emb_dim=t_emb_dim))

        self.condition_proj.apply(init_orth)

    def forward(self, X, t_emb, c_emb):
        X = self.residual1(X, t_emb)
        #c_emb = self.condition_proj(c_emb)

        for attn, blk in zip(self.attentions,self.residuals):
            #X = attn(X, c_emb, c_emb, is_causal=False)
            X = blk(X, t_emb)
        return X
    



class DownSample1d(nn.Module):
    def __init__(self, in_channels, out_channels, res, t_emb, c_emb_dim, seq_len, num_blocks=2, stride=2, type='ResNet', residual_butterfly=True):
        super().__init__()
        self.residual_butterfly=residual_butterfly

        if type=='ResNet':
            self.residual = Down_ResNet_Blocks(in_channels, out_channels, t_emb, num_blocks, stride=stride)
        elif type=='Attention':
            self.residual = Down_Attention_Blocks(in_channels, out_channels, t_emb, num_blocks, stride=stride)
        elif type=='CrossAttention':
            self.residual = Down_CrossAttention_Blocks(in_channels, out_channels, res, t_emb, c_emb_dim, seq_len, num_blocks, stride=stride)
        else:
            print(f"Not Implemented Downsampling Type: {type}")
        
        if stride==2:
            self.downsample = nn.Sequential(nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1),
                               #nn.MaxPool2d(3,2, padding=1),
                               #nn.GroupNorm(32, out_channels) if out_channels%32==0 else nn.BatchNorm1d(out_channels),
                               #nn.Mish(),
                               #Residual_Block(out_hiddens, out_hiddens)
                              )
            self.downsample[0].apply(init_orth)
        else:
            self.downsample = None

    def forward(self, X, t_emb, c_emb):
        X, residual = self.residual(X, t_emb, c_emb)

        if self.downsample != None:
            X = self.downsample(X)

        if self.residual_butterfly:
            residual = residual + (X,)

        return X, residual
    

class UpSample1d(nn.Module):
    def __init__(self, in_channels, out_channels, prev_out_hiddens, res, t_emb, c_emb_dim, seq_len, num_blocks=2, stride=2, type='ResNet'):
        super().__init__()
        

        if type=='ResNet':
            self.residual = Up_ResNet_Blocks(in_channels, out_channels, prev_out_hiddens, t_emb, num_blocks, stride=stride)
        elif type=='Attention':
            self.residual = Up_Attention_Blocks(in_channels, out_channels, prev_out_hiddens, t_emb, num_blocks, stride=stride)
        elif type=='CrossAttention':
            self.residual = Up_CrossAttention_Blocks(in_channels, out_channels, prev_out_hiddens, res, t_emb, c_emb_dim, seq_len, num_blocks, stride=stride)
        else:
            print(f"Not Implemented Upsampling Type: {type}")
        
        if stride==2:
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2),
                                          nn.Conv1d(out_channels, out_channels, 3, padding=1))
            self.upsample[1].apply(init_orth)
        else:
            self.upsample = nn.Identity()

    def forward(self, X, t_emb, c_emb, res_sample):
        X = self.residual(X, t_emb, c_emb, res_sample)
        X = self.upsample(X)

        return X




class UNet_Temporal(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, hidden_groups=[16,32,64], strides=[2,2,1], # either 1 or 2
                 down_blocks = ['ResNet', 'Attention', 'ResNet'],
                 up_blocks = ['ResNet', 'Attention', 'ResNet'],
                 num_blocks = 2,
                 num_blocks_upsample=3,
                 middle_cross_attn=True,
                 t_emb=512, c_emb_dim=512,
                 seq_len=128,
                 res=8):
        super().__init__()
        assert len(strides)==len(hidden_groups), f'Strides must have same len as hidden groups, got {len(strides)} and {len(hidden_groups)}'
        assert len(down_blocks)==len(hidden_groups), f'Down Blocks must have same len as hidden groups: {len(down_blocks)} and {len(hidden_groups)}'
        assert len(up_blocks)==len(hidden_groups), f'Up Blocks must have same len hidden groups: {len(up_blocks)} and {len(hidden_groups)}'
        assert all([i==1 for i in strides]) or (all([i==2 for i in strides[:-1]]) and strides[-1]==1)

        self.downsample = nn.ModuleList([])
        self.upsample = nn.ModuleList([])
        
        
        hidden_groups=np.array(hidden_groups)
        
        self.in_conv = nn.Conv1d(in_channel, hidden_groups[0], 3, padding=1)
        self.out_conv = nn.Sequential(
                                      nn.BatchNorm1d(hidden_groups[0]),
                                      nn.Mish(),
                                      nn.Conv1d(hidden_groups[0], in_channel, 3, padding=1)
                                     )
        
        out_hidden = hidden_groups[0]
        for i in range(len(hidden_groups)):
            in_hidden = out_hidden
            out_hidden = hidden_groups[i]
            self.downsample.append(DownSample1d(in_hidden, out_hidden, res, t_emb, c_emb_dim, seq_len, num_blocks, stride=strides[i], type=down_blocks[i],
                                                residual_butterfly=( i!=(len(hidden_groups)-1) )))
            res = res//2 if strides[i]==2 else res

        i+=1

        
        
        #self.middle = UNet_Middle(hidden_groups[-1], hidden_groups[-1], res, t_emb, c_emb_dim, seq_len, num_blocks=1, middle_cross_attn=middle_cross_attn)
        self.middle = UNet_Middle(hidden_groups[-1], hidden_groups[-1], res, t_emb, c_emb_dim, seq_len, num_blocks=1 if middle_cross_attn else 0, middle_cross_attn=middle_cross_attn)

        
        hidden_groups = np.flip(hidden_groups,0)
        
        out_hiddens = hidden_groups[0]
        for i in range(0, len(hidden_groups)):
            prev_out_hiddens = out_hiddens
            out_hiddens = hidden_groups[i]
            in_hiddens = hidden_groups[min(i + 1, len(hidden_groups) - 1)]
            self.upsample.append(UpSample1d(in_hiddens, out_hiddens, prev_out_hiddens, res, t_emb, c_emb_dim, seq_len, num_blocks_upsample, stride=strides[i], type=up_blocks[i]))
            res = res*2 if strides[i]==2 else res
        
        self.in_conv.apply(init_orth)
        self.out_conv.apply(init_orth)
        
    def forward(self, X, t_emb, c_emb):
        
        X = self.in_conv(X)

        residuals = [X]

        for i, blk in enumerate(self.downsample):
            X, residual = blk(X, t_emb, c_emb)
            residuals += residual
            
        X = self.middle(X, t_emb, c_emb)
        
        
        for i, blk in enumerate(self.upsample):
            res_samples = residuals[-len(blk.residual.residuals) :]
            residuals = residuals[: -len(blk.residual.residuals)]
            
            X = blk(X, t_emb, c_emb, res_samples)

        X = self.out_conv(X)

        return X
