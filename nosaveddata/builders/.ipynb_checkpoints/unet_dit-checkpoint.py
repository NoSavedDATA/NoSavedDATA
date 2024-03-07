# REFERENCES
# https://github.com/facebookresearch/DiT/blob/main/models.py

from .weight_init import *

from .mlp import MLP
from .transformer import *



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT_FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, np.prod(patch) * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.linear.apply(init_gpt)
        self.adaLN_modulation.apply(init_zeros)
        
        

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

class UNet_DiT(nn.Module):
    def __init__(self, in_channels, d_model, num_blks, nhead, patch=(2,2), img_size=(32,32),
                             dropout = 0.1, bias=False, report_params_count=True,
                             ffn_mult=4):
        super().__init__()
        self.first_channel=in_channels
        self.patches = np.prod(patch)
        self.img_size=img_size
        self.N = int(np.prod(img_size)/self.patches)
        
        self.ts = TimestepEmbedder(d_model)
        
        self.in_proj = MLP(in_channels*self.patches, out_hiddens=d_model, last_init=init_xavier)
        
        self.dit =  DiT_Transformer(d_model, num_blks, nhead, self.patches,
                             dropout = 0.1, bias=False, report_params_count=True,
                             ffn_mult=4)
        self.final_layer = DiT_FinalLayer(d_model, patch, in_channels)
        
        self.init_weights()
    
    def init_weights(self):
        # Zero-out output layers:
        self.final_layer.adaLN_modulation[-1].apply(init_zeros)
        self.final_layer.linear.apply(init_zeros)
    
    def patchify(self, X):
        X = X.view(-1, self.patches*self.first_channel, self.N).transpose(-2,-1)
        return X
    def depatchify(self, X):
        X = X.transpose(-2,-1).contiguous().view(-1, self.first_channel,*self.img_size)
        return X
    
    def forward(self, x, t):
        c = self.ts(t)
        
        x = self.patchify(x)
        x = self.in_proj(x)
        
        x = self.dit(x, c)
        
        x = self.final_layer(x, c)
        x = self.depatchify(x)
        
        return x
    

# Reports for 400k steps and no guidance

def UNet_DiT_S_4(**kwargs):
    # FID 100.41
    # GFlop 1.41
    # Params 33
    return UNet_DiT(d_model=384, num_blks=12, patch=(4,4), nhead=6, **kwargs)

def UNet_DiT_S_2(**kwargs):
    # FID 68.4
    # GFlop 6.06
    # Params 33
    return UNet_DiT(d_model=384, num_blks=12, patch=(2,2), nhead=6, **kwargs)

def UNet_DiT_B_4(**kwargs):
    # FID 68.38
    # GFlop 5.56
    # Params 130
    return UNet_DiT(d_model=768, num_blks=12, patch=(4,4), nhead=12, **kwargs)

def UNet_DiT_B_2(**kwargs):
    # FID 43.47
    # GFlop 23.01
    # Params 130
    return UNet_DiT(d_model=768, num_blks=12, patch=(2,2), nhead=12, **kwargs)

def UNet_DiT_L_4(**kwargs):
    # FID 45.64
    # GFlop 19.7
    # Params 458
    return UNet_DiT(d_model=1024, num_blks=24, patch=(4,4), nhead=16, **kwargs)

def UNet_DiT_L_2(**kwargs):
    # FID 23.33
    # GFlop 80.71
    # Params 458
    return UNet_DiT(d_model=1024, num_blks=24, patch=(2,2), nhead=16, **kwargs)

def UNet_DiT_XL_2(**kwargs):
    # FID 25.21
    # GFlop 118.64
    # Params 675
    return UNet_DiT(d_model=1152, num_blks=28, patch=(2,2), nhead=16, **kwargs)
