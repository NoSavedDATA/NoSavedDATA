import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDIMScheduler

from .weight_init import *

from .unet import sinusoidal_embedding
from .transformer_llama import RMSNorm, FFN_LLaMa





class MLP(nn.Module):
    def __init__(self, in_hiddens=512, med_hiddens=512, out_hiddens=512, layers=1,
                 init=init_relu, in_act=nn.SiLU(), out_act=nn.Identity(),
                 ln_eps=1e-3, last_init=init_xavier, bias=True):
        super().__init__()
        # Special MLP with custom options for non last layer and last layer Linears.

        modules=[]
        self.init=init
        self.last_init=last_init
        
        hiddens=in_hiddens
        _out_hiddens = med_hiddens
        act = in_act
        for l in range(layers):
            last_layer = l==(layers-1)
            if last_layer:
                _out_hiddens = out_hiddens
                act = out_act
            modules.append(nn.Linear(hiddens, _out_hiddens, bias=bias))
            
            modules.append(act)
            hiddens=med_hiddens
        self.mlp=nn.Sequential(*modules)
        #print(self.mlp)

        
        self.init_weights()
        
    def init_weights(self):
        self.mlp.apply(self.init)
        self.mlp[-2].apply(self.last_init)
        
        
    def forward(self,X):
        return self.mlp(X)




class MLP_LayerNorm(nn.Module):
    def __init__(self, in_hiddens=512, med_hiddens=512, out_hiddens=512, layers=1,
                 init=init_relu, in_act=nn.SiLU(), out_act=nn.Identity(),
                 ln_eps=1e-3, last_init=init_relu, add_last_norm=False, bias=True):
        super().__init__()
        # Special MLP with custom options for non last layer and last layer Linears.

        modules=[]
        self.init=init
        self.last_init=last_init
        
        hiddens=in_hiddens
        _out_hiddens = med_hiddens
        act = in_act
        for l in range(layers):
            last_layer = l==(layers-1)
            if last_layer:
                _out_hiddens = out_hiddens
                act = out_act
            modules.append(nn.Linear(hiddens, _out_hiddens, bias=(bias and last_layer and not add_last_norm)))
            
            if not last_layer or add_last_norm:
                modules.append(nn.LayerNorm(_out_hiddens, eps=ln_eps))
            modules.append(act)
            hiddens=med_hiddens
        self.mlp=nn.Sequential(*modules)
        #print(self.mlp)

        self.i=2
        if add_last_norm:
            self.i=3
        
        self.init_weights()
        
    def init_weights(self):
        self.mlp.apply(self.init)
        self.mlp[-self.i].apply(self.last_init)
        
        
    def forward(self,X):
        return self.mlp(X)


class MLP_RMSNorm(nn.Module):
    def __init__(self, in_hiddens=512, med_hiddens=512, out_hiddens=512, layers=1,
                 init=init_relu, in_act=nn.SiLU(), out_act=nn.Identity(),
                 ln_eps=1e-3, last_init=init_relu, add_last_norm=True, bias=True):
        super().__init__()
        # Special MLP with custom options for non last layer and last layer Linears.
        # RMS Norm has no bias, so bias=True by default.

        modules=[]
        self.init=init
        self.last_init=last_init
        
        hiddens=in_hiddens
        _out_hiddens = med_hiddens
        act = in_act
        for i in range(layers):
            last_layer = i==(layers-1)
            if last_layer:
                _out_hiddens = out_hiddens
                act = out_act
            modules.append(nn.Linear(hiddens, _out_hiddens, bias=bias))
            
            
            if (not last_layer) or add_last_norm:
                modules.append(RMSNorm(_out_hiddens, eps=ln_eps))
            modules.append(act)
            hiddens=med_hiddens
        self.mlp=nn.Sequential(*modules)
        #print(self.mlp)

        self.i=2
        if add_last_norm:
            self.i=3
        
        self.init_weights()
        
    def init_weights(self):
        self.mlp.apply(self.init)
        self.mlp[-self.i].apply(self.last_init)
        
        
    def forward(self,X):
        return self.mlp(X)



class Merge(nn.Module):
    def __init__(self, hiddens=512, gate_type='pointwise', merge=True):
        super().__init__()
        # Special MLP with custom options for non last layer and last layer Linears.

        self.feature_mlp = MLP_RMSNorm(hiddens, out_hiddens=hiddens, out_act=nn.SiLU(), last_init=init_relu)

        gate_out = hiddens if gate_type=='pointwise' else 1
        self.gate_mlp = MLP(hiddens, out_hiddens=gate_out, last_init=init_alphastar_special, bias=True)

        self.forward = self.normal_forward if merge else self.ignore_forward
        
    def ignore_forward(self, X, high_level_X):
        return X

    def normal_forward(self, X, high_level_X):
        gate = F.sigmoid(self.gate_mlp(X) + self.gate_mlp(high_level_X))
        
        return self.feature_mlp(X)*gate + (1-gate)*self.feature_mlp(high_level_X)


class VectorResblock(nn.Module):
    def __init__(self, hiddens=512, out_hiddens=512, layers=1, gate_type='pointwise',
                 init=init_relu, in_act=nn.SiLU(), out_act=nn.Identity(),
                 ln_eps=1e-3, last_init=init_relu, add_last_norm=True, merge=True, bias=True):
        super().__init__()
        # Special MLP with custom options for non last layer and last layer Linears.

        modules=[]
        self.layers=layers
        self.init=init
        self.last_init=last_init
        

        self.merge = Merge(hiddens, gate_type, merge)
        

        self.mlp = nn.ModuleList([])
        for l in range(layers-1):
            last_layer = l==layers-2
            self.mlp.append(MLP_RMSNorm(hiddens, out_hiddens=hiddens, bias=bias, out_act=in_act, last_init=init_relu if not last_layer else init_alphastar_special, add_last_norm=True))
            

        self.mlp.append(MLP(hiddens, out_hiddens=out_hiddens, bias=True, out_act=out_act, last_init=init_xavier))
    

    def forward(self, X, high_level_X=None):
        X = self.merge(X, high_level_X)

        shortcut = X

        for i, blk in enumerate(self.mlp):
            if i==self.layers-1:
                break
            X = blk(X)

        X = X + shortcut
        return self.mlp[-1](X), X


class FFN_Resblock(nn.Module):
    def __init__(self, hiddens=512, out_hiddens=512, layers=1, gate_type='pointwise',
                 init=init_relu, in_act=nn.SiLU(), out_act=nn.Identity(),
                 ln_eps=1e-3, last_init=init_relu, add_last_norm=False, merge=True, bias=False):
        super().__init__()
        # Special MLP with custom options for non last layer and last layer Linears.

        modules=[]
        self.layers=layers
        self.init=init
        self.last_init=last_init
        
        self.merge = Merge(hiddens, gate_type, merge)
        

        self.mlp = nn.ModuleList([])
        for l in range(layers-1):
            last_layer = l==layers-2
            
            self.mlp.append(nn.Sequential(RMSNorm(hiddens),
                                          FFN_LLaMa(hiddens, hiddens*4)))

            

        if out_hiddens!=0:
            self.mlp.append(MLP(hiddens, out_hiddens=out_hiddens, bias=True, out_act=out_act, last_init=init_xavier))
            
            for blk in self.mlp[:-1]:
                blk.apply(self._init_weights)
        else:
            self.mlp.append(nn.Identity())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, X, high_level_X=None):
        X = self.merge(X, high_level_X)

        for i, blk in enumerate(self.mlp):
            if i==self.layers-1:
                break
            X = blk(X) + X
            
        return self.mlp[-1](X), X



class TimeSiren(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(TimeSiren, self).__init__()
        # just a fully connected NN with sin activations
        self.lin1 = nn.Linear(input_dim, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x

class MLP_Sieve(nn.Module):
    def __init__(self, noised_dim, in_hiddens=512, med_hiddens=512, layers=4,
                 init=init_relu, in_act=nn.SiLU(), out_act=nn.Identity(),
                 ln_eps=1e-3, last_init=init_xavier, bias=True, concat_hs=True):
        super().__init__()
        # Special MLP with custom options for non last layer and last layer Linears.
        
        self.init=init
        self.last_init=last_init
        self.noised_dim = noised_dim

        self.noised_emb = nn.Sequential(nn.Linear(noised_dim, med_hiddens),
                                   in_act,
                                   nn.Linear(med_hiddens, med_hiddens)
                                  )
        self.t_emb = TimeSiren(1, med_hiddens)


        if concat_hs:
            self.mlp_in = nn.Sequential(nn.Linear(med_hiddens*2+in_hiddens*2, med_hiddens), in_act)
            self.forward = self.forward_hs
        else:
            self.mlp_in = nn.Sequential(nn.Linear(med_hiddens*2+in_hiddens, med_hiddens), in_act)
            self.forward = self.forward_no_hs


        
        self.mlp_meds = nn.ModuleList([])
        for i in range(layers-2):
            self.mlp_meds.append(nn.Sequential(nn.Linear(med_hiddens + noised_dim + 1, med_hiddens), in_act))
        
        self.mlp_out = nn.Sequential(nn.Linear(med_hiddens + noised_dim + 1, noised_dim), out_act)
        
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012)

        self.init_weights()
        

    def init_weights(self):
        self.noised_emb[0].apply(self.init)
        self.mlp_in.apply(self.init)
        self.mlp_meds.apply(self.init)
        self.mlp_out.apply(self.last_init)
    

    def add_noise(self, y, t):
        noise = torch.rand_like(y)
        noised = self.noise_scheduler.add_noise(y, noise, t)

        return noised, noise
        
    def denoise(self, X, inference_timesteps=20, n_samples=1):
        noised = torch.randn(X.shape[0], self.noised_dim, device='cuda', dtype=torch.float)
        #X = X.repeat_interleave(n_samples, 0)

        self.noise_scheduler.set_timesteps(inference_timesteps)
        timesteps = torch.arange(inference_timesteps)*(1000//inference_timesteps)

        for t in timesteps:
            # 1. predict noise model_output
            noise_pred = self(X, noised, torch.tensor([t]*X.shape[0], device='cuda', dtype=torch.float).view(-1,1))#.sample
            #noise_pred = self.noise_scheduler.scale_model_input(noise_pred, t)

            # 2. compute previous image: x_t -> x_t-1
            #image = scheduler.step(model_output, t, image).prev_sample
            noised = self.noise_scheduler.step(
                    noise_pred, t-1, noised,# eta=1, use_clipped_model_output=False,
                ).prev_sample
        return noised



    def forward_hs(self, X, hs, noised, t):
        
        noised_emb = self.noised_emb(noised)
        t_emb = self.t_emb((t/1000).float())
        X = self.mlp_in(torch.cat((hs, X, noised_emb, t_emb), -1))
        for blk in self.mlp_meds:
            X = blk(torch.cat((X/1.41421, noised, t), -1)) + X/1.41421


        return self.mlp_out(torch.cat((X, noised, t), -1))
    
    def forward_no_hs(self, X, noised, t):
        
        noised_emb = self.noised_emb(noised)
        t_emb = self.t_emb((t/1000).float())
        X = X.squeeze(1)

        #print(f"{X.shape, noised_emb.shape, t_emb.shape}")
        X = self.mlp_in(torch.cat((X, noised_emb, t_emb), -1))

        for blk in self.mlp_meds:
            X = blk(torch.cat((X/1.41421, noised, t), -1)) + X/1.41421

        return self.mlp_out(torch.cat((X, noised, t), -1))