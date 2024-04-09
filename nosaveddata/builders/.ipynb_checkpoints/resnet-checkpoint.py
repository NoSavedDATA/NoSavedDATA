import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_init import *
from .transformer import  ConvAttnBlock
from ..nsd_utils.networks import params_count
from ..nsd_utils.save_hypers import nsd_Module

import numpy as np

    

            

class DQN_Conv(nn.Module):
    def __init__(self, in_hiddens, hiddens, ks, stride, padding=0, max_pool=False, norm=True, init=init_relu, act=nn.SiLU(), bias=True):
        super().__init__()
        
        self.conv = nn.Sequential(#nn.Conv2d(in_hiddens, hiddens, ks, stride, padding, padding_mode='replicate'),
                                  nn.Conv2d(in_hiddens, hiddens, ks, stride, padding, bias=bias),
                                  nn.MaxPool2d(3,2,padding=1) if max_pool else nn.Identity(),
                                  (nn.GroupNorm(32, hiddens, eps=1e-6) if hiddens%32==0 else nn.BatchNorm2d(hiddens, eps=1e-6)) if norm else nn.Identity(),
                                  act,
                                  )
        self.conv.apply(init)
        
    def forward(self, X):
        return self.conv(X)
        
class DQN_CNN(nn.Module):
    def __init__(self, in_hiddens, hiddens, ks, stride, padding=0):
        super().__init__()
        
        self.cnn = nn.Sequential(DQN_Conv(4, 32, 8, 4),
                                 DQN_Conv(32, 64, 4, 2),
                                 DQN_Conv(64, 64, 3, 1)
                                )
    def forward(self, X):
        
        return self.cnn(X)



        
class Residual_Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, act=nn.SiLU(), out_act=nn.SiLU(), norm=True, init=init_relu, bias=True):
        super().__init__()
        
        

        conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1,
                                            stride=stride, bias=bias),
                              (nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6)) if norm else nn.Identity(),
                              act)
        conv2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                              (nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6)) if norm else nn.Identity(),
                              out_act)

        conv1.apply(init)
        conv2.apply(init if out_act!=nn.Identity() else init_xavier)
        
        self.conv = nn.Sequential(conv1, conv2)
        
        self.proj=nn.Identity()
        if stride>1 or in_channels!=channels:
            self.proj = nn.Conv2d(in_channels, channels, kernel_size=1,
                        stride=stride)
        
        self.proj.apply(init_proj2d)
        
    def forward(self, X):
        Y = self.conv(X)
        Y = Y+self.proj(X)
        return Y


class ConvNeXt_Block(nn.Module):
    def __init__(self, in_channels, channels, scale=4, stride=1, act=nn.GELU(), norm=True, init=init_relu):
        super().__init__()
        
        

        conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=7, padding=3,
                                            stride=stride),
                              nn.LayerNorm(channels, eps=1e-6) if norm else nn.Identity())
        conv2 = nn.Sequential(nn.Conv2d(channels, channels*scale, kernel_size=1, padding=0),
                              act)
        conv3 = nn.Conv2d(channels*scale, channels, kernel_size=1, padding=0)

        conv1.apply(init_orth)
        conv2.apply(init)
        conv1.apply(init_orth)
        
        self.conv = nn.Sequential(conv1, conv2, conv3)
        
        self.proj=nn.Identity()
        if stride>1 or in_channels!=channels:
            self.proj = nn.Conv2d(in_channels, channels, kernel_size=1,
                        stride=stride)
        
        self.proj.apply(init_proj2d)
        
    def forward(self, X):
        Y = self.conv(X)
        Y = Y+self.proj(X)
        return Y
    
    
class Inverse_Residual_Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, out_act=nn.SiLU()):
        super().__init__()
        self.conv = nn.Sequential(#nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, padding_mode='replicate'),
                                  nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                                  nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6),
                                  nn.SiLU(),
                                  #nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate'),
                                  nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                                  nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6),
                                  out_act,
                                  nn.UpsamplingNearest2d(scale_factor=2))
        
        self.proj=nn.Identity()
        if stride>1:
            if in_channels!=channels:
                self.proj = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                      nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
                self.proj.apply(init_proj2d)
            else:
                self.proj = nn.UpsamplingNearest2d(scale_factor=2)
        
        
        if out_act==nn.Sigmoid():
            self.out_activation=nn.Sigmoid()
            self.conv.apply(init_dreamer_uniform)
            
        else:
            self.conv.apply(init_relu)
        
        
    def forward(self, X):
        Y = self.conv(X)
        Y = Y+self.proj(X)
        return Y


class IMPALA_YY(nsd_Module):
    def __init__(self, first_channels=12, scale_width=1, norm=True, init=init_relu, act=nn.SiLU()):
        super().__init__()

        self.yin = self.get_yin(1, 16*scale_width, 32*scale_width)
        
        self.yang = self.get_yang(first_channels, 16*scale_width)
                                 
        self.head = nn.Sequential(self.get_yang(16*scale_width, 32*scale_width),
                                  self.get_yang(32*scale_width, 32*scale_width, last_relu=True))
        
        params_count(self, 'IMPALA ResNet')

    def get_yin(self, in_hiddens, hiddens, out_hiddens):
        blocks = nn.Sequential(DQN_Conv(in_hiddens, hiddens, 3, 1, 1, max_pool=True, act=self.act, norm=self.norm, init=self.init),
                               Residual_Block(hiddens, hiddens, norm=self.norm, act=self.act, init=self.init),
                               Residual_Block(hiddens, hiddens, norm=self.norm, act=self.act, init=self.init),
                               #DQN_Conv(hiddens, out_hiddens, 3, 1, 1, max_pool=True, act=self.act, norm=self.norm, init=self.init),
                               #Residual_Block(out_hiddens, out_hiddens, norm=self.norm, act=self.act, init=self.init),
                               #Residual_Block(out_hiddens, out_hiddens, norm=self.norm, act=self.act, init=self.init)
                              )
        return blocks          
        
    def get_yang(self, in_hiddens, out_hiddens, last_relu=False):
        
        blocks = nn.Sequential(DQN_Conv(in_hiddens, out_hiddens, 3, 1, 1, max_pool=True, act=self.act, norm=self.norm, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, norm=self.norm, act=self.act, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, norm=self.norm, act=self.act, init=self.init, out_act=self.act if last_relu else nn.Identity())
                              )
        
        return blocks
    
    def forward(self, X):

        y = self.yin(X[:,-3:].mean(-3)[:,None])
        x = self.yang(X)
        
        #X = x*(1-y) + x + y
        X = 0.67*x + 0.33*y
        
        return self.head(X)


class IMPALA_Resnet(nsd_Module):
    def __init__(self, first_channels=12, scale_width=1, norm=True, init=init_relu, act=nn.SiLU(), bias=True):
        super().__init__()
        
        self.cnn = nn.Sequential(self.get_block(first_channels, 16*scale_width),
                                 self.get_block(16*scale_width, 32*scale_width),
                                 self.get_block(32*scale_width, 32*scale_width, last_relu=True))
        params_count(self, 'IMPALA ResNet')
        
    def get_block(self, in_hiddens, out_hiddens, last_relu=False):
        
        blocks = nn.Sequential(DQN_Conv(in_hiddens, out_hiddens, 3, 1, 1, max_pool=True, bias=self.bias, act=self.act, norm=self.norm, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, bias=self.bias, norm=self.norm, act=self.act, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, bias=self.bias, norm=self.norm, act=self.act, init=self.init, out_act=self.act if last_relu else nn.Identity())
                              )
        
        return blocks
        
    def forward(self, X):
        return self.cnn(X)


class IMPALA_Resnet_Whitened(nsd_Module):
    def __init__(self, first_channels=12, scale_width=1, norm=True, init=init_relu, act=nn.SiLU(), bias=True):
        super().__init__()
        # REQUIRES init_whitening_conv WEIGHT INITIALIZATION. This weight init is made over the training distribution. 5000 samples should be ok
        # lhs 2 is because we use concatenate positive and negative eigenvectors, 3 is the kernel size
        self.whitened_channels = 2 * first_channels * 3**2
        
        self.cnn = nn.Sequential(self.whitened_block(first_channels, 16*scale_width),
                                 self.get_block(16*scale_width, 32*scale_width),
                                 self.get_block(32*scale_width, 32*scale_width, last_relu=True))
        self.cnn[0][1].apply(init)
        params_count(self, 'IMPALA ResNet')

    def whitened_block(self, in_hiddens, out_hiddens, last_relu=False):
        
        blocks = nn.Sequential(DQN_Conv(in_hiddens, self.whitened_channels, 3, 1, 1, max_pool=True, bias=self.bias, act=self.act, norm=self.norm, init=self.init),
                               nn.Conv2d(self.whitened_channels, out_hiddens, 1, padding=0, stride=1, bias=self.bias),
                               Residual_Block(out_hiddens, out_hiddens, bias=self.bias, norm=self.norm, act=self.act, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, bias=self.bias, norm=self.norm, act=self.act, init=self.init, out_act=self.act if last_relu else nn.Identity())
                              )
        
        return blocks
    
    def get_block(self, in_hiddens, out_hiddens, last_relu=False):
        
        blocks = nn.Sequential(DQN_Conv(in_hiddens, out_hiddens, 3, 1, 1, max_pool=True, bias=self.bias, act=self.act, norm=self.norm, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, bias=self.bias, norm=self.norm, act=self.act, init=self.init),
                               Residual_Block(out_hiddens, out_hiddens, bias=self.bias, norm=self.norm, act=self.act, init=self.init, out_act=self.act if last_relu else nn.Identity())
                              )
        
        return blocks
        
    def forward(self, X):
        return self.cnn(X)



class IMPALA_ConvNeXt(nsd_Module):
    def __init__(self, first_channels=12, scale_width=1, norm=True, init=init_relu, act=nn.SiLU()):
        super().__init__()
        
        self.cnn = nn.Sequential(self.get_block(first_channels, 16*scale_width),
                                 self.get_block(16*scale_width, 32*scale_width),
                                 self.get_block(32*scale_width, 32*scale_width, last_relu=True))
        params_count(self, 'IMPALA ConvNeXt')
    def get_block(self, in_hiddens, out_hiddens, last_relu=False):
        
        blocks = nn.Sequential(DQN_Conv(in_hiddens, out_hiddens, 3, 1, 1, max_pool=True, act=self.act, norm=self.norm, init=self.init),
                               ConvNeXt_Block(out_hiddens, out_hiddens, norm=self.norm, act=self.act, init=self.init),
                               ConvNeXt_Block(out_hiddens, out_hiddens, norm=self.norm, act=self.act, init=self.init)
                              )
        
        return blocks
        
    def forward(self, X):
        return self.cnn(X)




'''Dreamer V3'''



class Dream_CNN_Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, out_act=nn.SiLU(), num_res_blocks=0):
        super().__init__()
        
        
        
        self.conv = nn.Sequential(#nn.Conv2d(in_channels, channels, kernel_size=4, padding=1, padding_mode='replicate',
                                  nn.Conv2d(in_channels, channels, kernel_size=4, padding=1,
                                            stride=stride),
                                  #nn.GroupNorm(channels//8, channels, eps=1e-6),
                                  nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6),
                                  out_act,
                                  *[Residual_Block(channels, channels, out_act=out_act) for i in range(num_res_blocks)],
                                  )
        

        
        if out_act==nn.Sigmoid() or out_act==nn.Identity():
            self.conv.apply(init_dreamer_uniform)
        else:
            self.conv.apply(init_relu)
        
    def forward(self, X):
        Y = self.conv(X)
        return Y



    
    
class Inverse_Dreamer_Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, out_act=nn.SiLU(), num_res_blocks=0):
        super().__init__()
        self.conv = nn.Sequential(*[Residual_Block(in_channels, in_channels, out_act=nn.SiLU()) for i in range(num_res_blocks)],
                                  nn.UpsamplingNearest2d(scale_factor=2),
                                  #nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=1, padding_mode='replicate'),
                                  nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, stride=1),
                                  nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6),
                                  out_act,
                                  )
        
        if out_act==nn.Sigmoid() or out_act==nn.Identity():
            self.conv.apply(init_dreamer_uniform)
        else:
            self.conv.apply(init_relu)
        
        
    def forward(self, X):
        Y = self.conv(X)
        
        return Y


class Dreamer_Encoder(nn.Module):
    def __init__(self, hiddens=32):
        super().__init__()
    
        self.cnn = nn.Sequential(#nn.Conv2d(3, hiddens*2**0, 4, stride=2, padding=1, padding_mode='replicate'),
                                    nn.Conv2d(3, hiddens*2**0, 4, stride=2, padding=1),
                                    nn.GroupNorm(32, hiddens, eps=1e-6) if hiddens%32==0 else nn.BatchNorm2d(hiddens, eps=1e-6),
                                    nn.SiLU(),
                                    Dream_CNN_Block(hiddens*2**0, hiddens*2**1,  stride=2, num_res_blocks=0),
                                    Dream_CNN_Block(hiddens*2**1, hiddens*2**2, stride=2, num_res_blocks=0),
                                    Dream_CNN_Block(hiddens*2**2, hiddens*2**3, stride=2, num_res_blocks=0),
                                    #Dream_CNN_Block(hiddens*2**3, hiddens*2**4, stride=2, num_res_blocks=0),
                                )
        self.cnn[0].apply(init_relu)
        
    def forward(self, X):
        return self.cnn(X)

class Dreamer_Decoder(nn.Module):
    def __init__(self, hiddens=32):
        super().__init__()

        self.cnn = nn.Sequential(Inverse_Dreamer_Block(hiddens*2**4, hiddens*2**3, num_res_blocks=0),
                                 Inverse_Dreamer_Block(hiddens*2**3, hiddens*2**2, num_res_blocks=0),
                                 Inverse_Dreamer_Block(hiddens*2**2, hiddens*2**1, num_res_blocks=0),
                                 Inverse_Dreamer_Block(hiddens*2**1, 3, out_act=nn.Identity()),
                                )

    def forward(self, X):
        return self.cnn(X)

        
class TS8_Resnet(nn.Module):
    def __init__(self, scale_width=8):
        super().__init__()
        
        self.cnn = nn.Sequential(self.get_block(3,16*scale_width),
                                 self.get_block(16*scale_width,32*scale_width),
                                 self.get_block(32*scale_width,64*scale_width))
    
    def get_block(self, in_hiddens, out_hiddens):
        blocks = nn.Sequential(nn.Conv2d(in_hiddens, out_hiddens, 3, stride=1, padding=1),
                               nn.MaxPool2d(3,2,padding=1),
                               #nn.BatchNorm2d(out_hiddens, eps=1e-6),
                               nn.GroupNorm(32, out_hiddens, eps=1e-6) if out_hiddens%32==0 else nn.BatchNorm2d(out_hiddens, eps=1e-6),
                               nn.SiLU(),
                               Residual_Block(out_hiddens, out_hiddens),
                               Residual_Block(out_hiddens, out_hiddens)
                              )
        blocks[0].apply(init_relu)
        
        return blocks
        
    def forward(self, X):
        return self.cnn(X)



class StableDiffusion_Decoder(nn.Module):
    def __init__(self, z_channels, ch=32, ch_mult=(8,4,2,1), attn_resolutions=(4, 8), num_res_blocks=2, resolution=4, out_ch=3):
        super().__init__()

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[0]
        curr_res = resolution #// 2**(self.num_resolutions-1)
        #self.z_shape = (1,z_channels,curr_res,curr_res)
        

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Residual_Block(block_in,
                                          block_in)
        self.mid.attn_1 = ConvAttnBlock(block_in)
        self.mid.block_2 = Residual_Block(block_in,
                                          block_in)



        # upsampling
        self.up = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(Residual_Block(block_in,
                                            block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(ConvAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = nn.UpsamplingNearest2d(scale_factor=2)
                curr_res = curr_res * 2
            self.up.append(up) # prepend to get consistent order

        # end
        self.conv_out = nn.Sequential(nn.BatchNorm2d(block_in),
                                      nn.SiLU(),
                                      torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
        
        self.conv_in.apply(init_relu)
        self.conv_out.apply(init_xavier)

    def forward(self, z):

        self.last_z_shape = z.shape

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        return self.conv_out(h)