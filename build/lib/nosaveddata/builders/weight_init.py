import torch
import numpy as np
import torch.nn as nn
import math


'''MLP AND LINEARS'''

def init_relu(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.orthogonal_(module.weight, gain=1.41421)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_orth(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.orthogonal_(module.weight, gain=1)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_xavier(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_uniform_(module.weight, gain=1)

        if module.bias is not None:
            nn.init.zeros_(module.bias)
            
def init_xavier_normal(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_normal_(module.weight, gain=1)

        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_zeros(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.zeros_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_sigmoid(module):
    #print(f"The init sigmoid was only tested by the package's author at the CfC.")
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_normal_(module.weight, gain=1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_lecun(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.normal_(module.weight, mean=0.0, std=1.0 / (module.weight.shape[1])**0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_tanh(module):
    #print(f"The init tanh was only tested by the package's author at the CfC.")
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.xavier_normal_(module.weight, gain=1.6667)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_deep_lstm(module):
    # Ref: Sequence to Sequence Learning with Neural Networks
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        nn.init.uniform_(module.weight, -0.08, 0.08)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_alphastar_special(module):
    # Ref: Alphastar
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.005)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def init_emb(module):
    if type(module) == nn.Linear:
        torch.nn.init.normal_(module.weight, std=math.sqrt(1/module.weight.shape[0]))
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias) 
            
    if type(module) == nn.Embedding:
        torch.nn.init.normal_(module.weight, std=math.sqrt(1/module.weight.shape[1]))

def init_saving_variance(module, num_blks):
    
    torch.nn.init.xavier_uniform_(module.weight, gain=torch.tensor(4*num_blks).pow(-1/4))
    if hasattr(module, 'bias'):
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
            

def init_gpt(module):
    #print(f"From init_gpt.\nGpt proj linears should have a special weight initialization not implemented here.")
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #torch.nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #torch.nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
        

def init_proj(module):
    assert not isinstance(module, nn.Conv1d) and not isinstance(module, nn.Conv2d) and not isinstance(module, nn.Conv3d)
    if isinstance(module, nn.Linear):
        nn.init.eye_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)




        
'''CNN'''

def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d or type(module) == nn.Conv3d:
        #nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='SiLU')
        nn.init.orthogonal_(module.weight, 1)
        #nn.init.orthogonal_(module.weight, 1.41421)
        #nn.init.xavier_uniform_(module.weight, 1)
        #nn.init.xavier_uniform_(module.weight, 1.41421)


        if module.bias is not None:
            nn.init.zeros_(module.bias)

def init_partial_dirac(module):
    if type(module) in (nn.Conv2d, nn.Conv1d, nn.Conv3d):
        w = module.weight.data
        
        nn.init.dirac_(module.weight[:w.shape[1]])
        nn.init.xavier_uniform_(module.weight[w.shape[1]:], gain=1)

        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if type(module) == nn.Linear:
        print(f"ERROR: ONLY CONVOLUTIONS ARE SUPPORTED BY THE DIRAC INITIALIZATION.")

def init_dreamer_normal(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.Conv1d or type(module) == nn.Conv3d:

        if type(module)==nn.Linear():
            space = module.weight.shape[1] * module.weight.shape[0]
            in_num = space * module.weight.shape[1]
            out_num = space * module.weight.shape[1]
        else:
            space = module.kernel_size[0] * module.kernel_size[1]
            in_num = space * module.in_channels
            out_num = space * module.out_channels
        
        std = np.sqrt((1/np.mean(np.array([in_num, out_num])))) / 0.87962566103423978
        nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        

        if module.bias is not None:
            nn.init.zeros_(module.bias)
        

def init_dreamer_uniform(m):
    # Same as xavier uniform
    '''
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        #nn.init.orthogonal_(m.weight, 1.41421)
    '''
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        limit = np.sqrt(3 * scale)
        nn.init.uniform_(m.weight.data, a=-limit, b=limit)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    

    
def init_proj2d(module):
    if type(module) in (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d):
        torch.nn.init.dirac_(module.weight, groups=1)
        
        if module.bias is not None:
            nn.init.zeros_(module.bias)


'''WHITENED LAYERS'''

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))
    layer.weight.requires_grad=False