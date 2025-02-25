import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import random
import os


def params_count(model, name='Model'):
    params_to_count = [p for p in model.parameters() if p.requires_grad]
    parameters_count = sum(p.numel() for p in params_to_count)        
    print(f'{name} Parameters: {parameters_count/1e6:.2f}M')
    return parameters_count


def params_count_no_embedding(model, name='Model'):
    params_to_count = [
        p for name, module in model.named_modules() 
        if not isinstance(module, nn.Embedding) 
        for p in module.parameters() if p.requires_grad
    ]
    parameters_count = sum(p.numel() for p in params_to_count)
    print(f'{name} Parameters (excluding embeddings): {parameters_count/1e6:.2f}M')
    return parameters_count
def ideal_lr(model):
    n = torch.tensor(params_count_no_embedding(model))
    return [0.003239 - 0.0001395*n.log(), 0.003239 + 0.0001395*n.log()]


def params_and_grad_norm(model):
    param_norm, grad_norm = 0, 0
    for n, param in model.named_parameters():
        if not n.endswith('.bias'):
            param_norm += torch.norm(param.data)
            if param.grad is not None:
                grad_norm += torch.norm(param.grad)
    return param_norm, grad_norm


# From STORM Atari-100k
def seed_np_torch(seed=20001118):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

def statistical_difference(p1, p2, n):
    # order invariant
    
    d=torch.tensor(p1-p2).abs()
    std = 1.65 * math.sqrt((p1*(1-p1) + p2*(1-p2))/n)
    difference = torch.tensor([d-std, d+std])
        
    difference = difference.sort()[0]
    
    return difference


def save_checkpoint(net, model_target, optimizer, sched, step, save_every, path):
    if (step%save_every)==0:
            torch.save({
                    'model_state_dict': net.state_dict(),
                    'model_target_state_dict': model_target.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': sched.state_dict(),
                    'step': step,
                    }, path)

def save_checkpoint_16(net, model_target, optimizer, sched, scaler, step, save_every, path):
    if (step%save_every)==0:
            torch.save({
                    'model_state_dict': net.state_dict(),
                    'model_target_state_dict': model_target.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': sched.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'step': step,
                    }, path)


def load_checkpoint(model, optim, sched, path):
    ckpt = torch.load(path)

    model.load_state_dict(ckpt['model_state_dict'])
    optim.load_state_dict(ckpt['optimizer_state_dict'])
    sched.load_state_dict(ckpt['scheduler_state_dict'])
    step = ckpt['step']

    return model, optim, sched, step

def load_checkpoint_target(model, model_target, optim, sched, path):
    ckpt = torch.load(path)

    model.load_state_dict(ckpt['model_state_dict'])
    model_target.load_state_dict(ckpt['model_target_state_dict'])
    optim.load_state_dict(ckpt['optimizer_state_dict'])
    sched.load_state_dict(ckpt['scheduler_state_dict'])
    step = ckpt['step']

    return model, model_target, optim, sched, step




def load_checkpoint_16(model, optim, sched, scaler, path):
    ckpt = torch.load(path)

    model.load_state_dict(ckpt['model_state_dict'])
    optim.load_state_dict(ckpt['optimizer_state_dict'])
    sched.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    step = ckpt['step']

    return model, optim, sched, scaler, step

def load_checkpoint_16_target(model, model_target, optim, sched, scaler, path):
    ckpt = torch.load(path)

    model.load_state_dict(ckpt['model_state_dict'])
    model_target.load_state_dict(ckpt['model_target_state_dict'])
    optim.load_state_dict(ckpt['optimizer_state_dict'])
    sched.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    step = ckpt['step']

    return model, model_target, optim, sched, scaler, step