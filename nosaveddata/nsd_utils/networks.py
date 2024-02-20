import torch
import torch.nn.functional as F
import numpy as np

import random
import os


def params_count(model, name='Model'):
    params_to_count = [p for p in model.parameters() if p.requires_grad]
    print(f'{name} Parameters: {sum(p.numel() for p in params_to_count)/1e6:.2f}M')


def params_and_grad_norm(model):
    param_norm, grad_norm = 0, 0
    for param in model.parameters():
        param_norm += torch.norm(param.data)
        if param.grad is not None:
            grad_norm += torch.norm(param.grad)
    return param_norm, grad_norm

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