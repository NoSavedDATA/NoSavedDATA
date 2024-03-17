# REFERENCES
# Bigger Better Faster

import torch


def network_ema(target_network, new_network, alpha=0.5):
    for (param_name, param_target), param_new  in zip(target_network.cuda().named_parameters(), new_network.parameters()):
        if 'ln' in param_name: #layer norm
            param_target.data = param_new.data.clone()
        else:
            param_target.data = alpha * param_target.data + (1 - alpha) * param_new.data.clone()


def renormalize(tensor):
    shape = tensor.shape
    tensor = tensor.view(shape[0], -1)
    max_value,_ = torch.max(tensor, -1, keepdim=True)
    min_value,_ = torch.min(tensor, -1, keepdim=True)
    return ((tensor - min_value) / (max_value - min_value + 1e-5)).view(shape)