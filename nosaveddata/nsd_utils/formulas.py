import torch
from torch import nn
import torch.nn.functional as F


def min_norm(x):
    return x-x.min()

def min_max_norm(x):
    return (x-x.min())/x.max()

def mean_std_norm(x):
    return (x-x.mean())/x.std()