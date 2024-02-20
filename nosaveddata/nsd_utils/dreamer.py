# REFERENCES
# Dreamer V3
# ChatGPT for the Returns Normalizer


import torch
import torch.nn.functional as F


def log_with_base(x, base):
    return torch.log(x) / torch.log(torch.tensor(base, dtype=x.dtype))


def symlog_base(x, base=10, scale=1):
    return torch.sign(x)*log_with_base((x.abs()+1), base)*scale
#def symexp_base(x, base=10, scale=1):
#    return torch.sign(x)*(base.pow(x.abs()/scale)-1)


def symlog(x, scale=1):
    return torch.sign(x)*torch.log(x.abs()+1)*scale
def symexp(x, scale=1):
    return torch.sign(x)*(torch.exp(x.abs()/scale)-1)


def two_hot_no_symlog(labels, num_buckets, boundaries, scale=1):
    sl=labels
    buckets=torch.bucketize(sl, boundaries)
    
    twohot=torch.zeros(labels.shape[0], boundaries.shape[0], device='cuda')
    twohot=twohot.view(-1)
    twohot_idx=torch.arange(labels.shape[0], device='cuda')*num_buckets+buckets
    #print(f"{twohot_idx}")

    twohot[twohot_idx]   = (boundaries[buckets-1]-sl).abs() / (boundaries[buckets-1]-boundaries[buckets]).abs()
    twohot[twohot_idx-1] = (boundaries[buckets]-sl).abs() / (boundaries[buckets-1]-boundaries[buckets]).abs()

    twohot=twohot.view(labels.shape[0], num_buckets)

    return twohot

def two_hot_view_no_symlog(labels, num_buckets, boundaries, scale=1):
    labels=labels.clone()
    labels=labels.view(-1)
    
    sl=labels
    buckets=torch.bucketize(sl, boundaries)
    
    
    twohot=torch.zeros(labels.shape[0], boundaries.shape[0], device='cuda')
    twohot=twohot.view(-1)
    twohot_idx=torch.arange(labels.shape[0], device='cuda')*num_buckets+buckets
    #print(f"{twohot_idx}")
    
    twohot[twohot_idx]   = (boundaries[buckets-1]-sl).abs() / (boundaries[buckets-1]-boundaries[buckets]).abs()
    twohot[twohot_idx-1] = (boundaries[buckets]-sl).abs() / (boundaries[buckets-1]-boundaries[buckets]).abs()
    
    twohot=twohot.view(labels.shape[0],num_buckets)
    
    return twohot.contiguous()

def two_hot(labels, num_buckets, boundaries, scale=1):
    sl=symlog(labels,scale)
    buckets=torch.bucketize(sl, boundaries)
    
    twohot=torch.zeros(labels.shape[0], boundaries.shape[0], device='cuda')
    twohot=twohot.view(-1)
    twohot_idx=torch.arange(labels.shape[0], device='cuda')*num_buckets+buckets
    #print(f"{twohot_idx}")

    twohot[twohot_idx]   = (boundaries[buckets-1]-sl).abs() / (boundaries[buckets-1]-boundaries[buckets]).abs()
    twohot[twohot_idx-1] = (boundaries[buckets]-sl).abs() / (boundaries[buckets-1]-boundaries[buckets]).abs()

    twohot=twohot.view(labels.shape[0], num_buckets)

    return twohot


def two_hot_view(labels, num_buckets, boundaries, scale=1):
    labels=labels.clone()
    labels=labels.view(-1)
    
    sl=symlog(labels,scale)
    buckets=torch.bucketize(sl, boundaries)
    
    
    
    twohot=torch.zeros(labels.shape[0], boundaries.shape[0], device='cuda')
    twohot=twohot.view(-1)
    twohot_idx=torch.arange(labels.shape[0], device='cuda')*num_buckets+buckets
    #print(f"{twohot_idx}")
    
    twohot[twohot_idx]   = (boundaries[buckets-1]-sl).abs() / (boundaries[buckets-1]-boundaries[buckets]).abs()
    twohot[twohot_idx-1] = (boundaries[buckets]-sl).abs() / (boundaries[buckets-1]-boundaries[buckets]).abs()
    
    twohot=twohot.view(labels.shape[0],num_buckets)
    
    return twohot.contiguous()




class ReturnsNormalizer:
    def __init__(self, decay_rate):
        self.decay_rate = decay_rate
        self.range_average = None
        self.max=1

    def reset(self):
        self.range_average=None
        
    def update(self, returns):
        
        p5_values = torch.quantile(returns, 0.05)
        p95_values = torch.quantile(returns, 0.95)
        
        self.p5=p5_values
        self.p95=p95_values
        
        if returns.max() > self.max:
            self.max = returns.max()
        
        range_batch = torch.maximum(p95_values.abs() - p5_values.abs(), torch.tensor(1.0))
        
        
        if self.range_average is None:
            self.range_average = range_batch.abs()
        else:
            # Update ema
            self.range_average = (
                self.range_average * self.decay_rate + (1 - self.decay_rate) * range_batch.abs()
            )
        
    def normalize(self, returns, eps=1e-6):
        return returns / (self.range_average + eps)