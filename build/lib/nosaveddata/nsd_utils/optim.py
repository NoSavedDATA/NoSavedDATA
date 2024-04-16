import torch
import torch.nn
import torch.nn.functional as F

class Lookahead:
    def __init__(self, net, steps, k=5):
        self.k=k
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}
        self.sched = 0.95**k * (torch.arange(steps+1) / steps)**3

    def update(self, net, step):
        decay = self.sched[step].item()
        if step%self.k==0:
            for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
                #ema_param.mul_(decay).lerp_(net_param, 1-decay)
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)
                
    def update_fixed_decay(self, net, decay, step):
        if step%self.k==0:
            for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)