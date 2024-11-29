import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

#scheduler = WarmUpLR(optimizer, warmup_steps=0, min_lr=js['lr_rl']*0.05, max_lr=js['lr_rl'],
#                     after_scheduler_steps=1e6)

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, min_lr, max_lr, after_scheduler_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.after_scheduler_steps = after_scheduler_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.T_max = warmup_steps + after_scheduler_steps
        
        super().__init__(optimizer, last_epoch=last_epoch)
        
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.min_lr + (self.max_lr - self.min_lr) * 
                        (self.last_epoch / self.warmup_steps) 
                        for base_lr in self.base_lrs]
        elif self.last_epoch < (self.T_max):
            return [self.min_lr + (self.max_lr - self.min_lr) *
                (1 + math.cos(math.pi * (self.last_epoch-self.warmup_steps) / self.after_scheduler_steps)) / 2
                for base_lr in self.base_lrs]
        return [self.min_lr for base_lr in self.base_lrs]



class WarmUp_Linear(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, min_lr, max_lr, after_scheduler_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.after_scheduler_steps = after_scheduler_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.T_max = warmup_steps + after_scheduler_steps
        
        super().__init__(optimizer, last_epoch=last_epoch)
        
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.min_lr + (self.max_lr - self.min_lr) * 
                        (self.last_epoch / self.warmup_steps) 
                        for base_lr in self.base_lrs]
                        
        return [self.min_lr for base_lr in self.base_lrs]
    
    
    """
    def step(self):
        super(WarmUpLR, self).step()
    
    
    def step(self):
        print(self.last_epoch)
        if self.last_epoch < self.warmup_steps:
            super(WarmUpLR, self).step()
        elif self.last_epoch < (self.warmup_steps+self.after_scheduler_steps):
            self.after_scheduler.step()
        else:    
            super(WarmUpLR, self).step()
    """
    

def Triangle_Scheduler(optimizer, steps, start=0.2, end=0.07, peak=0.23):
    def triangle(steps, start, end, peak):
        xp = torch.tensor([0, int(peak * steps), steps])
        fp = torch.tensor([start, 1, end])
        x = torch.arange(1+steps)
        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])
        indices = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indices = torch.clamp(indices, 0, len(m) - 1)
        return m[indices] * x + b[indices]
    lr_schedule = triangle(steps, start, end, peak)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: lr_schedule[i])


    
class Sophia_WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, min_lr, max_lr, min_rho, max_rho, after_scheduler_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.after_scheduler_steps = after_scheduler_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_rho = min_rho
        self.max_rho = max_rho
        self.T_max = warmup_steps + after_scheduler_steps
        
        super().__init__(optimizer, last_epoch=last_epoch)
        
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [(self.min_lr + (self.max_lr - self.min_lr) * #lr
                     (self.last_epoch / self.warmup_steps),
                     (self.min_rho + (self.max_rho - self.min_rho) * #rho
                     (self.last_epoch / self.warmup_steps)))
                    for base_lr in self.base_lrs]
        elif self.last_epoch < self.T_max:
            return [(self.min_lr + (self.max_lr - self.min_lr) *
                     (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / self.after_scheduler_steps)) / 2, #lr
                     self.min_rho + (self.max_rho - self.min_rho) *
                     (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / self.after_scheduler_steps)) / 2) #rho
                    for base_lr in self.base_lrs]
        return [(self.min_lr, self.min_rho) for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        
        
        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                #warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()
        
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, (lr, rho) = data
            param_group['lr'] = lr
            param_group['rho'] = rho
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
        

            
            
            
            
class _enable_get_lr_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False