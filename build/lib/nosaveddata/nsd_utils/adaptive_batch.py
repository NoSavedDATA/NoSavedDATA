import torch
from torch import nn
import torch.nn.functional as F

import random


class AdaptiveBatchRecorder:
    def __init__(self, model, group_by=3, avg_by=4, alpha=1.1, beta=3, min_steps=10, max_steps=2000):
        self.acc_grads = {}
        
        self.group_by = max(group_by,1)
        self.avg_by = avg_by

        self.alpha = alpha
        self.beta = beta

        self.steps_since_update = 0
        self.max_steps = max_steps
        self.min_steps = min_steps


        self.groups = {}
        self.group_deltas = {}
        self.tgt_group = None

        self.start_groups(model)

    
    def start_groups(self, model):

        for name, param in model.named_parameters():
            self.groups[self.param_group_name(name)] = []
            self.group_deltas[self.param_group_name(name)] = [torch.tensor(0,device='cuda', dtype=torch.float)]


        keys = list(self.groups.keys())
        self.tgt_group = random.choice(keys)
        print(f"Adaptive batch recorder has {len(keys)} groups.")


    def gumbel_noise(self, n):
        return -(-(torch.rand(n,device='cuda')).log()).log()

    def param_group_name(self,s):
        return '.'.join(s.split('.')[:-self.group_by])


    def angle_between(self, x, y):
        norms = (x.pow(2).sum(-1).sqrt() * y.pow(2).sum(-1).sqrt()).clamp(min=1e-8)
        dot_product = (x * y).sum(-1)
        cosine = (dot_product / norms).clamp(-1.0, 1.0)
        angle = torch.arccos(cosine)

        return angle.mean()*57.3

     
    def sample_group(self):
        v = torch.stack(self.groups[self.tgt_group])

        a_min, a_max = v.min(), v.max()

        self.group_deltas[self.tgt_group].append(a_max - a_min)
        self.group_deltas[self.tgt_group] = self.group_deltas[self.tgt_group][-self.avg_by:]

        # print(f"{self.group_deltas}")

        a_mean = torch.stack([torch.stack(v).mean() for v in self.group_deltas.values()])

        # print(f"A MEAN: {a_mean}")

        a_star = a_mean + self.gumbel_noise(a_mean.shape[-1])

        logits = a_star.abs().pow(self.beta)

        p = logits / logits.sum()


        sample = torch.multinomial(p,1)

        self.groups[self.tgt_group] = []
        self.tgt_group = list(self.groups.keys())[sample]

        # print(f"new group is: {self.tgt_group}")

    def __call__(self, model): # After loss.backward()

        if self.steps_since_update > 0:
            # print(f"append []")
            self.groups[self.tgt_group].append([])
        # print(f"TGT GROUP: {self.tgt_group}")

        for name, param in model.named_parameters():
            group = self.param_group_name(name)
            # print(f"name: {name}")

            if param.grad is not None:
                if name in self.acc_grads.keys():
                    

                    if group==self.tgt_group:
                        param_pre = self.acc_grads[name]

                        self.acc_grads[name] += param.grad
                        grad_angle = self.angle_between(param_pre, self.acc_grads[name])
                        
                        self.groups[group][-1].append(grad_angle)


                    

                else:
                    self.acc_grads[name]  = param.grad
            else:
              print(f"grad of {name} is none")


        # self.groups[self.tgt_group] = self.groups[self.tgt_group][-self.avg_by:]

        self.steps_since_update += 1

        if self.steps_since_update>1:
            self.groups[self.tgt_group][-1] = torch.stack(self.groups[self.tgt_group][-1]).mean() # mean group

            min_a = torch.stack(self.groups[self.tgt_group]).min()

            # print(f"{self.groups[self.tgt_group]}")
            # print(f"{self.groups[self.tgt_group][-1]}")
            # print(f"{min_a*self.alpha}")


            if (self.groups[self.tgt_group][-1] > min_a*self.alpha and self.steps_since_update > self.min_steps) or self.steps_since_update > self.max_steps:
                steps = self.steps_since_update
                self.sample_group()
                self.overwrite_grads(model)
                return True, steps
        return False, self.steps_since_update
     

    def overwrite_grads(self, model):
        for name, param in model.named_parameters():
            param.grad = self.acc_grads[name] / self.steps_since_update

        self.steps_since_update = 1