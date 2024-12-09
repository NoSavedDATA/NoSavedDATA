import torch
import torch.nn as nn 



class FiLM_2dBlock(nn.Module):
    def __init__(self, in_hiddens=512, out_hiddens=512):
        super(FiLM_2dBlock, self).__init__()
        
        self.film_mult = nn.Linear(in_hiddens, out_hiddens)
        self.film_add = nn.Linear(in_hiddens, out_hiddens)
        
        nn.init.zeros_(self.film_mult.weight)
        nn.init.zeros_(self.film_add.weight)
        
    def forward(self, X, condition):
        beta = self.film_mult(condition)
        gamma = self.film_add(condition)
        
        #print(X.shape, beta.shape, gamma.shape)
        beta  = beta.view(X.size(0), X.size(1), 1, 1)
        gamma = gamma.view(X.size(0), X.size(1), 1, 1) #B, C, 1, 1
        
        X = (1 + gamma)*X + beta
        
        return X
        