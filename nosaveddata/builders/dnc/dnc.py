import torch
import torch.nn.functional as F
from torch import nn

from .memory import DNC_Memory
from ..cfc import *

eps=torch.tensor(1e-6,device='cuda')

def oneplus(X):
    #return 1/(1+torch.exp(X))
    return 1 + torch.log( 1 + torch.exp(X) + eps )




class DNC(nn.Module):
    def __init__(self, W, R, N, M, batch_size, seq_len=8, dropout=0.):
        super().__init__()
        self.W = W
        self.R = R
        self.batch_size = batch_size
        self.is_multiple_heads = R>1
        self.seq_len = seq_len
        
        hparams = {'backbone_activation': 'silu', 'backbone_units': W, 'backbone_layers': 1, 'backbone_dr': dropout}

        self.controller_aux = CfcCell(W, W, seq_len, hparams)
        self.controller = CfcCell(W+R*W, W, seq_len, hparams)
        
        # Create memory
        self.memory = DNC_Memory(N, M, batch_size)
        self.M = M
        self.N = N 
        
        
        # Used together with the controller network
        self.v = nn.Linear(W, W)
        self.r_weights = nn.Linear(W*R, W)
        self.xi = nn.Linear(W, (W*R) + 3*W + 5*R + 3)
        
        
        self.prev_write_weights = torch.zeros(batch_size,N).cuda()
        self.prev_read_weights = torch.zeros(batch_size,R,N).cuda()
        self.prev_reads = torch.zeros(batch_size,R*W).cuda()
        
        # Writing logic
        self.usage = torch.zeros(batch_size,N).cuda()
        self.precedence_w = torch.zeros(batch_size,N).cuda()
        self.link_matrix = torch.zeros(batch_size,N,N).cuda()
    
    def detach_grads(self, memory):
        self.prev_reads = self.prev_reads.detach()
        self.prev_read_weights = self.prev_read_weights.detach()
        self.prev_write_weights = self.prev_write_weights.detach()

        self.usage = self.usage.detach()
        self.precedence_w = self.precedence_w.detach()
        self.link_matrix = self.link_matrix.detach()

        memory.memory = memory.memory.detach()
        #self.memory.memory = self.memory.memory.detach()

    def forward(self, X, hs, memory, reset_idxs):
        # reset_idxs: binary number per batch element indicating wether there is a memory reset (1) or not (0)
        self.detach_grads(memory)

        y, hiddens_states = [], []
        for i in range(X.shape[1]):
            reset = reset_idxs[:,i]
            memory.reset_idx(reset.nonzero().flatten())
            hs = hs*(1-reset[:,None])
            
            hs = self.controller(torch.cat((X[:,i], self.prev_reads), -1), hs, torch.tensor([i]*X.shape[0]).to('cuda'))
            hiddens_states.append(hs)

            xi = self.xi(hs)
            
            self.interface_memory(xi)
            self.get_write_weights(memory)
            mem_read = self.read_memory(memory)
            mem_read = mem_read.view(-1, self.W*self.R)
            
            y.append((self.v(hs) + self.r_weights(mem_read)).unsqueeze(1))
            
            self.prev_reads = mem_read.clone().detach()
            
            
        return torch.stack(y,1).squeeze(2), torch.stack(hiddens_states,1)
    
    
    def interface_memory(self, xi):
        W, R = self.W, self.R
        
        last_pos=0
        next_pos=W*R
        self.read_keys = xi[:,:next_pos].view(self.batch_size,R,W)
        
        last_pos=next_pos
        next_pos+=W
        self.write_key = xi[:,last_pos:next_pos]
        last_pos=next_pos
        next_pos+=W
        self.write_vector = xi[:,last_pos:next_pos]
        last_pos=next_pos
        next_pos+=W
        self.erase_vector = torch.sigmoid(xi[:,last_pos:next_pos])
        
        last_pos=next_pos
        next_pos+=R
        self.free_gates = torch.sigmoid(xi[:,last_pos:next_pos]).view(self.batch_size,R)
        last_pos=next_pos
        next_pos+=R
        self.read_strengths = oneplus(xi[:,last_pos:next_pos])
        last_pos=next_pos
        next_pos+=3*R
        self.read_modes = F.softmax(xi[:,last_pos:next_pos].view(self.batch_size,R,3), -1)#.view(self.batch_size,R*3)
        
        last_pos=next_pos
        next_pos+=1
        self.write_strength = oneplus(xi[:,last_pos:next_pos])
        last_pos=next_pos
        next_pos+=1
        self.allocation_gate = torch.sigmoid(xi[:,last_pos:next_pos])
        last_pos=next_pos
        next_pos+=1
        self.write_gate = torch.sigmoid(xi[:,last_pos:next_pos])
        
        
        
    def get_write_weights(self, memory):
        write_content_addressing = memory.content_addressing(self.write_key, self.write_strength)
        

        retention_vector = torch.prod(1 - self.free_gates.unsqueeze(-1)*self.prev_read_weights, dim=1)
        
        
        self.usage = (self.usage + self.prev_write_weights -self.usage*self.prev_write_weights)*retention_vector
        #print('USAGE', self.usage.shape, retention_vector.shape)
        
        
        #sorted_idx is the free list
        sorted_idx = self.usage.sort().indices
        cumprod = self.usage.sort().values.cumprod(dim=-1)
        
        allocation = (1-self.usage)*cumprod.gather(1,sorted_idx)
        
        write_weights = self.write_gate*(self.allocation_gate*allocation + (1-self.allocation_gate)*write_content_addressing)
        
        
        write_weights=write_weights.unsqueeze(-1)
        self.link_matrix = (1-write_weights-write_weights.transpose(-2,-1))*self.link_matrix+\
                            write_weights*self.precedence_w.unsqueeze(-1).transpose(-2,-1)
        write_weights=write_weights.squeeze(-1)
        
        #Zero diagonal
        mask = torch.ones(self.batch_size,self.N,self.N).cuda()
        diagonal=torch.arange(self.N)
        mask[:,diagonal, diagonal]=0
        self.link_matrix=self.link_matrix*mask
        #print('Link Matrix', self.link_matrix)

        self.precedence_w=(1-write_weights.sum(dim=-1,keepdim=True))*self.precedence_w+write_weights
        
        
        memory.write(write_weights, self.erase_vector, self.write_vector)
        self.prev_write_weights = write_weights.clone()
        
        
        
    
    def read_memory(self, memory):
        read_content_addressing = memory.content_addressing(self.read_keys, self.read_strengths, self.is_multiple_heads)
        
        #print(self.prev_read_weights.shape, self.link_matrix.shape)
        forward_w = self.prev_read_weights@self.link_matrix
        backward_w = self.prev_read_weights@self.link_matrix.transpose(-2,-1)
        #print('All types of reads shapes',forward_w.shape, backward_w.shape, read_content_addressing.shape)
        #print('Read modes', self.read_modes.shape)

        
        aux = torch.cat((backward_w.unsqueeze(2), read_content_addressing.unsqueeze(2), forward_w.unsqueeze(2)),2)
        read_weights = (aux*self.read_modes.unsqueeze(-1)).sum(2)
        #read_weights = backward_w*self.read_modes[:,:,0].unsqueeze(-1) + read_content_addressing*self.read_modes[:,:,1].unsqueeze(-1) + forward_w*self.read_modes[:,:,2].unsqueeze(-1)
        #print(read_weights.shape)
        
        #print(memory.memory.shape, read_weights.shape)
        self.prev_read_weights = read_weights
        return memory.read(read_weights, self.is_multiple_heads)
        
        