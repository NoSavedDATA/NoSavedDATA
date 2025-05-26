import torch
import torch.nn.functional as F
from torch import nn

class DNC_Memory(nn.Module):

    def __init__(self, memory_units, memory_unit_size, batch_size):
        super(DNC_Memory, self).__init__()
        
        """ N = No. of memory units (rows) """
        self.n = memory_units
        """ M = Size of each memory unit (cols)"""
        self.m = memory_unit_size
        
        # Define the memory matrix of shape (batch_size, N, M)
        self.memory = torch.zeros([batch_size, self.n, self.m]).cuda()
        
        # Layer to learn initial values for memory reset
        # self.memory_bias_fc = nn.Linear(1, self.n * self.m)
        
        # Reset/Initialize
        self.reset(batch_size)
        
    def read(self, weights, is_multiple_heads=False):
        '''Returns a read vector using the attention weights
        Args:
            weights (tensor): attention weights (batch_size, N)
        Returns:
            (tensor): read vector (batch_size, M)
        '''
        if not is_multiple_heads:
            weights=weights.unsqueeze(1)
        read_vec = torch.matmul(weights, self.memory).squeeze(1)
        
        #print('read vec', read_vec.shape, self.memory.shape)
        return read_vec
    
    def write(self, weights, erase_vec, add_vec):
        '''Erases and Writes a new memory matrix
        Args:
            weights (tensor): attention weights (batch_size, N)
            erase_vec (tensor): erase vector (batch_size, M)
            add_vec (tensor): add vector (batch_size, M)
        '''
        #print('write_weights', weights.shape, self.memory.shape)
        # Erase
        memory_erased = self.memory * (1 - weights.unsqueeze(2) * erase_vec.unsqueeze(1))
        # Add
        self.memory = memory_erased + (weights.unsqueeze(2) * add_vec.unsqueeze(1))
        
    def content_addressing(self, query, beta, is_multiple_heads=False):
        '''Performs content addressing and returns the content_weights
        Args:
            query (tensor): query vector (batch_size, M)
            beta (tensor): query strength scalar (batch_size, 1)
        Returns:
            (tensor): content weights (batch_size, N)
        '''
        if not is_multiple_heads:
            # Compare query with every location in memory using cosine similarity
            similarity_scores = F.cosine_similarity(query.unsqueeze(1), self.memory, dim=2)
            # Apply softmax over the product of beta and similarity scores
            content_weights = F.softmax(beta * similarity_scores, dim=1)
        else:
            batch_size, num_heads = beta.shape
            
            similarity_scores = F.cosine_similarity(query.unsqueeze(2), self.memory.unsqueeze(1), dim=-1)
            content_weights = F.softmax(beta.unsqueeze(-1) * similarity_scores, dim=1)
            #content_weights = content_weights.view(batch_size,num_heads,-1)
        
        return content_weights

    def reset(self, batch_size):
        '''Reset/initialize the memory'''
        # Parametric Initialization
        # in_data = torch.tensor([[0.]]) # dummy input
        # Generate initial memory values
        # memory_bias = torch.sigmoid(self.memory_bias_fc(in_data))
        # Push it to memory matrix
        # self.memory = memory_bias.view(self.n, self.m).repeat(batch_size, 1, 1)
        
        # Uniform Initialization of 1e-6
        #self.memory = torch.Tensor().new_full((batch_size, self.n, self.m), 1e-6).cuda()
        self.memory = torch.zeros(batch_size, self.n, self.m, device='cuda')
    
    def reset_idx(self, memory_reset_idx):
        '''Reset/initialize the memory at a given batch index'''
        # Parametric Initialization
        # in_data = torch.tensor([[0.]]) # dummy input
        # Generate initial memory values
        # memory_bias = torch.sigmoid(self.memory_bias_fc(in_data))
        # Push it to memory matrix
        # self.memory = memory_bias.view(self.n, self.m).repeat(batch_size, 1, 1)
        

        #self.memory[memory_reset_idx] *= torch.zeros(self.n, self.m, device='cuda')
        zero_pos = torch.ones_like(self.memory)
        zero_pos[memory_reset_idx] *= 0
        self.memory = self.memory * zero_pos
        
    def detach_memory(self):
        self.memory = self.memory.detach()
        
        #It has so many comments that it feels like this code was copied from the github.
        #Adaptations were made to use batch size and multiple read heads.