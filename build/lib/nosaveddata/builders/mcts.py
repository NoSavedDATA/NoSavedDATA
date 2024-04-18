import torch
from torch import nn
import torch.nn.functional as F

from ..nsd_utils.save_hypers import nsd_Module

class MCTS_Node(nsd_Module):
    def __init__(self, z, Q, reward, prev_state, n_actions, hiddens=2048):
        super().__init__()
        
        self.transitions = [None]*self.n_actions
        
        self.n = torch.zeros(n_actions, device='cuda')
        
        self.p = F.softmax(Q,-1)
        self.Q = torch.zeros_like(Q)

        self.choosen_action = torch.tensor(-1, device='cuda', dtype=torch.long)
        
        

    
    def get_stats(self):

        return self.Q, self.z, self.p, self.n, self.reward, self.choosen_action
        
    def forward(self, x):

        return x


class MCTS(nsd_Module):
    def __init__(self, n_actions, k=5, c1=1.25, c2=19652, batch_size=32, dirichlet_noise=0.3, n_sim=8):
        super().__init__()
    

    def get_root(self, model, x):
        z, Q = model.get_root(x)
        Q = (Q*model.support).sum(-1)

        nodes = []
        for i in range(Q.shape[0]):
            root = MCTS_Node(z[i], Q[i], torch.tensor([0]*self.batch_size).cuda()[i], prev_state=None, n_actions=self.n_actions)
            root.p = (1-self.dirichlet_noise)*root.p + self.dirichlet_noise*torch.randn_like(root.p)
            nodes.append(root)
        self.root = nodes
        return self.root

    def collate_nodes(self):
        Q, Z, P, N, R, A = [], [], [], [], [], []
        for node in self.cur_state:
            q, z, p, n, r, a = node.get_stats()
            Q.append(q)
            Z.append(z)
            P.append(p)
            N.append(n)
            R.append(r)
            A.append(a)
        return torch.stack(Q,0), torch.stack(Z,0), torch.stack(P,0), torch.stack(N,0), torch.stack(R,0), torch.stack(A,0)
    
    def transition(self, model, x, action):
        
        z, Q, reward_pred = model.transition_one_step(x, action)
        Q = (Q*model.support).sum(-1)
    
        nodes = []
        
        for i in range(Q.shape[0]):
            if self.cur_state[i].transitions[action[i]] == None:
                node = MCTS_Node(z[i], Q[i], reward_pred[i], prev_state=self.cur_state[i], n_actions=self.n_actions)
                nodes.append(node)
                self.cur_state[i].transitions[action[i]] = node
            else:
                nodes.append(self.cur_state[i].transitions[action[i]])
                

        return nodes

    
    def backup(self, model):
        
        Q, z, p, n, r_t, choosen_action = self.collate_nodes()
        
        next_values = model.get_Q_last_state(z)#[:,None]

        rewards = [r_t]
        gammas = torch.ones(self.batch_size, self.k, device='cuda')*0.997
        

        for i in range(len(self.cur_state)):
            self.cur_state[i] = self.cur_state[i].prev_state
            

        for l in range(self.k):
            Q, z, p, n, r_t, choosen_action = self.collate_nodes()
            
            r = torch.stack(list(reversed(rewards)), -1)

            returns = (r*gammas[:,:l+1].cumprod(-1)).sum(-1) + next_values*(gammas[:,:l+1].prod(-1))
            #print(f"backup returns {returns.shape}")
            #print(f"n choosen action {n}\n{choosen_action}\n")
            
            n_action = n[torch.arange(self.batch_size), choosen_action]
            

            Q[torch.arange(self.batch_size), choosen_action] = (n_action*Q[torch.arange(self.batch_size),choosen_action] + returns) / (n_action+1)
            
            n[torch.arange(self.batch_size), choosen_action] += 1

            
            
            rewards.append(r_t)
            
            for i in range(len(self.cur_state)):
                self.cur_state[i].Q = Q[i]
                self.cur_state[i].n = n[i]
                
                self.cur_state[i] = self.cur_state[i].prev_state
                
        
    def forward(self, model, x):
        with torch.no_grad():
            self.cur_state = self.get_root(model, x)
    
    
            for sim in range(self.n_sim):
                actions_to_step = []
                for l in range(self.k):
                    q, z, p, n, _, _ = self.collate_nodes()
                    
                    #print(f"mcts current q, p, n: {q.shape, p.shape, n.shape}")
                    
                    
                    Q = q + p * (n.sum(-1,keepdim=True).sqrt() / (1+n)) * (self.c1 + torch.log( (n.sum(-1,keepdim=True)+self.c2+1 ) / self.c2 ))
                    action = Q.argmax(-1)
                    
                    actions_to_step.append(action)
                    #print(f"{Q}")
                    #print(f"Mcts action: {action}")
                    
                    for i in range(len(self.cur_state)):
                        self.cur_state[i].choosen_action = action[i]
                        
                    self.cur_state = self.transition(model, z, action)
                    
                self.backup(model)
                self.cur_state = self.root
                
            Q = self.collate_nodes()[0]
            return Q, Q.argmax(-1), actions_to_step