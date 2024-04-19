import torch
from torch import nn
from torch.nn import functional as F


from .weight_init import *
from .mlp import  *
from ..nsd_utils.networks import params_count
from ..nsd_utils.save_hypers import nsd_Module

class Residual_Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, act=nn.SiLU(), out_act=nn.SiLU(), norm=True, init=init_xavier, bias=True):
        super().__init__()
        
        

        conv1 = nn.Sequential(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1,
                                            stride=stride, bias=bias),
                              (nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6)) if norm else nn.Identity(),
                              act)
        conv2 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias),
                              (nn.GroupNorm(32, channels, eps=1e-6) if channels%32==0 else nn.BatchNorm2d(channels, eps=1e-6)) if norm else nn.Identity())

        conv1.apply(init)
        conv2.apply(init if out_act!=nn.Identity() else init_xavier)
        
        self.conv = nn.Sequential(conv1, conv2)
        
        self.proj=nn.Identity()
        if stride>1 or in_channels!=channels:
            self.proj = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1,
                        stride=stride)
        
        self.proj.apply(init_proj2d)
        self.out_act = out_act
        
    def forward(self, X):
        Y = self.conv(X)
        Y = Y+self.proj(X)
        return self.out_act(Y)
    


class EffZ_Perception(nsd_Module):
    def __init__(self, n_actions, scale_width=1, act=nn.ReLU()):
        super().__init__()
        
        
        self.conv1 = nn.Sequential(nn.Conv2d(12, 32, 3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(32),
                                   act)
        self.conv2 = Residual_Block(32, 32, act=act, out_act=act)
        self.conv3 = Residual_Block(32, 64, act=act, out_act=act, stride=2)
        self.conv4 = Residual_Block(64, 64, act=act, out_act=act)
        self.pool1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv5 = Residual_Block(64, 64, act=act, out_act=act)
        self.pool2 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv6 = Residual_Block(64, 64, act=act, out_act=act)
        
        self.conv1.apply(init_xavier)
        
        self.conv = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.pool1,
                                   self.conv5, self.pool2, self.conv6)
        
    def forward(self, X):
        X = self.conv(X)
        return X

class _1conv_residual(nn.Module):
    def __init__(self, hiddens, act=nn.ReLU()):
        super().__init__()
        
        self.net = nn.Sequential(nn.Conv2d(hiddens+1, hiddens, 3, padding=1, bias=False),
                                        nn.BatchNorm2d(hiddens))
        
    def forward(self, x):
        proj = x[:,:-1]
        x = self.net(x)
        
        return x+proj
        
class RewardPred(nsd_Module):
    def __init__(self, in_channels, out_channels, in_hiddens, hiddens, bottleneck=32, out_dim=51, act=nn.ReLU(), k=5):
        super().__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                 nn.BatchNorm2d(out_channels),
                                 act)
        
        self.lstm = nn.LSTMCell(in_hiddens, hiddens)
        self.norm_relu = nn.Sequential(nn.BatchNorm1d(hiddens))
        
        self.mlp = MLP_LayerNorm(hiddens, bottleneck, out_dim, layers=2, in_act=act, init=init_xavier, last_init=init_zeros)
        
    def forward(self, x):
        bs, seq = x.shape[:2]
        
        x = self.conv(x.view(bs*seq, *x.shape[-3:])).view(bs,seq,-1)
        
        ht = torch.zeros(x.shape[0], self.hiddens, device='cuda')
        ct = torch.zeros_like(ht)
        
        hs = []
        for i in range(self.k):
            
            ht, ct = self.lstm(x[:,i], (ht, ct))
            hs.append(ht)
        hs = torch.stack(hs,1)
        
        x = self.mlp(hs)
        
        return x
    
    def transition_one_step(self, x, ht):
        
        x = self.conv(x).view(x.shape[0],-1)
        #print('reward one step', x.shape)
        
        ht, ct = self.lstm(x, ht)
        
        x = self.mlp(ht)
        
        return x, (ht,ct)
        

class ActorCritic(nsd_Module):
    def __init__(self, in_channels, out_channels, in_hiddens, bottleneck=32, out_value=51, out_policy=1, act=nn.ReLU()):
        super().__init__()
        
        self.residual = Residual_Block(in_channels, in_channels, act=self.act, out_act=self.act)
        
        conv_policy = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                 nn.BatchNorm2d(out_channels),
                                 act)
        conv_value  = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                 nn.BatchNorm2d(out_channels),
                                 act)
        
        self.policy = nn.Sequential(conv_policy,
                                    nn.Flatten(-3,-1),
                                   MLP_LayerNorm(in_hiddens, bottleneck, out_policy, layers=2, in_act=act, init=init_xavier, last_init=init_zeros))
        self.value = nn.Sequential(conv_policy,
                                    nn.Flatten(-3,-1),
                                   MLP_LayerNorm(in_hiddens, bottleneck, out_value,  layers=2, in_act=act,
                                                 out_act=nn.Softmax(dim=-1), init=init_xavier, last_init=init_zeros))
        
        
    def forward(self, x):
        bs, seq = x.shape[:2]
        x = self.residual(x.view(-1, *x.shape[-3:]))
        
        logits = self.policy(x).view(bs, seq, -1)
        probs = F.softmax(logits, -1)
        
        value_probs = self.value(x).view(bs, seq, -1)
        
        return logits, probs, value_probs
    
        
    
class EfficientZero(nsd_Module):
    def __init__(self, n_actions, hiddens=512, mlp_layers=1, scale_width=4,
                 n_atoms=51, Vmin=-10, Vmax=10):
        super().__init__()
        self.support = torch.linspace(Vmin, Vmax, n_atoms).cuda()
        
        self.hiddens=hiddens
        self.scale_width=scale_width
        self.act = nn.ReLU()
        
        
        #self.encoder_cnn = IMPALA_Resnet(scale_width=scale_width, norm=False, init=init_xavier, act=self.act)
        self.encoder_cnn = EffZ_Perception(n_actions, scale_width)
        

        self.projection = MLP_LayerNorm(2304*scale_width, hiddens, hiddens*2,
                                        init=init_xavier, last_init=init_xavier, layers=3, in_act=self.act,
                                        add_last_norm=False)
        self.prediction = MLP_LayerNorm(hiddens*2, hiddens, hiddens*2, layers=2,
                                        init=init_xavier, last_init=init_xavier,
                                        in_act=self.act, add_last_norm=False)
        
                                       
            
        self.transition = nn.Sequential(_1conv_residual(64, self.act),
                                        Residual_Block(64, 64, act=self.act, out_act=self.act))
        
        self.reward_pred = RewardPred(64, 16, 16*((96//16)**2), hiddens)
        #self.reward_pred = RewardPred(64, 16, 576, hiddens)

        
        self.ac = ActorCritic(64, 16, 16*((96//16)**2), out_policy=n_actions)
    
        params_count(self, 'Efficient Zero Network')
        
    
    def forward(self, X, y_action):
        z_proj, z = self.encode(X)
        
        
        #q, action = self.q_head(X)
        logits, probs, value_probs = self.ac(z)
        
        z_proj_pred, reward_pred = self.get_transition(z[:,0][:,None], y_action)

        #return q, action, X[:,1:].clone().detach(), z_pred
        return z_proj, z_proj_pred, reward_pred, logits, probs, value_probs
    

    def encode(self, X):
        batch, seq = X.shape[:2]
        self.batch = batch
        self.seq = seq
        X = self.encoder_cnn(X.contiguous().view(self.batch*self.seq, *(X.shape[2:])))
        X = X.contiguous().view(self.batch, self.seq, *X.shape[-3:])
        z = X.clone()
        X = X.flatten(-3,-1)
        X = self.projection(X)
        return X, z
    
    def encode_z(self, X):
        batch, seq = X.shape[:2]
        self.batch = batch
        self.seq = seq
        X = self.encoder_cnn(X.contiguous().view(self.batch*self.seq, *(X.shape[2:])))
        X = X.contiguous().view(self.batch, self.seq, *X.shape[-3:])
        
        return X
    
    def get_zero_ht(self, batch_size):
        ht = torch.zeros(batch_size, self.hiddens, device='cuda')
        ct = torch.zeros_like(ht)
        return (ht, ct)
    
    def transition_one_step(self, z, action, ht):
        z = z.contiguous().view(-1, *z.shape[-3:])
        
        action_one_hot = (
            torch.ones(
                (
                    z.shape[0],
                    z.shape[2],
                    z.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        
        action = (action[:, None, None] * action_one_hot / self.n_actions)[:,None]
        #print('one step', z.shape, action.shape)
        z_pred = torch.cat( (z, action), 1)
        z_pred = self.transition(z_pred)
        
        #print('one step z transition', z.shape)
        
        
        reward_pred, ht = self.reward_pred.transition_one_step(z_pred, ht)
        
        #print('one step z_pred reward_pred', z_pred.shape, reward_pred.shape)

        
        return z_pred, reward_pred, ht
    
    def get_transition(self, z, action):
        z = z.contiguous().view(-1, *z.shape[-3:])
        
        action_one_hot = (
            torch.ones(
                (
                    z.shape[0],
                    5,
                    z.shape[2],
                    z.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        
        action = (action[:, :, None, None] * action_one_hot / self.n_actions)[:,:,None]

        #print('transition full', z.shape, action.shape)
        z_pred = torch.cat( (z, action[:,0]), 1)
        z_pred = self.transition(z_pred)
        
        
        z_preds=[z_pred.clone()]
        

        for k in range(4):
            z_pred = torch.cat( (z_pred, action[:,k+1]), 1)
            z_pred = self.transition(z_pred)
            
            
            z_preds.append(z_pred)
        
        
        z_pred = torch.stack(z_preds,1)
        
        reward_pred = self.reward_pred(z_pred)
        
        #print('transition full z_pred reward_pred', z_pred.shape, reward_pred.shape)

        z_proj_pred = self.projection(z_pred.flatten(-3,-1)).view(self.batch,5,-1)
        z_proj_pred = self.prediction(z_proj_pred)
        
        return z_proj_pred, reward_pred