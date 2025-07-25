import numpy as np
import torch
import torch.nn as nn
from carefl.nflib.nets import MLP4


class DAGAffineCL(nn.Module):
    
    def __init__(self, dim, cond_idx, trans_idx, net_class=MLP4, nh=24, scale_shift_base=False):
        super().__init__()
        self.dim = dim
        self.cond_idx = cond_idx
        self.trans_idx = trans_idx
        # self.id_idx = [i for i in range(dim) if i not in self.trans_idx]  # unchanged part
        
        in_dim = len(self.cond_idx)
        out_dim = len(self.trans_idx)
        
        if len(self.cond_idx) > 0:
            self.s_cond = net_class(in_dim, out_dim, nh)
            self.t_cond = net_class(in_dim, out_dim, nh)
        else:
            self.s_base = nn.Parameter(torch.randn(1, out_dim), requires_grad=True) if scale_shift_base else None
            self.t_base = nn.Parameter(torch.randn(1, out_dim), requires_grad=True) if scale_shift_base else None

        
    def forward(self, x):
        x_cond = x[:, self.cond_idx] if len(self.cond_idx) > 0 else None
        x_trans = x[:, self.trans_idx]
        
        
        if len(self.cond_idx) > 0:
            s = self.s_cond(x_cond)
            t = self.t_cond(x_cond)
        else:
            s = self.s_base if self.s_base is not None else torch.zeros_like(x_trans)
            t = self.t_base if self.t_base is not None else torch.zeros_like(x_trans)
                
       
        z_trans = x_trans * torch.exp(s) + t
        z = x.clone()
        z[:, self.trans_idx] = z_trans
        log_det = torch.sum(s, dim=1)
        
        return z, log_det
    
    def backward(self, z):
        z_cond = z[:, self.cond_idx] if len(self.cond_idx) > 0 else None
        z_trans = z[:, self.trans_idx]
        
        if len(self.cond_idx) > 0:
            s = self.s_cond(z_cond)
            t = self.t_cond(z_cond)
        else:
            s = self.s_base if self.s_base is not None else torch.zeros_like(z_trans)
            t = self.t_base if self.t_base is not None else torch.zeros_like(z_trans)
            
        x_trans = (z_trans - t) * torch.exp(-s)
        x = z.clone()
        x[:, self.trans_idx] = x_trans
        log_det = -torch.sum(s, dim=1)
        return x, log_det
    
class NormalizingFlow(nn.Module):
    
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        
    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det
    
    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m).to(z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det
    
    
class NormalizingFlowModel(nn.Module):
    
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
        
        
    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det
    
    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det
    
    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs
    
    def log_likelihood(self, x):
        if type(x) is np.ndarray:
            x = torch.tensor(x.astype(np.float32))
            _, prior_logprob, log_det = self.forward(x)
            return (prior_logprob + log_det).cpu().detach().numpy()