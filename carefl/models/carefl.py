import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
import igraph as ig
from notears.linear import notears_linear
from notears.utils import simulate_dag, simulate_parameter, simulate_linear_sem, simulate_nonlinear_sem, tanh_sem_jacobian, is_dag
from tqdm.auto import tqdm
from carefl.nflib.flows import DAGAffineCL, NormalizingFlowModel
from carefl.nflib.nets import MLP4
from carefl.data.generate_synth_data import CustomSyntheticDataset


class CAREFL:
    def __init__(self, config):
        self.config = config
        self.meta_data = self.config.meta_data
        self.notears = self.config.notears
        self.carefl = self.config.carefl
        self.training = self.config.training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.flow = None


    def _simulate_sem(self):
        
        d = self.meta_data.d
        n = self.meta_data.n_samples
        s0 = self.meta_data.s0
        graph_type = self.meta_data.graph_type
        
        
        if self.meta_data.causal_mech == 'linear':
            self.B_true = simulate_dag(d, s0, graph_type)
            self.W = simulate_parameter(self.B_true)
            X = simulate_linear_sem(self.W, n, self.meta_data.noise_dist)
        else:
            if self.meta_data.gam_scale is None:
                raise ValueError('gamma scale must be specified for nonlinear SEM')
            gamma = np.ones(d) * self.meta_data.gam_scale
            self.B_true = simulate_dag(d, s0, graph_type)
            W_ = simulate_parameter(self.B_true)
            self.W = tanh_sem_jacobian(gamma, W_)
            X = simulate_nonlinear_sem(W_, gamma, n, self.meta_data.causal_mech)

        return X


    def _get_datasets(self, X):

        dset = CustomSyntheticDataset(X.astype(np.float32), self.device)
        
        return dset
    
    def _notears_linear(self, X):
        
        W_est = notears_linear(X, self.notears.lambda1, 'l2', w_threshold=self.notears.w_threshold)
        B_est = (W_est != 0).astype(np.int32)
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
        return B_est
    
    def predict_intervention(self, int_idx, int_val, n_samples=100):
    
        if isinstance(int_idx, (int, np.integer)):
            int_idx = [int_idx]
        if np.isscalar(int_val):
            int_val = [float(int_val)]
        assert len(int_idx) == len(int_val)

        int_idx = list(int_idx)
        int_val = list(int_val)
        device = self.device
        
        flows = self.flow.flow.flows
        z = self.flow.prior.sample((n_samples,)).to(device)
        for affine in flows[::-1]:
            trans_idx = affine.trans_idx[0]
            if trans_idx in int_idx:
                z[:, trans_idx] = int_val[int_idx.index(trans_idx)] # 介入値を強制的に設定
            else:
                z, _ = affine.backward(z)
                
        return z
    
        
    def predict_counterfactual(self, ):
        return 


    def _get_flow_arch(self):
        
        d = self.meta_data.d
        
        if self.carefl.prior_dist == 'normal':
            prior = MultivariateNormal(torch.zeros(d).to(self.device), torch.eye(d).to(self.device))
        else:
            raise ValueError(f'Prior distribution {self.carefl.prior_dist} not supported')
        
        if self.config.carefl.net_class == 'mlp4':
            net_class = MLP4
        else:
            raise ValueError(f'Net class {self.carefl.net_class} not supported')
        
        if self.config.notears.use_notears:
            self.B_est = self._notears_linear(self.X)
            G = ig.Graph.Adjacency(self.B_est.tolist(), mode='directed')
        else:
            G = ig.Graph.Adjacency(self.B_true.tolist(), mode='directed')
        
        ordered_vertices = G.topological_sorting()
        flow_list = []
        for v in ordered_vertices[::-1]:
        #for v in ordered_vertices:
            cond_idx = G.neighbors(v, mode=ig.IN)
            affine = DAGAffineCL(d, cond_idx, [v], net_class, self.carefl.nh, self.carefl.scale_shift_base)
            flow_list.append(affine)
        
        flow = NormalizingFlowModel(prior, flow_list).to(self.device)
        
        return flow
            


    def _train(self):
        
        self.X = self._simulate_sem()
        dset = self._get_datasets(self.X)
        train_loader = DataLoader(dset, shuffle=True, batch_size=self.training.batch_size)

        flow = self._get_flow_arch()
        flow.train()
        
        optimizer = optim.Adam(flow.parameters(), lr=1e-3)
        
        loss_vals = []
        for e in tqdm(range(self.training.epochs)):
            loss_val = 0.0
            for _, x in enumerate(train_loader):
                x = x.to(self.device)
                _, prior_logprob, log_det = flow(x)
                loss = - torch.sum(prior_logprob + log_det)
                loss_val += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            loss_vals.append(loss_val) # バッチごとに割らないでよいのか？？？
            
        self.flow = flow
        return flow, loss_vals
        
        
    def _forward_flow(self, data):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.forward(torch.tensor(data.astype(np.float32)).to(self.device))[0][-1].detach().cpu().numpy()

    def _backward_flow(self, latent):
        if self.flow is None:
            raise ValueError('Model needs to be fitted first')
        return self.flow.backward(torch.tensor(latent.astype(np.float32)).to(self.device))[0][-1].detach().cpu().numpy()