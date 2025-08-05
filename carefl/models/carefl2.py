import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
import igraph as ig
from notears.utils import is_dag
from tqdm.auto import tqdm
from carefl.nflib.flows import DAGAffineCL, NormalizingFlowModel
from carefl.nflib.nets import MLP4
from carefl.data.generate_synth_data import CustomSyntheticDataset


class CAREFL:
    """
    CusalPiplineのyaml用CAREFL
    """
    def __init__(self, config):
        self.config = config
        self.meta_data = self.config.image_data.meta_data
        self.notears = self.config.notears
        self.carefl = self.config.carefl
        self.training = self.config.carefl_training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.flow = None


    def _get_datasets(self, X):
        
        if isinstance(X, torch.Tensor):
            dset = CustomSyntheticDataset(X, self.device)
        elif isinstance(X, np.ndarray):
            dset = CustomSyntheticDataset(X.astype(np.float32), self.device)
        else:
            raise ValueError("X must be a torch.Tensor or np.ndarray")
        
        return dset
    
    
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
    
        
    def predict_counterfactual(self, x_obs, cf_idx, cf_val):
        
        if isinstance(cf_idx, (int, np.integer)):
            cf_idx = [cf_idx]
        if np.isscalar(cf_val):
            cf_val = [float(cf_val)]
        assert len(cf_idx) == len(cf_val)

        cf_idx = list(cf_idx)
        cf_val = list(cf_val)
        device = self.device
        
        flows = self.flow.flow.flows
        
        # Abduction: 観測変数に対応する潜在変数を推定
        # x_obs = torch.from_numpy(x_obs.astype(np.float32)).to(device)
        z = self.flow.forward(x_obs)[0][-1]
        
        # Action & Prediction: 介入による因果モデルの変更と観測に対応する潜在変数を用いた推論
        for affine in flows[::-1]:
            trans_idx = affine.trans_idx[0]
            if trans_idx in cf_idx:
                z[:, trans_idx] = cf_val[cf_idx.index(trans_idx)]
            else:
                z, _ = affine.backward(z)
        
        return z


    def _get_flow_arch(self, B_est):
        
        
        d = self.meta_data.d
        
        if self.carefl.prior_dist == 'normal':
            prior = MultivariateNormal(torch.zeros(d).to(self.device), torch.eye(d).to(self.device))
        else:
            raise ValueError(f'Prior distribution {self.carefl.prior_dist} not supported')
        
        if self.config.carefl.net_class == 'mlp4':
            net_class = MLP4
        else:
            raise ValueError(f'Net class {self.carefl.net_class} not supported')
        
        assert is_dag(B_est)
        G = ig.Graph.Adjacency(B_est.tolist(), mode='directed')
        
        ordered_vertices = G.topological_sorting()
        flow_list = []
        for v in ordered_vertices[::-1]:
            cond_idx = G.neighbors(v, mode=ig.IN)
            affine = DAGAffineCL(d, cond_idx, [v], net_class, self.carefl.nh, self.carefl.scale_shift_base, self.carefl.inverse_model)
            flow_list.append(affine)
        
        flow = NormalizingFlowModel(prior, flow_list).to(self.device)
        
        return flow
            
    
    def _train(self, X, B_est):
        
        dset = self._get_datasets(X)
        train_loader = DataLoader(dset, shuffle=True, batch_size=self.config.carefl_training.batch_size)

        flow = self._get_flow_arch(B_est)
        flow.train()
        
        optimizer = optim.Adam(flow.parameters(), lr=1e-3)
        
        loss_vals = []
        for e in tqdm(range(self.config.carefl_training.epochs)):
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