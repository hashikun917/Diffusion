import torch
from torch.utils.data import Dataset
import numpy as np


class CustomSyntheticDataset(Dataset):
    def __init__(self, X, device='cpu'):
        self.device = device
        
        if isinstance(X, torch.Tensor): # いずれXはtensorに統一
            self.x = X.to(device)
        elif isinstance(X, np.ndarray):
            self.x = torch.from_numpy(X).to(device)
        else:
            raise ValueError("X must be a torch.Tensor or np.ndarray")
    
        self.len = self.x.shape[0]
        self.data_dim = self.x.shape[1]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.x[idx]
    
    def get_metadata(self):
        return {
            'n': self.len,
            'data_dim': self.data_dim,
        }