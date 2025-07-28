import torch
from torch.utils.data import Dataset


class CustomSyntheticDataset(Dataset):
    def __init__(self, X, device='cpu'):
        self.device = device
        self.x = torch.from_numpy(X).to(device)
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