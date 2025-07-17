import torch


def min_max_scale(tensor, min_val, max_val):
    return (tensor - min_val) / (max_val - min_val)

def scale_conditions(cond, cond_names, cond_stats):
    """
    Args:
        cond : torch.Tensor of shape (B, K) - raw condition values
        cond_names  : list of K strings - condition names (e.g., ['intensity', 'thickness'])
        cond_stats  : dict - loaded from cond_scaling.json
    Returns:
        scaled_tensor : torch.Tensor of shape (B, K) - normalized to 0–1
    """
    scaled = []
    for k, name in enumerate(cond_names):
        min_val = cond_stats[name]['min']
        max_val = cond_stats[name]['max']
        scaled_val = min_max_scale(cond[:, k], min_val, max_val)
        scaled.append(scaled_val.unsqueeze(1))
    return torch.cat(scaled, dim=1)