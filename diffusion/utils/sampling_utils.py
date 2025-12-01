 from diffusion.utils.utils import load_json
from torch.nn import functional as F

def get_models_functions(config, anti_causal_predictors):
    
    dataset = config.image_data.name
    causal_graph = load_json(config.image_data.meta_data.graph_path)
    
    def cond_fn(x, t, cond):
        with torch.enable_grad():
            
            x = x.detach().requires_grad_(True)
            
            if dataset == "morphomnist":
                
                attrs = causal_graph.keys()
                
                grad = 0
                for key, clfs in anti_causal_predictors.items():
                    
                    attr_idx = attrs.index(key)
                    parents = causal_graph[key]
                    parents_idx = [attrs.index(parent) for parent in parents] if parents else None
                    
                    target = cond[:, attr_idx]
                    
                    if parents:
                        pred = clfs(x, t, cond[:, parents_idx]) # cond[:, parents_idx]は(B, len(parents))のshapeを想定
                    else:
                        pred = clfs(x, t) # (B, 1)を想定
                    
                    mse = F.mse_loss(pred, target, reduction='none') # (B, 1)
                    log_prob = -0.5 * mse.sum(dim=1) # (B,)
                    
                    grad += torch.autograd.grad(log_prob.sum(), x)[0]
                    
        return grad
    
    return cond_fn