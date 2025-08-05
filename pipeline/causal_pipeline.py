import torch

import numpy as np
from pathlib import Path

from diffusion.diffusion_model.models.model11 import UNet
from diffusion.utils.diffusion_utils import DiffusionUtils
from notears.linear import notears_linear
from carefl.models.carefl2 import CAREFL



class CausalPipeline:
    def __init__(self, config):
        self.config = config # namespace型のconfig
        self.notears = config.notears
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.output_dir = Path(config.output_dir) # その日の日付のディレクトリを自動で作成するようにしたい。保存

        
    def run_causal_discovery(self, X: np.ndarray) -> np.ndarray:
        print("Running causal discovery for meta-data")
        W_est = notears_linear(X, self.notears.lambda1, 'l2', w_threshold=self.notears.w_threshold)
        B_est = (W_est != 0).astype(np.int32)
        
        """
        notears_dir = self.output_dir / 'notears_results'
        notears_dir.mkdir(parents=True, exist_ok=True)
        np.save(notears_dir / 'W_est.npy', W_est)
        np.save(notears_dir / 'B_est.npy', B_est)
        """
        
        return B_est
    
    def train_meta_causal_model(self, X: np.ndarray, B_est: np.ndarray) -> CAREFL:
        print("Training meta-causal model")
        carefl = CAREFL(self.config)
        _ = carefl._train(X, B_est) # これではB_estを再度探索するためrun_causal_discoveryの結果と異なる可能性あり
        self.carefl = carefl
        
        return carefl

    def train_guided_diffusion(self, images: torch.Tensor, metadata: torch.Tensor):
        print("Training guided diffusion model")
        
        # self.denoise_model = denoise_modelとして介入・反事実推論プログラムの入力からdenoise_modelを消しても良い
        # 実装途中
        return
    
    def intervene_generate(self, int_idx, int_val, num_samples: int=100, w: float=0.1):
        print("Intervening and generating samples")
        
        x_intervened = self.carefl.predict_intervention(int_idx, int_val, n_samples=num_samples)
        x_intervened = torch.from_numpy(x_intervened).to(self.device)
        
        utils = DiffusionUtils(timesteps=self.config.diffusion.timesteps)
        imgs = utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, x_intervened, w, self.config.image_data.image_size, num_samples, channels=1)
          
        return imgs
    
    
    def counterfactual_editing(self, images_obs: torch.Tensor, x_obs: torch.Tensor, cf_idx, cf_val, w: float=0.1):
        
        utils = DiffusionUtils(timesteps=self.config.diffusion.timesteps)
        
        # abduction
        noises = utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, x_obs, w, self.config.image_data.image_size, images_obs.shape[0], channels=1, reverse=True, input_img=images_obs)
        
        # action & prediction
        x_cf = self.carefl.predict_counterfactual(x_obs, cf_idx, cf_val)
        # x_cf = torch.from_numpy(x_cf).to(self.device)
        imgs = utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, x_cf, w, self.config.image_data.image_size, images_obs.shape[0], channels=1, reverse=False, noise=noises[-1])
         
        return imgs
        
        
        
        
        
        
        
        
        