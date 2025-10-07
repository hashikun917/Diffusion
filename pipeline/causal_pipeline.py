import torch

import numpy as np
from pathlib import Path

from diffusion.diffusion_model.models.model11 import UNet
from diffusion.utils.diffusion_utils import DiffusionUtils
from notears.linear import notears_linear
from notears.nonlinear import NotearsMLP, NotearsSobolev
from notears.nonlinear import notears_nonlinear
from carefl.models.carefl2 import CAREFL

from diffusion.utils.utils import load_json

from diffusion.utils.utils import scale_image_0_1_to_0_255


class CausalPipeline:
    def __init__(self, config):
        self.config = config # namespace型のconfig
        self.notears = config.notears
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.output_dir = Path(config.output_dir) # その日の日付のディレクトリを自動で作成するようにしたい。保存

        
    def run_causal_discovery(self, X: np.ndarray) -> np.ndarray:
        print("Running causal discovery for meta-data")
        
        d = X.shape[1]
        
        if self.notears.method == 'linear':
            W_est = notears_linear(X, self.notears.lambda1, 'l2', w_threshold=self.notears.w_threshold)
        elif self.notears.method == 'mlp':
            model = NotearsMLP(dims=[d, 10, 1], bias=True)
            W_est = notears_nonlinear(model, X, lambda1=self.notears.lambda1, lambda2=self.notears.lambda1, w_threshold=self.notears.w_threshold)
        elif self.notears.method == 'sob':
            model = NotearsSobolev(d, 10)
            W_est = notears_nonlinear(model, X, lambda1=self.notears.lambda1, lambda2=self.notears.lambda1, w_threshold=self.notears.w_threshold)
        
        B_est = (W_est != 0).astype(np.int32)
        
        
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
    
    def produce_counterfactuals(self, factual_batch: dict, do_parent: str, w: float=0.8):
        
        cond_stats = load_json(self.config.image_data.meta_data.cond_stats_path)
        
        utils = DiffusionUtils(timesteps=self.config.diffusion.timesteps)
        
        thickness = factual_batch['thickness'][:, None].float()
        intensity = factual_batch['intensity'][:, None].float()
        slant = factual_batch['slant'][:, None].float()
        width = factual_batch['width'][:, None].float()
        x_obs = torch.cat([thickness, intensity, slant, width], dim=1).to(self.device)
        images_obs = factual_batch['image'].unsqueeze(1).to(self.device).float() / 255.0 
        
        cf_idx = [list(factual_batch.keys()).index(do_parent) for _ in range(factual_batch['image'].shape[0])]
        min = float(cond_stats[do_parent]['min'])
        max = float(cond_stats[do_parent]['max'])
        cf_val = np.random.uniform(min, max, size=factual_batch['image'].shape[0]).tolist()

        # todo: 将来的にはここでoursとcausal vaeを選択できるようにする
        # if model == 'ours':
        noises = utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, x_obs, w, factual_batch['image'].shape[2], images_obs.shape[0], channels=1, reverse=True, input_img=images_obs)
        x_cf = self.carefl.predict_counterfactual(x_obs, cf_idx, cf_val)
        temp_imgs = utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, x_cf, w, factual_batch['image'].shape[2], images_obs.shape[0], channels=1, reverse=False, noise=noises[-1])
        cf_images = scale_image_0_1_to_0_255(temp_imgs[-1])
        
        counterfactual_batch = {
        'image': cf_images,
        'thickness': x_cf[:, 0],
        'intensity': x_cf[:, 1],
        'slant': x_cf[:, 2],
        'width': x_cf[:, 3]
        }
            
        # elif model == 'causal vae'
        
        
        return counterfactual_batch
    

        
        
        
        
        
        
        
        
        