import torch
from torch.utils.data import Dataset

import numpy as np
from pathlib import Path
from typing import Dict, List

from dataset.morphomnist import load_morphomnist_like

from diffusion.diffusion_model.models.model11 import UNet
from diffusion.utils.diffusion_utils import DiffusionUtils
from diffusion.utils.utils import load_json
from diffusion.utils.utils import scale_image_0_1_to_0_255

from notears.linear import notears_linear
from notears.nonlinear import NotearsMLP, NotearsSobolev
from notears.nonlinear import notears_nonlinear

from carefl.models.carefl2 import CAREFL

from pipeline.utils import normalize



# MIN_MAX = {
#     "thickness": [0.82152224, 6.384839],
#     "intensity": [66.48045, 254.93214],
#     "slant": [-43.692436, 66.94711],
#     "width": [10.000215, 24.999382],
#     "image": [0.0, 255.0]
# }

class CausalPipeline:
    def __init__(self, config):
        self.config = config # namespace型のconfig
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.notears = config.notears
        
        self.causal_graph = load_json(config.image_data.meta_data.graph_path)
        self.attrs = list(self.causal_graph.keys())
        
        self.utils = DiffusionUtils(timesteps=config.diffusion.timesteps)
        
        self.w = config.diffusion.guidance.guidance_scale
        
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
        _ = carefl._train(X, B_est)
        self.carefl = carefl
        
        return carefl

    # def train_guided_diffusion(self, images: torch.Tensor, metadata: torch.Tensor):
    #     print("Training guided diffusion model")
        
    #     # self.denoise_model = denoise_modelとして介入・反事実推論プログラムの入力からdenoise_modelを消しても良い
    #     # 実装途中
    #     return
    
    def prepare_models(self):
        
        _, _, metrics_df = load_morphomnist_like(self.config.image_data.data_dir)
        metrics = metrics_df.to_numpy().astype(np.float32)
        
        if self.config.evaluate.run_causaldiscovery:
            B = self.run_causal_discovery(metrics)
        else:
            B = np.load(self.config.image_data.meta_data.causalgraph_path)
        
        normalized_metrics = normalize(metrics)
        carefl = self.train_meta_causal_model(normalized_metrics, B)
        
        denoise_model = UNet().to(self.device)
        checkpoint = torch.load(self.config.diffusion.checkpoint_path)
        denoise_model.load_state_dict(checkpoint['denoise_model_state_dict'])
        denoise_model.eval()
        self.denoise_model = denoise_model
    
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
    
    # def produce_counterfactuals(self, factual_batch: dict, do_parent: str, intervention_source: Dataset,  w: float=0.8):
        
        
    #     utils = DiffusionUtils(timesteps=self.config.diffusion.timesteps)
        
    #     attrs = ['thickness', 'intensity', 'slant', 'width']
    #     x_obs = torch.cat([factual_batch[attr][:, None].float() for attr in attrs], dim=1).to(self.device)
    #     images_obs = factual_batch['image'].unsqueeze(1).to(self.device).float() / 255.0 
        
        
    #     cf_idx = attrs.index(do_parent) 
    #     interventions = intervention_source[do_parent]
        
    #     noises = utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, x_obs, w, images_obs.shape[2], images_obs.shape[0], channels=1, reverse=True, input_img=images_obs)
    #     x_cf = self.carefl.predict_counterfactual2(x_obs, cf_idx, interventions)
    #     temp_imgs = utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, x_cf, w, images_obs.shape[2], images_obs.shape[0], channels=1, reverse=False, noise=noises[-1])
    #     cf_images = scale_image_0_1_to_0_255(temp_imgs[-1]) # 0~255スケール整数に変換
        
    #     # 画像・メタ情報のスケールを全て[-1,1]に変換
    #     temp_images = cf_images / 255.0 # [0, 255] -> [0, 1]
    #     scaled_cf_images = 2 * temp_images - 1 # [0, 1] -> [-1, 1]
        
    #     def normalize_meta(values):
    #         for k, v in MIN_MAX.items():
    #             if k == 'image':
    #                 continue
    #             idx = attrs.index(k)
    #             values[:,idx] = (values[:,idx] - v[0]) / (v[1] - v[0])
    #             values[:,idx] = 2 * values[:,idx] - 1
                
    #         return values
                
    #     scaled_x_cf = normalize_meta(x_cf)
        
    #     counterfactual_batch = {'image': scaled_cf_images, **{attr: scaled_x_cf[:, attrs.index(attr)] for attr in attrs}}
        
        
        
    #     return counterfactual_batch
    
    
    def diffuse(self, factual_batch: dict):
        
        cond_obs = torch.cat([factual_batch[attr] for attr in self.attrs], dim=1).to(self.device)
        images_obs = factual_batch['image'].to(self.device).float()
        
        noise_images = self.utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, cond_obs, self.w, images_obs.shape[2], images_obs.shape[0], channels=1, reverse=True, input_img=images_obs)
        z = self.carefl.flow.forward(cond_obs)[0][-1]
        
        diffused_noise = {'noise_image': noise_images[-1], 'z': z}
        return diffused_noise
    
    def denoise(self, interventions: dict, diffused_noise: dict):
        
        indexed_intervention = {self.attrs.index(attr): intervention.squeeze(-1) for attr, intervention in interventions.items()}
        cond_cf = self.carefl.generate_on_intervention(diffused_noise['z'], indexed_intervention)
        
        noise_image = diffused_noise['noise_image']
        images_cf = self.utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, cond_cf, self.w, noise_image.shape[2], noise_image.shape[0], channels=1, reverse=False, noise=noise_image)
        
        counterfactual_batch = {'image': images_cf[-1].clamp(min=-1.0, max=1.0), **{attr: cond_cf[:, [self.attrs.index(attr)]].clamp(min=-1.0, max=1.0) for attr in self.attrs}}
        
        return counterfactual_batch
        
        
        
        
        
        
        
        
        
        
        