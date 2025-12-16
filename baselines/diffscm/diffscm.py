import torch
from torch.utils.data import Dataset

import numpy as np
from pathlib import Path
from typing import Dict, List
import os

from diffusion.diffusion_model.models.model11 import UNet
from diffusion.classifier_guidance.classifier import EncoderUNet, AntiCausalPredictor
from diffusion.utils.diffusion_utils import DiffusionUtils
from diffusion.utils.sampling_utils import get_models_functions
from diffusion.utils.utils import load_json
from diffusion.utils.utils import scale_image_0_1_to_0_255

from notears.linear import notears_linear
from notears.nonlinear import NotearsMLP, NotearsSobolev
from notears.nonlinear import notears_nonlinear

from carefl.models.carefl2 import CAREFL

from pipeline.utils import normalize




class DiffusionSCM:
    def __init__(self, config):
        self.config = config # namespace型のconfig
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.use_carefl = config.evaluate.use_carefl
        self.anticausal = config.anticausal_predictor
        self.anticausal_training = config.anticausal_predictor_training
        self.encoder = config.anticausal_predictor.encoder_unet
        
        self.causal_graph = load_json(config.image_data.meta_data.graph_path)
        self.attrs = list(self.causal_graph.keys())
        
        self.utils = DiffusionUtils(timesteps=config.diffusion.timesteps)
        
        self.w = config.diffusion.guidance.guidance_scale
        
    def train_meta_causal_model(self, X: np.ndarray, B: np.ndarray) -> CAREFL:
        print("Training meta-causal model")
        carefl = CAREFL(self.config)
        _ = carefl._train(X, B)
        self.carefl = carefl
        return carefl
    
    def prepare_models(self):
        
        if self.use_carefl:
            
            _, _, metrics_df = load_morphomnist_like(self.config.image_data.data_dir)
            metrics = metrics_df.to_numpy().astype(np.float32)
            normalized_metrics = normalize(metrics)
            
            B = np.load(self.config.image_data.meta_data.causalgraph_path)
            
            carefl = self.train_meta_causal_model(normalized_metrics, B)
            
        
        anti_causal_predictors = {attr: AntiCausalPredictor(encoder=EncoderUNet(cdim=len(self.causal_graph[attr]), mod_ch=self.encoder.mod_channel, ch_mul=self.encoder.channel_mult, pool=self.encoder.pool_type, out_channels=self.encoder.out_channels), classifier_width=self.anticausal.classifier_width) for attr in self.causal_graph.keys()}
        for key , predictor in anti_causal_predictors.items():
            file_name = next((file for file in os.listdir(self.anticausal.checkpoint_dir) if key in file), None)
            predictor.load_state_dict(torch.load(self.anticausal.checkpoint_dir + file_name , map_location=self.device)["predictor_state_dict"])
            predictor.eval()
            predictor.to(self.device)
            
        self.cond_fn = get_models_functions(self.config, anti_causal_predictors)
        
        denoise_model = UNet().to(self.device)
        checkpoint = torch.load(self.config.diffusion.checkpoint_path)
        denoise_model.load_state_dict(checkpoint['denoise_model_state_dict'])
        denoise_model.eval()
        self.denoise_model = denoise_model
    
    
    def diffuse(self, factual_batch: dict):
        
        cond_obs = torch.cat([factual_batch[attr] for attr in self.attrs], dim=1).to(self.device)
        images_obs = factual_batch['image'].to(self.device).float()
        
        noise_images = self.utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, cond_obs, self.w, images_obs.shape[2], images_obs.shape[0], channels=1, reverse=True, input_img=images_obs)
        if self.use_carefl:
            z = self.carefl.flow.forward(cond_obs)[0][-1]
            diffused_noise = {'noise_image': noise_images[-1], 'z': z}
        else:
            diffused_noise = {'noise_image': noise_images[-1]}
        
        return diffused_noise
    
    def denoise(self, factual_batch: dict, interventions: dict, diffused_noise: dict):
        
        if self.use_carefl:
            indexed_intervention = {self.attrs.index(attr): intervention.squeeze(-1) for attr, intervention in interventions.items()}
            cond_cf = self.carefl.generate_on_intervention(diffused_noise['z'], indexed_intervention)
        else:
            cond_cf = torch.cat([factual_batch[attr] if attr not in interventions.keys() else interventions[attr] for attr in self.attrs], dim=1).to(self.device)
        
        noise_image = diffused_noise['noise_image']
        images_cf = self.utils.ddim_p_sample_loop(self.denoise_model, 0.0, 1, cond_cf, self.w, noise_image.shape[2], noise_image.shape[0], channels=1, reverse=False, noise=noise_image, cond_fn=self.cond_fn)
        
        counterfactual_batch = {'image': images_cf[-1].clamp(min=-1.0, max=1.0), **{attr: cond_cf[:, [self.attrs.index(attr)]].clamp(min=-1.0, max=1.0) for attr in self.attrs}}
        
        return counterfactual_batch