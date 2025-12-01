from dataset.morphomnist import MorphoMNISTLike2
from dataset.transforms import ReturnDictTransform

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from pipeline.utils import dict2namespace

from diffusion.utils.utils import load_json, save_anticausal_predictor_checkpoint, save_anticausal_predictor_results
from diffusion.utils.diffusion_utils import DiffusionUtils
from diffusion.classifier_guidance.classifier import EncoderUNet, AntiCausalPredictor

import argparse
import yaml
import os
from datetime import datetime
from tqdm import tqdm




class AntiCausalTrainer:
    def __init__(self, config, result_dir):
        
        # configの読み込み
        self.config = config
        self.result_dir = result_dir
        self.training = config.anticausal_predictor_training
        self.attribute_size = load_json(config.image_data.meta_data.attribute_size_path)
        self.causal_graph = load_json(config.image_data.meta_data.graph_path)
        
        # データセット及びデータローダの設定
        transform = ReturnDictTransform(self.attribute_size)
        dataset = MorphoMNISTLike2(attribute_size=self.attribute_size, split='train', normalize_=True, transform=transform, data_dir=self.config.image_data.data_dir)
        self.trainloader = DataLoader(dataset, batch_size=self.training.batch_size, shuffle=True)
        
        # モデル、オプティマイザ、スケジューラの設定
        self.anticausal_predictors = {attr: AntiCausalPredictor(encoder=EncoderUNet(cdim=len(self.causal_graph[attr]))) for attr in self.causal_graph.keys()}
        self.optimizers = {attr: optim.Adam(self.anticausal_predictors[attr].parameters(), lr=1e-4, weight_decay=0.0) for attr in self.causal_graph.keys()}
        
        # todo:チェックポイントが存在するなら読み込み
        
    def train(self):
        
        utils = DiffusionUtils(self.config.diffusion.timesteps)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        save_interval = self.training.epochs // 2
        
        loss_dict = {}
        for attr, predictor in self.anticausal_predictors.items():
            bar = tqdm(range(self.training.epochs)) # todo: チェックポイントがある場合はrange(start_epoch, self.training.epochs)にする
            
            optimizer = self.optimizers[attr]
            
            predictor = predictor.to(device)
            predictor.train()
            
            loss_dict[attr] = []
            for e in bar:
                for step, batch in enumerate(self.trainloader):
                    
                    optimizer.zero_grad()
                    
                    images = batch['image'].to(device).float()
                    target = batch[attr].to(device)
                    parents = self.causal_graph[attr]
                    parent_cond = torch.cat([batch[parent] for parent in self.causal_graph[attr]], dim=1).to(device) if len(parents) > 0 else torch.Tensor([])
                    
                    t = torch.randint(0, self.config.diffusion.timesteps, (images.shape[0],), device=device).long()
                    noise = torch.randn_like(images)
                    noised_images = utils.q_sample(images, t, noise)
                    
                    pred = predictor(noised_images, t, parent_cond)
                    
                    loss = F.mse_loss(pred, target)
                    loss.backward()
                    optimizer.step()
                    
                    loss_dict[attr].append(loss.item())
                    
                if (e + 1) % save_interval == 0:
                    date = datetime.now().strftime("%Y-%m-%d_%H")
                    save_anticausal_predictor_checkpoint(predictor, optimizer, e + 1, filename=os.path.join(self.config.anticausal_predictor_training.checkpoint_dir, f'checkpoint_{attr}_{date}.pth'))
                
        save_anticausal_predictor_results(loss_dict, self.result_dir)
        print("Training completed!")
             

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Config file for DiffSCM", default="config/config_diffscm.yaml")
    parser.add_argument("--result-dir", type=str, help="Directory to save results of training anticausal predictor", default="results/anticausal_predictor_training")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()
    
    with open(args.config, 'r') as f:
        config_raw = yaml.load(f, Loader=yaml.FullLoader)
    
    config = dict2namespace(config_raw)
    
    now = datetime.now().strftime('%Y-%m-%d_%H')
    result_dir = os.path.join(args.result_dir, now)
    os.makedirs(result_dir, exist_ok=True)
    
    trainer = AntiCausalTrainer(config, result_dir)
    trainer.train()
    