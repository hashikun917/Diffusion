import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.morphomnist import MorphoMNISTLike2

from diffusion.utils.utils import load_json, save_checkpoint, load_model_from_file, save_results
from diffusion.utils.diffusion_utils import DiffusionUtils

import os
from tqdm import tqdm
from datetime import datetime
import argparse
import yaml
from pipeline.utils import dict2namespace


"""
morphomnist_trainer.pyからの差分：
・[-1, 1]にスケーリングされたmorphomnistデータセットを用いて拡散モデルを学習
・configをyaml形式からdict2namespaceで変換
・全体的にコードを整理
"""


class Trainer:
    def __init__(self, config, result_dir):
      
        # configの読み込み
        self.config = config
        
        self.attribute_size = load_json(self.config.image_data.meta_data.attribute_size_path)
        
        self.diffusion = self.config.diffusion
        self.unet = self.diffusion.unet
        self.training = self.config.diffusion_training
        
        self.result_dir = result_dir

        # データセット及びデータローダの準備
        dataset = MorphoMNISTLike2(attribute_size=self.attribute_size, split='train', normalize_=True, transform=None, data_dir=self.config.image_data.data_dir)
        self.trainloader = DataLoader(dataset, batch_size=self.training.batch_size, shuffle=True)
        
        # モデルの読み込み
        loaded_model = load_model_from_file(self.unet.model_path, 'UNet')
        hyperparameters_dict = {
            'ch_mul': self.unet.ch_mul,
            'num_res_blocks': self.unet.num_res_blocks,
            'num_groups': self.unet.num_groups,
            'droprate': self.unet.droprate,
            'cond_type': self.unet.cond_type,
            'use_attn': self.unet.use_attn,
        }
        self.denoise_model = loaded_model(**hyperparameters_dict)
        
        # オプティマイザ及びスケジューラの設定
        if self.training.optim.type == 'Adam':
            self.optimizer = optim.Adam(self.denoise_model.parameters(), lr=1e-3)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    milestones=[int(self.training.epochs * ratio) for ratio in [0.5, 0.8]],
                    gamma=0.2
                )

        # todo:途中のチェックポイントがある場合は読み込み
        self.start_epoch = 0
        self.losses = []
        
    def train(self):
      
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        save_interval = self.training.epochs // 2
        utils = DiffusionUtils(timesteps=self.diffusion.timesteps)
        bar = tqdm(range(self.start_epoch, self.training.epochs))
        
        self.denoise_model = self.denoise_model.to(device)
        self.denoise_model.train()
        
        for e in bar:
            for step, batch in enumerate(self.trainloader): # データセットごとに多少異なる
                self.optimizer.zero_grad()
                
                images = batch[0].to(device).float()
                cond = batch[1].to(device)
                
                # 一定確率で条件情報をドロップアウト
                drop_mask = (torch.rand(cond.shape[0], 1, device=device) < self.diffusion.cfg.uncond_rate).float()  # shape: (B, 1)
                dropped_cond = cond * (1.0 - drop_mask)  # 値を0にする
            
                b = images.shape[0]
                t = torch.randint(0, self.diffusion.timesteps, (b,), device=device).long()
                loss = utils.p_losses(self.denoise_model, images, t, c=dropped_cond)

                self.losses.append(loss.item())
                loss.backward()
                self.optimizer.step()


            self.scheduler.step()
            bar.set_description(f'Epoch {e+1}/{self.training.epochs}, Loss: {loss.item():.4f}')

            # checkpointの保存
            if (e + 1) % save_interval == 0:
                date = datetime.now().strftime("%Y-%m-%d_%H")
                save_checkpoint(self.denoise_model, self.optimizer, self.scheduler, e + 1, loss.item(),
                                filename=os.path.join(self.training.checkpoint_dir, f'checkpoint_{date}.pth'))
                
        save_results(self.losses, self.result_dir)
        print("Training completed!")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="Config file for our model", default="config/config_all.yaml")
    parser.add_argument("--result-dir", type=str, help="Directory to save results of training diffusion model denoiser UNet", default="results/denoiser_training")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    
    with open(args.config_path, 'r') as f:
        config_raw = yaml.load(f, Loader=yaml.FullLoader)
    
    config = dict2namespace(config_raw)
    
    now = datetime.now().strftime('%Y-%m-%d_%H')
    result_dir = os.path.join(args.result_dir, now)
    os.makedirs(result_dir, exist_ok=True)
    
    trainer = Trainer(config, result_dir)
    trainer.train()