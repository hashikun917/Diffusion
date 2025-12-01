import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import zipfile
from datetime import datetime
import shutil


from diffusion.utils.utils import load_json, save_checkpoint, load_checkpoint, load_model_from_file, resize_images
from diffusion.utils.diffusion_utils import DiffusionUtils
from diffusion.utils.sample import conditional_ddim_plot_samples
from diffusion.utils.preprocess import scale_conditions
from dataset.morphomnist import MorphoMNISTLike



class Trainer:
  def __init__(self, config_path, result_dir, cond_stats_path):
    # Load configuration from json file
    
    self.config = load_json(config_path)
    self.cond_stats = load_json(cond_stats_path)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.image_size = self.config['image_size']
    self.deepscm_model = self.config['deepscm_model']
    self.epochs = self.config['epochs']
    self.batch_size = self.config['batch_size']
    self.timesteps = self.config['timesteps']
    #self.dataset = self.config['dataset'] # 扱うデータセットを指定
    self.dataset_path = self.config['dataset_path'] # データセットのパス
    self.use_minmax_scale = self.config['use_minmax_scale']
    self.model_path = self.config['model_path']
    self.model_name = self.config['model_name']
    self.ch_mul = self.config['ch_mul']
    self.num_res_blocks = self.config['num_res_blocks']
    self.num_groups = self.config['num_groups']
    self.droprate = self.config['droprate']
    self.cond_type = self.config['cond_type']
    self.use_attn = self.config['use_attn']
    self.uncond_rate = self.config['uncond_rate']
    self.guidance_scale = self.config['guidance_scale']
    self.optimizer_type = self.config['optimizer_type']
    self.checkpoint_dir = self.config['checkpoint_dir']
    self.result_dir = result_dir
    self.save_interval = self.config.get('save_interval', 10)
    self.plot_interval = self.config.get('plot_interval', 10)

    # Prepare dataset and dataloader
    if self.deepscm_model == "thickness and intensity":
      columns = ['thickness', 'intensity']
    elif self.deepscm_model == "full model":
      columns = ['thickness', 'intensity', 'slant', 'width']
    dataset = MorphoMNISTLike(root_dir=self.dataset_path, columns=columns, train=True)
    self.trainloader = DataLoader(dataset, self.batch_size, shuffle=True)
    
    # Model, optimizer, and scheduler setup
    loaded_model = load_model_from_file(self.model_path, self.model_name)
    self.denoise_model = loaded_model(ch_mul=self.ch_mul, num_res_blocks=self.num_res_blocks, num_groups=self.num_groups, droprate=self.droprate, cond_type=self.cond_type, use_attn=self.use_attn).to(self.device)
    # self.denoise_model = ConditionalDenoiseModel().to(self.device)
    
    if self.optimizer_type == 'Adam':
      self.optimizer = optim.Adam(self.denoise_model.parameters(), lr=1e-3)
      self.scheduler = optim.lr_scheduler.MultiStepLR(
              self.optimizer,
              milestones=[int(self.epochs * ratio) for ratio in [0.5, 0.8]],
              gamma=0.2
          )

    # Load checkpoint if available
    self.start_epoch = 0
    self.losses = []
    if os.path.exists(self.config['checkpoint_path']):
      self.denoise_model, self.optimizer, self.scheduler, self.start_epoch, loss =load_checkpoint(
        self.denoise_model, self.optimizer, self.scheduler, filename=self.config['checkpoint_path']
      )
      self.losses = [loss]
      
  def train(self):
    utils = DiffusionUtils(timesteps=self.timesteps)
    bar = tqdm(range(self.start_epoch, self.epochs))
    start_time = time.time()
    total_plot_time = 0
    
    for e in bar:
      epoch_start_time = time.time()
      for step, batch in enumerate(self.trainloader): # データセットごとに多少異なる
        self.optimizer.zero_grad()
        
        images = batch['image'].unsqueeze(1).to(self.device).float() / 255.0 # チャネルを追加してデータの正規化
        if self.image_size != images.shape[2]:
          images = resize_images(images, (32, 32))
          
        if self.deepscm_model == "thickness and intensity":
          intensity = batch['intensity'][:, None].float() # (batch, ) to (batch, 1)
          thickness = batch['thickness'][:, None].float()
          # 2025/08/07 thickness, intensityの順序に入れ替えておく
          metrics = torch.cat([thickness, intensity], dim=1).to(self.device)
          if self.use_minmax_scale:
            metrics = scale_conditions(metrics, ['intensity', 'thickness'], self.cond_stats)
            
        elif self.deepscm_model == "full model":
          thickness = batch['thickness'][:, None].float()
          intensity = batch['intensity'][:, None].float()
          slant = batch['slant'][:, None].float()
          width = batch['width'][:, None].float()
          
          metrics = torch.cat([thickness, intensity, slant, width], dim=1).to(self.device)
          if self.use_minmax_scale:
            # 使う場合はslant, widthの統計量をcond_statsに保存せよ
            metrics = scale_conditions(metrics, ['thickness', 'intensity', 'slant', 'width'], self.cond_stats)
          
        # 一定確率で条件情報をドロップアウト
        drop_mask = (torch.rand(metrics.shape[0], 1, device=self.device) < self.uncond_rate).float()  # shape: (B, 1)
        metrics = metrics * (1.0 - drop_mask)  # 値を0にする
      
        
        b = images.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=self.device).long()
        loss = utils.p_losses(self.denoise_model, images, t, c=metrics)

        self.losses.append(loss.item())
        loss.backward()
        self.optimizer.step()


      self.scheduler.step()
      #self.losses.append(loss.item())
      bar.set_description(f'Epoch {e+1}/{self.epochs}, Loss: {loss.item():.4f}')

      # Save checkpoint
      if (e + 1) % self.save_interval == 0:
        date = datetime.now().strftime("%Y-%m-%d_%H")
        save_checkpoint(self.denoise_model, self.optimizer, self.scheduler, e + 1, loss.item(),
                          filename=os.path.join(self.checkpoint_dir, f'checkpoint_{date}.pth'))
      if (e + 1) % self.plot_interval == 0:
        plot_start_time = time.time()
        # 条件付きDDIM生成によるプロット
        conditional_ddim_plot_samples(self.denoise_model, eta=0.0, interval=1, cond=metrics[:5], w=self.guidance_scale, batch_size=5, image_size=self.image_size, timesteps=self.timesteps)
        if not os.path.exists(result_dir + '/samples'):
          os.mkdir(result_dir + '/samples')
        plt.savefig(result_dir + f'/samples/epoch{e + 1}.png')
        plot_end_time = time.time()
        plot_time = plot_end_time - plot_start_time
        total_plot_time += plot_time
        
        
      # Log time per epoch
      if (e + 1) % self.plot_interval == 0:
        epoch_time = time.time() - epoch_start_time - plot_time
      else:
        epoch_time = time.time() - epoch_start_time
      print(f'Epoch {e+1} took {epoch_time:.2f} seconds')
      
    # training summary
    total_time = time.time() - start_time - total_plot_time
    avg_epoch_time = total_time / self.epochs
    self.save_results(self.denoise_model, total_time, avg_epoch_time)

  def save_results(self, model, total_time, avg_epoch_time):
    # save loss values to csv
    loss_csv_path = os.path.join(self.result_dir, 'losses.csv')
    with open(loss_csv_path, mode='w') as f:
      f.write('epoch, loss\n')
      for i, loss in enumerate(self.losses):
        f.write(f'{i+1}, {loss}\n')
        
    # plot and save loss curve
    plt.figure()
    plt.plot(self.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    loss_plot_path = os.path.join(self.result_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    
    # save summary of training times
    summary_path = os.path.join(self.result_dir, 'training_summary.txt')
    with open(summary_path, mode='w') as f:
      f.write(f'Total training time: {total_time:.2f} seconds\n')
      f.write(f'Average time per epoch: {avg_epoch_time:.2f} seconds\n')
    
    # save model structure information
    # パラメータの総数を取得
    total_params = sum(p.numel() for p in model.parameters())
    with open(summary_path, mode='a') as f:
      f.write(f'\nTotal parameters: {total_params}\n')
      
      # 層ごとのパラメータを取得
      for i, (name, param) in enumerate(model.named_parameters()):
        f.write(f'Layer{i+1}: {name} | Size: {param.size()} | Number of parameters: {param.numel()}\n')
    
      # モデルの構造を表示
      f.write("\nModel structure:\n")
      model_structure = str(model)
      f.write(model_structure)
      
    # csvにモデルの情報を保存
    params_csv_path = os.path.join(self.result_dir, 'params.csv')
    with open(params_csv_path, mode='w') as f:
      f.write('Layer, Name, number of params\n')
      for i, (name, param) in enumerate(model.named_parameters()):
        f.write(f'{i+1}, {name}, {param.numel()}\n')

    # 最適化手法に関する情報を保存
    with open(summary_path, mode='a') as f:
      f.write('\nOptimizer State:\n')
      f.write(f'Optimizer Type: {type(self.optimizer).__name__}\n')
      for param_group in self.optimizer.param_groups:
        f.write(f'Learning Rate: {param_group["lr"]}\n')
        f.write(f'Betas: {param_group.get("betas", "N/A")}\n')
        f.write(f'Eps: {param_group.get("eps", "N/A")}\n')
        f.write(f'Weight Decay: {param_group.get("weight_decay", "N/A")}\n')
    
    # Copy config file to results directory
    date = datetime.now().strftime("%Y%m%d")
    config_copy_path = os.path.join(self.result_dir, f'config_{date}.json')
    shutil.copy(config_path, config_copy_path)

    
    # create zip archive of results


if __name__ == '__main__':
  import sys
  from datetime import datetime
  
  if len(sys.argv) != 4:
    print("Usage python morphomnist_trainer.py <config_path> <results_path> <cond_stats_path")
    sys.exit(1)
    
  config_path = sys.argv[1]
  results_path = sys.argv[2]
  cond_stats_path = sys.argv[3]
  
  # Create a directory for the current training session base on the current date
  current_date = datetime.now().strftime('%Y-%m-%d_%H')
  result_dir = os.path.join(results_path, current_date)
  os.makedirs(result_dir, exist_ok=True)
  
  trainer = Trainer(config_path, result_dir, cond_stats_path)
  trainer.train()