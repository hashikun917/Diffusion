import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from diffusion.utils.diffusion_utils import DiffusionUtils

def ddpm_plot_samples(denoise_model, batch_size, timesteps):
    """plot ddpm samples

    Args:
        denoise_model (_type_): _description_
        batch_size (_type_): _description_
    """
    
    utils = DiffusionUtils(timesteps=timesteps)
    imgs = utils.ddpm_p_sample_loop(denoise_model, image_size=28, batch_size=batch_size, channels=1)
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 3))
    for i in range(batch_size):
        ax = axes[i]
        img = imgs[timesteps][i].squeeze()
        ax.imshow(img, cmap='gray')

    plt.show()
    
    
# DDIM用

def ddim_plot_samples(denoise_model, eta, interval, batch_size, timesteps, reverse=False, input_img=None):
    """plot ddim samples
    Args:
        denoise_model (_type_): _description_
        batch_size (_type_): _description_
        eta ( float ): DDIMのノイズに関するハイパーパラメータ
        interval ( int ): 生成過程or推論過程を指定
        reverse ( bool ): 生成過程or推論過程を指定
        img ( b, c, h, w ): 推論過程の場合、入力の画像
    """
    
    utils = DiffusionUtils(timesteps=timesteps)
    
    if not reverse:
        imgs = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1)
        fig, axes = plt.subplots(1, batch_size, figsize=(15, 3))
        for i in range(batch_size):
            ax = axes[i]
            steps = int(timesteps / interval)
            ax.imshow(imgs[steps][i].cpu().numpy().squeeze(), cmap='gray')

        plt.show()
    
    if reverse:
        noises = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1, reverse=True, input_img=input_img)
        imgs = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1, reverse=False, noise=noises[-1])
        fig, axes = plt.subplots(2, batch_size, figsize=(15, 6))
        axes[0, 0].set_title('input image', loc='center', fontsize=14)
        axes[1, 0].set_title('reconstructed image', loc='center', fontsize=14)
        for i in range(batch_size):
            axes[0, i].imshow(input_img[i].squeeze().cpu().numpy(), cmap='gray')
            
            steps = int(timesteps / interval)
            axes[1, i].imshow(imgs[steps][i].cpu().numpy().squeeze(), cmap='gray')

        mse_loss = nn.MSELoss()
        loss = mse_loss(input_img, imgs[steps])
        print('平均二乗誤差:', loss.item())
        plt.show()
        
        
# 条件付き拡散モデル用DDIMサンプリング   
def conditional_ddim_plot_samples(denoise_model, eta, interval, cond, w, batch_size, image_size, timesteps, reverse=False, input_img=None, intervened_cond=None):
    """plot ddim samples
    Args:
        denoise_model (_type_): _description_
        batch_size (_type_): _description_
        eta ( float ): DDIMのノイズに関するハイパーパラメータ
        interval ( int ): 生成過程or推論過程を指定
        cond ( b, dim ): 条件情報
        w ( float ): ガイダンススケール
        batch_size ( int ): バッチサイズ
        timesteps ( int ): 時間ステップ数
        reverse ( bool ): 生成過程or推論過程を指定
        input_img ( b, c, h, w ): 推論過程の場合、入力の画像
    """
    
    utils = DiffusionUtils(timesteps=timesteps)
    
    # 生成過程
    if not reverse:
        """
        # model{n} n < 9の場合
        if cond is None:
            imgs = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1)
        else:
            imgs = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1, cond=cond)
        """
        # 分類器フリーガイダンスの場合
        imgs = utils.ddim_p_sample_loop(denoise_model, eta, interval, cond, w, image_size=image_size, batch_size=batch_size, channels=1)
            
        fig, axes = plt.subplots(2, batch_size, figsize=(15, 6))
        axes[0, 0].set_title('initial noise', loc='center', fontsize=14)
        axes[1, 0].set_title('generated image', loc='center', fontsize=14)
        for i in range(batch_size):

            steps = int(timesteps / interval)
            axes[0, i].imshow(imgs[0][i].cpu().numpy().squeeze(), cmap='gray')
            axes[1, i].imshow(imgs[steps][i].cpu().numpy().squeeze(), cmap='gray')
        
        plt.tight_layout()
        plt.show()
    
    # 推論過程
    elif reverse:
        """
        # model{n} n < 9の場合
        if cond is None:
            noises = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1, reverse=True, input_img=input_img)
            imgs = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1, reverse=False, noise=noises[-1])
        else:
            noises = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1, cond=cond, reverse=True, input_img=input_img)
            imgs = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1, cond=cond, reverse=False, noise=noises[-1])
        """
        # 分類器フリーガイダンスの場合
        noises = utils.ddim_p_sample_loop(denoise_model, eta, interval, cond, w, image_size=image_size, batch_size=batch_size, channels=1, reverse=True, input_img=input_img)
        imgs = utils.ddim_p_sample_loop(denoise_model, eta, interval, intervened_cond, w, image_size=image_size, batch_size=batch_size, channels=1, reverse=False, noise=noises[-1])
         
        fig, axes = plt.subplots(3, batch_size, figsize=(15, 6))
        axes[0, 0].set_title('input image', loc='center', fontsize=14)
        axes[1, 0].set_title('reconstructed image', loc='center', fontsize=14)
        axes[2, 0].set_title('inferred noise', loc='center', fontsize=14)
        for i in range(batch_size):
            axes[0, i].imshow(input_img[i].squeeze().cpu().numpy(), cmap='gray')
            
            steps = int(timesteps / interval)
            axes[1, i].imshow(imgs[steps][i].cpu().numpy().squeeze(), cmap='gray')
            
            axes[2, i].imshow(noises[steps][i].cpu().numpy().squeeze(), cmap='gray')
        
        plt.tight_layout()
        mse_loss = nn.MSELoss()
        loss = mse_loss(input_img, imgs[steps])
        print('平均二乗誤差:', loss.item())
        plt.show()