import matplotlib.pyplot as plt
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