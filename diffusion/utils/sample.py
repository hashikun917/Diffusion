import matplotlib.pyplot as plt
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
        img = imgs[timesteps-1][i].squeeze()
        ax.imshow(img, cmap='gray')

    plt.show()
    
    
# DDIM用

def ddim_plot_samples(denoise_model, eta, interval, batch_size, timesteps):
    """plot ddim samples

    Args:
        denoise_model (_type_): _description_
        batch_size (_type_): _description_
    """
    
    utils = DiffusionUtils(timesteps=timesteps)
    imgs = utils.ddim_p_sample_loop(denoise_model, eta, interval, image_size=28, batch_size=batch_size, channels=1)
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 3))
    for i in range(batch_size):
        ax = axes[i]
        steps = timesteps % interval
        img = imgs[steps-1][i].squeeze()
        ax.imshow(img, cmap='gray')

    plt.show()