import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

timesteps = 1000 # config設定できないか？
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DiffusionUtils:
  def __init__(self, timesteps=500, beta_start=1e-4, beta_end=0.02, device=None):
    self.timesteps = timesteps
    self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.betas = self.linear_beta_schedule(beta_start, beta_end).to(self.device)
    self.alphas_cumprod = torch.cumprod(1. - self.betas, axis=0).to(self.device)
    self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
    self.posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).to(self.device)


  def linear_beta_schedule(self, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, self.timesteps)

  def extract(self, a, t, x_shape: torch.Size):
    """各サンプルのステップtに対応するインデックスの要素を抽出する
      Args:
          a ( T, ): alphasやbetas
          t ( b, ): バッチ内各サンプルのタイムステップt
          x_shape: 画像サイズ
      Returns:
          out ( b, 1, 1, 1 ): xとの計算に使うためxに次元数を合わせて返す
      """
    batch_size = t.shape[0]
    out = a.gather(-1, t.to(a.device))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

  def q_sample(self, x_start, t, noise):
    """Reparameterizationを用いて，実画像x_0からノイズ画像x_tをサンプリングする（拡散過程）
      Args:
          x_start ( b, c, h, w ): 実画像x_0
          t ( b, ): バッチ内各サンプルのタイムステップt
          noise ( b, c, h, w ): 標準ガウス分布からのノイズ (epsilon)
      Returns:
          x_noisy: ( b, c, h, w ): ノイズ画像x_t
    """
    alphas_cumprod_t = self.extract(self.alphas_cumprod, t, x_start.shape)

    return torch.sqrt(alphas_cumprod_t) * x_start + torch.sqrt(1 - alphas_cumprod_t) * noise


  def p_losses(self, denoise_model, x_start, t, noise=None):
    """
      Args:
          denoise_model ( nn.Module ): U-Net
          x_start ( b, c, h, w ): 実画像x_0
          t ( b, ): バッチ内各サンプルのタイムステップt
    """
    if noise is None:
      noise = torch.randn_like(x_start)

    x_noisy = self.q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)
    loss = F.mse_loss(noise, predicted_noise)

    return loss



  # DDPM用


  @torch.no_grad()
  def ddpm_p_sample(self, denoise_model, x, t_index):
    """1ステップ逆過程を進む
      Args:
          model ( nn.Module ): U-Net
          x ( b, c, h, w ): ノイズ画像x_t
          t_index ( int ): サンプリングループにおける現在のタイムステップt（サンプル共通）
      Returns:
          x ( b, c, h, w ): ノイズ画像x_{t-1} or 実画像x_0
    """
    t = torch.full((x.shape[0],), t_index, device=x.device, dtype=torch.long) # t_index値が(b,)の形で並ぶ
    betas_t = self.extract(self.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = self.extract(torch.sqrt(1.0 - self.alphas_cumprod), t, x.shape)
    coef_t = self.extract(torch.sqrt(1.0 / (1. -self.betas)), t, x.shape)
    model_mean = coef_t * (x - betas_t * denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    if t_index == 0:
      return model_mean
    else:
      posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
      noise = torch.randn_like(x)
      return model_mean + torch.sqrt(posterior_variance_t) * noise

  @torch.no_grad()
  def ddpm_p_sample_loop(self, denoise_model, image_size, batch_size, channels=1):
    """逆過程のステップを繰り返し，画像を生成する
      Args:
          model ( nn.Module ): U-Net
          image_size ( int ): 画像サイズ
      Returns:
          imgs ( b, c, h, w ): 生成画像
    """

    # 純粋なノイズから逆過程を始める
    shape = (batch_size, channels, image_size, image_size)
    img = torch.randn(shape, device=device)

    # ループ
    imgs = []
    for t in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
      img = self.ddpm_p_sample(denoise_model, img, t)
      imgs.append(img.cpu().numpy())

    return imgs



  # DDIM用

  # one step sample 1
  @torch.no_grad()
  def ddim_p_sample(self, denoise_model, x, t_index, eta):
    """1ステップ逆過程を進む
      Args:
          model ( nn.Module ): U-Net
          x ( b, c, h, w ): ノイズ画像x_t
          t_index ( int ): サンプリングループにおける現在のタイムステップt（サンプル共通）
          prev_t_index ( int ): サンプリングループにおける前のステップ
          eta ( float ): DDIMのノイズに関するハイパーパラメータ
      Returns:
          x ( b, c, h, w ): ノイズ画像x_{t-1} or 実画像x_0
    """
    t = torch.full((x.shape[0],), t_index, device=x.device, dtype=torch.long)
    
    alphas_t = self.extract(self.alphas_cumprod, t, x.shape)
    alphas_prev_t = self.extract(self.alphas_cumprod_prev, t, x.shape)

    # predict noise using model
    epsilon_theta_t = denoise_model(x, t)

    # calculate x_{t-1}
    sigma_t = eta * torch.sqrt((1 - alphas_prev_t) / (1 - alphas_t) * (1 - alphas_t / alphas_prev_t))
    epsilon_t = torch.randn_like(x)
    x_t_minus_one = (
            torch.sqrt(alphas_prev_t / alphas_t) * x +
            (torch.sqrt(1 - alphas_prev_t - sigma_t ** 2) - torch.sqrt(
                (alphas_prev_t * (1 - alphas_t)) / alphas_t)) * epsilon_theta_t +
            sigma_t * epsilon_t
    )
    return x_t_minus_one
      


  @torch.no_grad()
  def ddim_p_sample_loop(self, denoise_model, image_size, batch_size, channels=1, eta=0.0):
    """逆過程のステップを繰り返し，画像を生成する
      Args:
          model ( nn.Module ): U-Net
          image_size ( int ): 画像サイズ
      Returns:
          imgs ( b, c, h, w ): 生成画像
    """

    # 純粋なノイズから逆過程を始める
    shape = (batch_size, channels, image_size, image_size)
    img = torch.randn(shape, device=device)

    # ループ
    imgs = []
    for t in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
      img = self.ddim_p_sample(denoise_model, img, t, eta)
      imgs.append(img.cpu().numpy())

    return imgs

    
    