import torch
import torch.nn as nn
import torch.nn.functional as F


"""
概要

・28->14->7->3とする

・グループ正規化を多用する

チェックポイント
checkpoint7.pth
"""



class DoubleConv(nn.Module):
  """
  二つの畳み込み層とReLU関数からなるモジュール

  Args:
    in_channels (int): Number of input channels
    out_channels (int): Number of output channels

  Input:
    x (torch.Tensor): Tensor of shape (batch_size, in_channles, height, width)

  Output:
    x (torch.Tensor): Tensor of shape (batch_size, out_channels, height, width)
  """

  def __init__(self, in_channels, out_channels, num_groups=1):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups, out_channels), # グループ正規化
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups, out_channels), # グループ正規化
        nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)


class Down(nn.Module):
  """
  DoubleConvとMaxPoolingを用いてダウンサンプルするモジュール

  Args:
    in_channels (int): Number of input channels
    out_channles (int): Numer of output channles

  Input:
    x (torch.Tensor): shape (batch_size, in_channels, height, width)

  Output:
    x (torch.Tensor): shape (batch_size, out_channels, height // 2, width // 2)
  """
  def __init__(self, in_channels, out_channels):
    super(Down, self).__init__()

    self.conv = DoubleConv(in_channels, out_channels)
    self.pool = nn.MaxPool2d(2)

  def forward(self, x):
    x = self.conv(x)
    x = self.pool(x)
    return x


class Up(nn.Module):
  """
  DoubleConvを用いたアップサンプリングを行うモジュール

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.

  Input:
    x1 (torch.Tensor): アップサンプリングを受けるテンソル (batch_size, in_channels, height, width)
    x2 (torch.Tensor): スキップ接続で受け取るテンソル (batch_size, in_channels // 2, height * 2, width * 2)

  Output:
    x (torch.Tensor): Tensor of shape (batch_size, out_channels, height * 2, width * 2)
  """

  def __init__(self, in_channels, out_channels):
    super(Up, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) # 転置畳み込み 画像サイズが2倍に
    self.conv = DoubleConv(in_channels, out_channels)

  def forward(self, x1, x2):
    x1 = self.up(x1) # x1のアップサンプル
    diffX = x2.size()[2] - x1.size()[2] # x1とx2の高さの差
    diffY = x2.size()[3] - x1.size()[3] # x1とx2の幅の差
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1) # スキップ接続 channelの次元で結合
    return self.conv(x)


class OutConv(nn.Module):
  """
  最終的な畳み込み層。最終出力に合うようにchannels数を合わせる

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.

  Input:
    x (torch.Tensor): Tensor of shape (batch_size, in_channels, height, width)

  Output:
    x (torch.Tensor): Tensor of shape (batch_size, out_channels, height, width)
  """

  def __init__(self, in_channels, out_channels):
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

  def forward(self, x):
    return self.conv(x)


class TimeEmbedding(nn.Module):
  """
  時間ステップを高次元空間に埋め込むためのモジュール

  Args:
    dim (int): Dimension of the embedding space

  Input:
    t (torch.Tensor): Tensor of shape (batch_size, )

  Output:
    t_emb (torch.Tensor): Tensor of shape(batch_size, dim, 1, 1)
  """

  def __init__(self, dim):
    super(TimeEmbedding, self).__init__()
    self.fc = nn.Linear(1, dim)
    self.conv = nn.Conv2d(dim, 512, kernel_size=1) # 64 => 512(ボトルネック次元)

  def forward(self, t):
    t = t[:, None].float() # (batch_size,) to (batch_size, 1)
    t_emb = self.fc(t).unsqueeze(-1).unsqueeze(-1) # (batch, dim) to (batch, dim, 1, 1)
    t_emb = self.conv(t_emb) # to (b, 512, 1, 1)
    return t_emb



class DenoiseModel(nn.Module):
  """
  時刻情報tを埋め込んだU-Netの実装

  Args:
    n_channels (int): Number of input channels
    n_classes (int): Number of output channels
    time_dim (int): Dimension of the time embedding space.

  Input:
    x (torch.Tensor): Tensor of shape (batch_size, n_channels, height, width)
    t (torch.Tensor): Tensor of shape (batch_size,)

  Output:
    logits (torch.Tensor): Tensor of shape (batch_size, n_classes, height, width)
  """

  def __init__(self, n_channels=1, n_classes=1, time_dim=64):
    super(DenoiseModel, self).__init__()
    self.time_dim = time_dim

    self.inc = DoubleConv(n_channels, 64) # (batch_size, n_channels, height, width) to (batch_size, 64, height, width)
    self.down1 = Down(64, 128) # to (b, 128, h/2, w/2)
    self.down2 = Down(128, 256) # to (b, 256, h/4, w/4)
    self.down3 = Down(256, 512) # to (b, 512, h/8, w/8)
    self.up1 = Up(512, 256) # (b, 512 + 512, h/8, w/8) to (b, 256, h/4, w/4) スキップ接続
    self.up2 = Up(256, 128) # (b, 256 + 256, h/4, w/4) to (b, 128, h/2, h/2)
    self.up3 = Up(128, 64) # (b, 128 + 128, h/2, w/2) to (b, 64, h, w)
    self.outc = OutConv(64, n_classes) # to (b, n_classes, h, w)

    self.time_embed = TimeEmbedding(time_dim)


  def forward(self, x, t):
    t_embed = self.time_embed(t)
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)

    x4 = x4 + t_embed

    x = self.up1(x4, x3)
    x = self.up2(x, x2)
    x = self.up3(x, x1)
    logits = self.outc(x)
    return logits