from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
概要
・以下の実装を参考にしている
https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch/blob/master/unet.py

・正弦波埋め込み
・ResBlockを多用
・条件情報（連続値）をいくつかの方法で埋め込み
 ・埋め込みを加算
 ・埋め込みをチャネル方向にconcat
 ・FiLMを用いてスケールシフト
 ・（ひつようであれば）cross-attention
 
 model9との差分
 条件間の相関を学習しないように条件ごとにMLPを個別に作成して埋め込み

チェックポイント
checkpoint_.pth
"""


def timestep_embedding(timesteps:torch.Tensor, dim:int, max_period=10000) -> torch.Tensor:
    """
    時間埋め込みを計算する関数
    Args:
        timesteps (torch.Tensor): サイズ(b,)の時間ステップのテンソル
        dim (int): 埋め込みの次元数
        max_period (int): 最大周期
    Returns:
        torch.Tensor: サイズ(b, dim)の時間埋め込みのテンソル
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class Upsample(nn.Module):
    """
    アップサンプリングモジュール
    Args:
        in_channels (int): 入力チャネル数
        out_channels (int): 出力チャネル数
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.layer = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f"Input channels {x.shape[1]} do not match expected {self.in_ch}"
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        output = self.layer(x)
        return output
    
class Downsample(nn.Module):
    """
    ダウンサンプリングモジュール
    Args:
        in_channels (int): 入力チャネル数
        out_channels (int): 出力チャネル数
    """
    def __init__(self, in_ch:int, out_ch:int, use_conv:bool):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        if use_conv:
            self.layer = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=2, padding=1)
        else:
            self.layer = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_ch, f"Input channels {x.shape[1]} do not match expected {self.in_ch}"
        output = self.layer(x)
        return output
    
class CondEmbedShared(nn.Module):
    """
    K個の条件y_kをそれぞれ独立にout_dim次元に埋め込む
    """
    def __init__(self, num_cond:int, out_dim:int):
        super().__init__()
        self.emb = nn.ModuleList([
            nn.Sequential(nn.Linear(1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)) for _ in range(num_cond)
        ])
    def forward(self, cond:torch.Tensor) -> torch.Tensor:
        # cond: (b, num_cond)
        out = [m(cond[:, i:i+1]) for i, m in enumerate(self.emb)] # num_cond個 [(b, out_dim), ..]
        return torch.stack(out, dim=1) # (b, num_cond, out_dim)
    
class CondEmbedConcat(nn.Module):
    """
    K個の条件y_kを連結してout_dim次元に埋め込む
    """
    
class EmbedBlock(nn.Module):
    """
    abstract class
    """
    @abstractmethod
    def forward(self, x, temb, cemb):
        """
        abstract method
        """
class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb)
            else:
                x = layer(x)
        return x
        
class ResBlock(EmbedBlock):
    """
    Residual Block with Group Normalization and SiLU activation
    Args:
        in_ch (int): Number of input channels
        out_ch (int): Number of output channels
        tdim (int): Dimension of time embedding
        cdim (int): Dimension of class embedding
        num_groups (int): Number of groups for Group Normalization
        droprate (float): Dropout rate
    """
    def __init__(self, in_ch:int, out_ch:int, tdim:int, cdim:int, num_cond:int, num_groups:int, droprate:float, cond_type:str):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tdim = tdim
        self.cdim = cdim
        self.num_cond = num_cond
        self.num_groups = num_groups
        self.droprate = droprate
        self.cond_type = cond_type
        
        self.block_1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        
        # model9からの差分:条件ごとに別々の埋め込みMLPを作成
        if cond_type == 'add':
            self.cemb_proj = nn.ModuleList([
                nn.Sequential(nn.SiLU(), nn.Linear(cdim, out_ch))
                for _ in range(num_cond)
            ])
        elif cond_type == 'film':
            self.cemb_proj = nn.ModuleList([
                nn.Sequential(nn.SiLU(), nn.Linear(cdim, out_ch * 2))
                for _ in range(num_cond)
            ])
        else:
            raise ValueError(f"Unknown cond_type: {cond_type}")
        
        
        """
        # model9での実装
        if self.cond_type == 'add':
            self.cemb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cdim, out_ch),
            )
        elif self.cond_type == 'film':
            self.cemb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cdim, out_ch * 2),
            )
        else:
            raise ValueError(f"Unknown cond_type: {self.cond_type}")
        """
    
        
        self.block_2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(),
            nn.Dropout(p = self.droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.residual = nn.Identity()
            
    def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
        latent = self.block_1(x)
        latent += self.temb_proj(temb)[:, :, None, None]
        
        # model9からの差分:条件ごとに別々の埋め込みMLPを作成
        if self.cond_type == 'add':
            add = 0
            for i in range(self.num_cond):
                add += self.cemb_proj[i](cemb[:, i:i+1].squeeze(1))
            latent += add[:, :, None, None]
        
        else:
            scale, shift = 0, 0
            for i in range(self.num_cond):
                emb = self.cemb_proj[i](cemb[:, i:i+1].squeeze(1))
                sc, sh = emb.chunk(2, dim=1)
                scale += sc
                shift += sh
            latent = latent * (1 + scale)[:, :, None, None] + shift[:, :, None, None]
        
        """
        # model9での実装
        # 条件の伝達方法切り替え
        if self.cond_type == 'add':
            latent += self.cemb_proj(cemb)[:, :, None, None]
        elif self.cond_type == 'film':
            emb = self.cemb_proj(cemb)[:, :, None, None]
            scale, shift = emb.chunk(2, dim=1)
            latent = latent * (1 + scale) + shift
        else:
            raise ValueError(f"Unknown cond_type: {self.cond_type}")
        """
        
        latent = self.block_2(latent)
        latent += self.residual(x)
        return latent
    
class AttnBlock(nn.Module):
    def __init__(self, in_ch:int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size = 1, stride=1, padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h
    
class UNet(nn.Module):
    def __init__(self, in_ch=1, mod_ch=64, out_ch=1, ch_mul=[1,2,4], num_res_blocks=1, cdim=2, use_conv=True, num_groups=16, droprate=0.0, cond_type='add', use_attn=False, dtype=torch.float32):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        self.cdim = cdim
        self.use_conv = use_conv
        self.num_groups = num_groups
        self.droprate = droprate
        self.cond_type = cond_type
        self.use_attn = use_attn
        self.dtype = dtype
        
        tdim = mod_ch * 4
        self.temb_layer = nn.Sequential(
            nn.Linear(mod_ch, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        
        # model9からの差分:条件ごとに別々のMLPを設計
        self.cemb_layer = CondEmbedShared(cdim, tdim)
        
        """
        # model9での実装
        self.cemb_layer = nn.Sequential(
            nn.Linear(self.cdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        """
        
        self.downblocks = nn.ModuleList([
            EmbedSequential(nn.Conv2d(in_ch, self.mod_ch, 3, padding=1))
        ])
        now_ch = self.ch_mul[0] * self.mod_ch
        chs = [now_ch]
        
        for i, mul in enumerate(self.ch_mul):
            nxt_ch = mul * self.mod_ch
            for _ in range(num_res_blocks):
                layers = [ResBlock(now_ch, nxt_ch, tdim, tdim, cdim, self.num_groups, self.droprate, self.cond_type)]
                if self.use_attn:
                    layers.append(AttnBlock(nxt_ch))
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mul) - 1:
                self.downblocks.append(EmbedSequential(Downsample(now_ch, now_ch, self.use_conv)))
                chs.append(now_ch)
        
        if use_attn:
            self.middleblocks = EmbedSequential(
            ResBlock(now_ch, now_ch, tdim, tdim, cdim, self.num_groups, self.droprate, self.cond_type),
            AttnBlock(now_ch),
            ResBlock(now_ch, now_ch, tdim, tdim, cdim, self.num_groups, self.droprate, self.cond_type),
        )
        else:
            self.middleblocks = EmbedSequential(
            ResBlock(now_ch, now_ch, tdim, tdim, cdim, self.num_groups, self.droprate, self.cond_type),
            ResBlock(now_ch, now_ch, tdim, tdim, cdim, self.num_groups, self.droprate, self.cond_type),
        )
        
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul))[::-1]:
            nxt_ch = mul * self.mod_ch
            for j in range(num_res_blocks + 1):
                layers = [ResBlock(now_ch + chs.pop(), nxt_ch, tdim, tdim, cdim, self.num_groups, self.droprate, self.cond_type)]
                if self.use_attn:
                    layers.append(AttnBlock(nxt_ch))
                now_ch = nxt_ch
                if i and j == self.num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential(*layers))
                
        self.out = nn.Sequential(
            nn.GroupNorm(self.num_groups, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x:torch.Tensor, t:torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
        temb = self.temb_layer(timestep_embedding(t, self.mod_ch))
        cemb = self.cemb_layer(cond)
        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            h = block(h, temb, cemb)
            hs.append(h)
        h = self.middleblocks(h, temb, cemb)
        for block in self.upblocks:
            h = torch.cat([h, hs.pop()], dim = 1)
            h = block(h, temb, cemb)
        h = h.type(self.dtype)
        return self.out(h)