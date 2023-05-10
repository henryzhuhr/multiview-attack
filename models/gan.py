import math

import torch
from torch import nn
from torch.nn import functional as F

from .vit import TextureEncoder
from .decoder import TextureDecoder
from .layers import (
    LatentResBlock,
    LatentStyledConv,
    PixelNorm,
    ToRGB,
    ConstantInput,
)


class Generator(nn.Module):
    def __init__(
        self,
        style_dim: int = 1024,  # latent dim
        cond_dim: int = 80,
        latent_dim: int = 1024,
        mix_prob=0.8,
    ):
        super().__init__()

        self.size = size = 1024
        self.cond_dim = cond_dim
        self.embedding_layer = nn.Linear(cond_dim, style_dim)

        self.style_dim = style_dim

        channel_multiplier = 2
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,                       # [4]
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])

        self.conv1 = LatentStyledConv(self.channels[4], self.channels[4], 3, style_dim)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        
        self.upsamples = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]
            self.upsamples.append(LatentStyledConv(in_channel, out_channel, 3, style_dim, upsample=True))
            self.convs.append(LatentStyledConv(out_channel, out_channel, 3, style_dim))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        self.n_latent = self.log_size * 2 - 2

        self.inject_index = int((1 - mix_prob) * self.n_latent)

        self.out_mapping_layer = nn.Sequential(
            nn.Linear(size * 3, size),
            nn.LeakyReLU(),
            nn.Linear(size, latent_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        latent_x: torch.Tensor, # [B, 1024]
        y: torch.Tensor,        # [B, num_classes] onehot
    ):
        cond_latent = self.embedding_layer(F.one_hot(y, num_classes=self.cond_dim).float())

        # TODO: 输入的两个latent 进行融合
        # inject_index = self.inject_index
        # latent = latent_x.unsqueeze(1).repeat(1, inject_index, 1)
        # latent2 = cond_latent.unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
        # latent = torch.cat([latent, latent2], 1) # [B, 18, 512]
        latent = cond_latent.unsqueeze(1).repeat(1, self.n_latent, 1)

        # TODO: 这里应该指的是style mixing，在第i层前用A图的latent code控制生图的低分辨率特征；
        # 在第i层后用B图的latent code控制高分辨率图特征，从而混合了A、B两张图

        # style mixing：可视化出了不同分辨率下style控制的属性差异、并且提供了一种属性融合方式。
        # 有2组latent code w1和w2，分别生成source图A和source图B，
        # style mixing就是在生成主支网络中选择一个交叉点，交叉点前的低分辨率合成使用w1控制，
        # 交叉点之后的高分辨率合成使用w2，这样最终得到的图像则融合了图A和图B的特征。
        # 根据交叉点位置的不同，可以得到不同的融合结果。

        # --------------------- #
        out = self.input(latent)
        # print(logheader(),"latent",latent.shape)

        out = self.conv1(out, latent[:, 0])
        skip: torch.Tensor = self.to_rgb1(out, latent[:, 1])

        i = 1
        for i, (upsample, conv, to_rgb) in enumerate(zip(
            self.upsamples[:: 1],    # from index 0, step=2
            self.convs[:: 1],
            self.to_rgbs,
        )): # yapf: disable

            out = upsample(out, latent[:, i])
            out = conv(out, latent[:, i + 1])
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        x = skip.reshape(skip.size(0), -1)
        x = self.out_mapping_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_dim=2048, cond_dim=2048):
        super().__init__()
        h_dim = 512
        self.input_conv = nn.Sequential(
            LatentResBlock(latent_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_conv = nn.Sequential(
            LatentResBlock(h_dim, h_dim // 4),
            nn.BatchNorm1d(h_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(h_dim // 4, 1)
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.input_conv.forward(x)
        x = self.out_conv(x)
        x = self.fc(x)
        return x


class TextureGenerator(nn.Module):
    def __init__(
        self,
        npoint: int = 2679,
        sample_point: int = 1024,
        ts: int = 4,              # texture size
        style_dim: int = 1024,    # latent dim
        cond_dim: int = 80,
        mix_prob=0.9,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.encoder = TextureEncoder(npoint=npoint, sample_point=sample_point, ts=ts, dim=style_dim)
        self.decoder = TextureDecoder(npoint=npoint, ts=ts, dim=style_dim)
        self.g_model = Generator(style_dim=style_dim, cond_dim=cond_dim, mix_prob=mix_prob)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        """ [B, N, T, T, T, C] """
        latent = self.encoder.forward(x)
        if y is None:
            y = torch.randn(latent.shape[0], self.cond_dim, device=latent.device)
        rec = self.g_model.forward(latent, y)
        return rec

    def encode(self, x: torch.Tensor):
        return self.encoder.forward(x)

    def decode(self, x: torch.Tensor):
        return self.decoder.forward(x)