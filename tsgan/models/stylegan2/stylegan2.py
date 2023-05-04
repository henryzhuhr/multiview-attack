import math
import random
import functools
import operator
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.autograd import Function
from .layers import (
    LatentResBlock,
    LatentStyledConv,
    PixelNorm,
    EqualLinear,
    ToRGB,
    ConstantInput,
    ConvLayer,
    ResBlock,
)
from .cross_attention import CrossAttention
from tsgan.utils import logheader


class Generator(nn.Module):
    def __init__(
        self,
        size=1024,                        # image size
        style_dim: int = 512,             # latent dim
        conditiom_latent_dim: int = 2048,
        n_mlp: int = 8,                   # Mapping network layers from z -> w
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size

        self.condition_mapping_layer = nn.Linear(conditiom_latent_dim, style_dim) # [B,2048]->[B,512]

        # G_Mapping network from z -> w
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation=True))
        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])

        self.conv1 = LatentStyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        self.latent_noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            self.latent_noises.register_buffer(f"latent_noise_{layer_idx}", torch.randn(*[1, 1, 2**res]))
            # print(f"latent_noise_{layer_idx}", torch.randn(*[1, 1, 2**res]).shape)

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]
            self.convs.append(
                LatentStyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel)
            )
            self.convs.append(LatentStyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel
        self.n_latent = self.log_size * 2 - 2
        # self.modulation_layers = [self.conv1.conv.modulation, self.to_rgb1.conv.modulation] + \
        #                          [layer.conv.modulation for layer in self.convs]            + \
        #                          [layer.conv.modulation for layer in self.to_rgbs]

        # TODO: set lr_mlp
        self.out_mapping_layer = nn.Sequential(
            EqualLinear(size * 3, size * 2, lr_mul=lr_mlp, activation=True),
            EqualLinear(size * 2, size, lr_mul=lr_mlp, activation=True),
            EqualLinear(size, style_dim, lr_mul=lr_mlp),
        )

    def forward(
        self,
        x: torch.Tensor,           # [B, 2048]
        cond_latent: torch.Tensor, # [B, 2048]
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
    ):

        styles = [self.style(s) for s in [x, cond_latent]]

        # noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]
        noise = [getattr(self.latent_noises, f"latent_noise_{i}") for i in range(self.num_layers)]

        if truncation < 1: # TODO: 该操作使得latent code w在平均值附近，生图质量不会太糟
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            # TODO: 输入的两个latent 进行融合
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent, latent2], 1) # [B, 18, 512]

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

        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip: torch.Tensor = self.to_rgb1(out, latent[:, 1])

        i = 1
        for i, (conv1, conv2, noise1, noise2, to_rgb) in enumerate(
            zip(
                self.convs[:: 2],                                   # from index 0, step=2
                self.convs[1 :: 2],
                noise[1 :: 2],
                noise[2 :: 2],
                self.to_rgbs,
            )
        ):

            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        x = skip.reshape(skip.size(0), -1)
        x = self.out_mapping_layer(x)

        return x if not return_latents else (x, latent)


class Discriminator(nn.Module):
    def __init__(self, latent_dim=2048, cond_dim=2048):
        super().__init__()
        h_dim = 512
        self.input_conv = LatentResBlock(latent_dim, h_dim)
        self.cond_conv = LatentResBlock(cond_dim, h_dim)
        # self.cross_attention = CrossAttention(h_dim, h_dim, heads=8, dim_head=32)
        emb_dim = h_dim + h_dim

        self.out_conv = LatentResBlock(emb_dim, emb_dim // 2)
        self.fc = nn.Sequential(
            EqualLinear(emb_dim // 2, emb_dim // 2),
            EqualLinear(emb_dim // 2, emb_dim // 4),
            EqualLinear(emb_dim // 4, emb_dim // 8),
            EqualLinear(emb_dim // 8, 1, activation=False),
        )

    def forward(self, latent: torch.Tensor, cond_latent: torch.Tensor):
        latent = self.input_conv.forward(latent)
        cond_latent = self.cond_conv.forward(cond_latent)
        x = torch.cat([latent, cond_latent], dim=1)

        x = self.out_conv(x)
        x = self.fc(x)
        return x
