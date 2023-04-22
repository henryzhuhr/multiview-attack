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
    PixelNorm,
    EqualLinear,
    EqualConv2d,
    StyledConv,
    ToRGB,
    ConstantInput,
    Upsample,
    Blur,
    Downsample,
    NoiseInjection,
    ConvLayer,ResBlock,
)


class Generator(nn.Module):
    def __init__(
        self,
        size,                     # image size
        style_dim,                # latent dim
        n_mlp,                    # Mapping network layers from z -> w
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))
        self.style = nn.Sequential(*layers) # G_Mapping network from z -> w

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
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2**i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2


        self.modulation_layers = [self.conv1.conv.modulation, self.to_rgb1.conv.modulation] + \
                                 [layer.conv.modulation for layer in self.convs]            + \
                                 [layer.conv.modulation for layer in self.to_rgbs]

    def forward(
        self,
        styles: List[Tensor],
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

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
        # (style mixing) latent [B, 18, 512]
        out = self.input.forward(latent)
        # print("latent", latent.size())
        # print("out", out.size())
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[:: 2], self.convs[1 :: 2], noise[1 :: 2], noise[2 :: 2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        # print("image", image.size())

        if return_latents:
            return image, latent
        else:
            return image, None


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
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

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2**(i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out
