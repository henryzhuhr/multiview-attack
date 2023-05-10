import math
import random
import functools
import operator
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.autograd import Function

from tsgan.utils import logheader
# print(logheader(),"input",input.shape) # TODO


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X 


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()

    return k


class ModulatedConv1d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size))
        self.modulation = nn.Sequential(nn.Linear(style_dim, in_channel),nn.LeakyReLU())
        self.demodulate = demodulate
        self.fused = fused


    def forward(self, input:torch.Tensor, style:torch.Tensor):
        batch, in_channel, size = input.size()
        style = self.modulation(style)
        style = style.view(batch, 1, in_channel, 1)

        weight = self.scale * (self.weight * style)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size)
        
        if self.upsample:
            input = input.view(1, batch * in_channel, size)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size
            )
            # out = conv2d_gradfix.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            out =F.conv_transpose1d(input, weight, stride=2, padding=1,output_padding=1,groups=batch)

            _, _, size = out.shape
            out = out.view(batch, self.out_channel, size)

        else:
            input = input.view(1, batch * in_channel, size)
            # out = conv2d_gradfix.conv2d(input, weight, padding=self.padding, groups=batch)
            out = F.conv1d(input, weight, padding=self.padding, groups=batch)
            _, _, size = out.shape
            out = out.view(batch, self.out_channel, size)
            

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, latent, noise=None):
        if noise is None:
            batch, _, size = latent.shape
            noise = latent.new_empty(batch, 1, size).normal_()

        return latent + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size))

    def forward(self, x):
        out = self.input.repeat(x.shape[0], 1, 1)
        return out


class LatentStyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
    ):
        super().__init__()
        self.conv = ModulatedConv1d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
        )
        self.bn=nn.BatchNorm1d(out_channel)
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = nn.LeakyReLU(out_channel)

    def forward(self, input, style):
        out = self.bn(self.conv(input, style))
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        if upsample:
            # self.upsample = Upsample(blur_kernel)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ModulatedConv1d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bn=nn.BatchNorm1d(3)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1))

    def forward(self, input, style, skip=None):
        out = self.bn(self.conv(input, style))
        out = out + self.bias
        if skip is not None:
            out = out + self.upsample(skip)
        return out


class LatentResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, in_channel)
        self.act1=nn.LeakyReLU()
        self.fc2 = nn.Linear(in_channel, out_channel)
        self.act2=nn.LeakyReLU()
        self.skip = nn.Linear(in_channel, out_channel, bias=False)

    def forward(self, x:Tensor):
        out = self.act1(self.fc1(x))
        out = self.act2(self.fc2(out))
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out
