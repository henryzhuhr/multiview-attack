import os
import sys
from typing import List
import numpy as np
import torch
from torch import Tensor, nn

sys.path.append(os.path.join(os.path.split(__file__)[0], ".."))
from autoencoder.texture_autoencoder import TextureAutoEncoder


def extract_into_tensor(a: Tensor, t: Tensor, x_shape: List[int]):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def reparameterize_gaussian(mean: Tensor, logvar: Tensor):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar: Tensor):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(torch.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z: Tensor):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * torch.pi) - z.pow(2) / 2
    return log_z


class VarianceSchedule(nn.Module):
    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0) # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)): # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class TextureDiffusion(nn.Module):
    """main class"""
    def __init__(
        self,
        num_points: int = 12306,
        num_feature: int = 4,
        latent_dim: int = 256,
        num_steps: int = 200,
        beta_1: float = 1e-4,
        beta_T: float = 0.05,
        sched_mode: str = "linear",
    ):
        super().__init__()
        self.texture_autoencoder = TextureAutoEncoder(
            num_feature=num_feature,
            num_points=num_points,
            latent_dim=latent_dim,
        )

        self.var_sched = VarianceSchedule(
            num_steps=num_steps,
            beta_1=beta_1,
            beta_T=beta_T,
            mode=sched_mode,
        )

    def get_loss(
        self,
        x_0: Tensor,     # 输入的原始点云数据
        t=None,
    ):
        x = x_0
        device = x_0.device
        B, N, Ts, Ts, Ts, C = x_0.size()
        feature_dim = Ts * Ts * Ts * C

        # --- Encode Texture ---
        latent_mu, latent_sigma = self.texture_autoencoder.encoder.forward(x)
        latent_x = latent_mu
        z = reparameterize_gaussian(mean=latent_mu, logvar=latent_sigma) # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)                   # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=latent_sigma)                  # (B, )
        loss_prior = (-log_pz - entropy).mean()

        # --- Diffusion Process ---
        # 根据迭代时刻采样参数 alpha_bar beta
        if t == None:
            t = self.var_sched.uniform_sample_t(B)
        t = self.var_sched.uniform_sample_t(B)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        # 任意时刻的采样值 x_t ，可以完全由 \x_0 和 \beta_t 推导得出
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).to(device)     # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).to(device) # (B, 1, 1)
        z_t = torch.randn_like(latent_x).to(device)              # noise: (B, N, d)
        x_t = c0 * latent_x + c1 * z_t

        # --- Denoise Process ---
        e_theta = self.texture_autoencoder.decoder.denoise(x_t, beta=beta, latent=latent_x)

        # --- Decode Texture ---

        
        return e_theta

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextureDiffusion()
    model.to(device)

    image = torch.rand([1, 3, 224, 224]).to(device)
    texture = torch.rand([1, 12306, 4, 4, 4, 3]).to(device)
    # model.get_loss(texture, image)
