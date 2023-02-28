import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models

import numpy as np

# from .resnet import ResNet34
import clip

from .encoder import PointNetEncoder, TextureEncoder

from .decoder import PointwiseNet


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


class DiffusionPoint(nn.Module):
    def __init__(self, decoder: PointwiseNet, var_sched: VarianceSchedule):
        super().__init__()
        self.decoder = decoder
        self.var_sched = var_sched

    def get_loss(
        self,
        x_0: Tensor,    # 输入的原始点云数据
        latent: Tensor, # 经过 PointNet 编码的 latent
        t=None,         # 采样时刻
    ):

        B, N, Ts, Ts, Ts, C = x_0.size()
        x_0 = x_0.view(B, N, -1, C)
        x_0 = x_0.view(B, N, -1)
        # x_0 = x_0.permute(0, 2, 1)

        batch_size, num_point, point_dim = x_0.size()

        # 根据迭代时刻采样参数 alpha_bar beta
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        # 任意时刻的采样值 x_t ，可以完全由 \x_0 和 \beta_t 推导得出
        # # torch.sqrt(alpha_bar).view(-1, 1, 1)     # (B, 1, 1)
        a = torch.sqrt(alpha_bar).view(-1, 1, 1)     # (B, 1, 1)
        b = torch.sqrt(1 - alpha_bar).view(-1, 1, 1) # (B, 1, 1)
        z_t = torch.randn_like(x_0)                  # (B, N, d)

        x_t = a * x_0 + b * z_t

        e_theta = self.decoder.forward(x_t, beta=beta, latent=latent)

        loss = F.mse_loss(
            e_theta.view(-1, point_dim),
            z_t.view(-1, point_dim),
            reduction='mean',
        )

        return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        # 获取高斯噪声 x_T
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        # trajectory 弹道，轨迹。 存储去噪过程中的轨迹
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            # 随机高斯噪声
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            # 获取预设的加噪参数
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.decoder.forward(x_t, beta=beta, context=context) # diffusion 计算过程
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()                                   # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()                                         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]


class TextureDiffusion(nn.Module):
    def __init__(
        self,
        texture_size=4,
        num_steps=200,
        beta_1=1e-4,
        beta_T=0.05,
        sched_mode="linear",
        device='cpu',
    ):
        super().__init__()
        zdim = 512                     # CLIP latent size=512
        self.encoder = TextureEncoder(
            texture_size=texture_size,
            zdim=zdim,
        ).to(device)
        self.encoder = self.encoder
        # self.image_encoder = ResNet34(image_encoder_classes)
        # self.image_encoder.load_state_dict(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).state_dict())
        # self.image_encoder.eval()

        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        self.image_encoder = clip_model

        out_dim = texture_size * texture_size * texture_size * 3
        self.decoder = PointwiseNet(
            out_dim=out_dim,
            zdim=zdim,
            residual=True,
        ).to(device)

        self.var_sched = VarianceSchedule(
            num_steps=num_steps,
            beta_1=beta_1,
            beta_T=beta_T,
            mode=sched_mode,
        ).to(device)

    def get_loss(self, x: Tensor):
        """ x: [B, N, Ts, Ts, Ts, 3] input texture """
        x_0 = x
        device = x.device
        B, N, Ts, Ts, Ts, C = x_0.size()
        x_0 = x_0.view(B, N, -1, C)
        x_0 = x_0.view(B, N, -1)
        batch_size, num_point, point_dim = x_0.size()

        # 根据迭代时刻采样参数 alpha_bar beta
        # if t == None:
        #     t = self.var_sched.uniform_sample_t(batch_size)
        t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        # 任意时刻的采样值 x_t ，可以完全由 \x_0 和 \beta_t 推导得出
        a = torch.sqrt(alpha_bar).view(-1, 1, 1).to(device)     # (B, 1, 1)
        b = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).to(device) # (B, 1, 1)
        z_t = torch.randn_like(x_0).to(device)                  # (B, N, d)

        x_t = a * x_0 + b * z_t

        mean, logvar = self.encoder.forward(x)

        e_theta = self.decoder.forward(x_t, beta=beta, latent=mean)

        loss = F.mse_loss(
            e_theta.view(-1, point_dim),
            z_t.view(-1, point_dim),
            reduction='mean',
        )
        return loss

    def sample(
        self,
        num_points: int,
        context: Tensor,
        point_dim: int = 3,
        flexibility: float = 0.0,
        ret_traj: bool = False,
    ):
        batch_size = context.size(0)
        # 获取高斯噪声 x_T
        x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        # trajectory 弹道，轨迹。 存储去噪过程中的轨迹
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            # 随机高斯噪声
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            # 获取预设的加噪参数
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.decoder.forward(x_t, beta=beta, context=context) # diffusion 计算过程
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()                                   # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()                                         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]
