import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


class ConcatSquashLinear(nn.Module):
    """
        MLP 变体
    """
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()

        self._hyper_gate = nn.Linear(dim_ctx, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, ctx, x):
        gate = F.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class PointwiseNet(nn.Module):
    def __init__(self, out_dim, zdim, residual=True):
        super().__init__()
        self.act = nn.GELU()
        self.residual = residual

        # self.time_embed = nn.Sequential(
        #     nn.Linear(model_channels, time_embed_dim),
        #     nn.SiLU(),
        #     nn.Linear(time_embed_dim, time_embed_dim),
        # )

        self.layers = nn.ModuleList(
            [
                ConcatSquashLinear(out_dim, 128, zdim + 3),
                ConcatSquashLinear(128, 256, zdim + 3),
                ConcatSquashLinear(256, 512, zdim + 3),
                ConcatSquashLinear(512, 256, zdim + 3),
                ConcatSquashLinear(256, 128, zdim + 3),
                ConcatSquashLinear(128, out_dim, zdim + 3)
            ]
        )

    def forward(
        self,
        x: Tensor,      # x:      Point clouds at some timestep t, (B, N, d).
        latent: Tensor, # latent: Shape latents. (B, F).
        beta: Tensor,   # beta:   Time. (B, ).
    ) -> Tensor:

        B = x.size(0)
        beta = beta.view(B, 1, 1)      # (B, 1, 1)
        latent = latent.view(B, 1, -1) # (B, 1, F)

        # timestep_embedding
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1) # (B, 1, 3)

        ctx_emb = torch.cat([time_emb, latent], dim=-1) # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class TextureDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        num_points: int = 12306,
        num_feature: int = 4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_feature = num_feature
        self.num_points = num_points
        self.out_dim = (num_feature**3) * 3

        self.conv1 = nn.Conv1d(latent_dim, 512, 1)
        self.conv2 = nn.Conv1d(512, 1024, 1)
        self.conv3 = nn.Conv1d(1024, 512, 1)
        self.conv4 = nn.Conv1d(512, self.out_dim, 1)

        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.out_dim)
        self.act = nn.GELU()

        self.dact = F.leaky_relu
        self.layers = nn.ModuleList(
            [
                ConcatSquashLinear(3, 128, num_feature),
                ConcatSquashLinear(128, 256, num_feature),
                ConcatSquashLinear(256, 512, num_feature),
                ConcatSquashLinear(512, 256, num_feature),
                ConcatSquashLinear(256, 128, num_feature),
                ConcatSquashLinear(128, 3, num_feature)
            ]
        )

    def forward(self, latent_x: Tensor):
        """
            [1, latent_dim] --> [B, N, Ts, Ts, Ts, 3]
        """
        B = latent_x.size(0)
        ts = self.num_feature
        x = latent_x
        x = x.repeat(1, self.num_points, 1).permute(0, 2, 1)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))

        x = x.view(B, self.num_points, (ts * ts * ts), 3)
        x = x.view(B, self.num_points, ts, ts, ts, 3)
        return x

    def denoise(self, x_t: Tensor, beta: Tensor, latent: Tensor):
        """
            [1, latent_dim] --> [B, N, Ts, Ts, Ts, 3]
        """
        batch_size = x_t.size(0)
        beta = beta.view(batch_size, 1, 1)      # (B, 1, 1)
        latent = latent.view(batch_size, 1, -1) # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1) # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, latent], dim=-1)                        # (B, 1, F+3)

        print(time_emb.size())
        print(ctx_emb.size())

        out = x_t
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.dact(out)

        return x + out


if __name__ == '__main__':

    latent_x = torch.rand([1, 256]).cuda()

    model = TextureDecoder(
        latent_dim=256,
        num_points=12306,
        texture_size=4,
    )
    model.cuda()
    model.train()
    x = model.forward(latent_x)
