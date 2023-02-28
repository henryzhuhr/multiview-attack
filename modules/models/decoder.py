import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

from modules.models.resnet import ResNet34


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
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class PointwiseNet(nn.Module):
    def __init__(self, out_dim, zdim, residual=True):
        super().__init__()
        self.act = F.leaky_relu
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


class ImageEncoder(nn.Module):
    def __init__(self, zdim=256) -> None:
        super().__init__()
        resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.encoder = ResNet34(1000)
        self.encoder.load_state_dict(resnet34.state_dict())
        self.encoder.eval()
        self.fc = nn.Linear(self.encoder.fc.in_features, zdim)

    def forward(self, x: Tensor):
        x = self.encoder(x)
        x = self.fc(x)
        return x