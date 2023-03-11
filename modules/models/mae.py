import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models

import numpy as np

try:
    from .encoder import PointNetEncoder
except ImportError as e:
    from encoder import PointNetEncoder


class Encoder2Decoder(nn.Module):
    def __init__(
        self,
        num_points: int = 3076,
        latent_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, latent_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor):
        # x: [B,512]
        x = x.unsqueeze(1).repeat(1, self.num_points, 1) # [B, N_p, 512]
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 512,
        out_dim: int = 192,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, out_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor):
        # x: [B, N, dim]
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


class MAE(nn.Module):
    def __init__(
        self,
        texture_size: int = 4,
        num_points: int = 12306,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'
        self.ts = texture_size
        self.mask_ratio = mask_ratio
        self.num_points = num_points
        self.num_visible = int((1 - self.mask_ratio) * num_points) # Divide into masked & un-masked groups
        self.num_masked = num_points - self.num_visible

        num_feature = (texture_size**3) * 3

        latent_dim = 512
        self.textuer_embed = nn.Linear(
            self.ts * self.ts * self.ts * 3,
            latent_dim,
        )

        # Add 1 for cls_token
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_points + 1, latent_dim))
        # Mask token
        self.mask_ratio = mask_ratio
        # mask token 的实质：1个可学习的共享向量
        self.mask_embed = nn.Parameter(torch.randn(latent_dim))

        # 实际划分的 patch 数量加上 1个 cls_token
        num_patches_plus_cls_token, encoder_dim = self.pos_embed.size()[-2 :]
        # 在 Decoder 中用作对 mask tokens 的 position embedding
        # Filter out cls_token 注意第1个维度去掉 cls_token
        self.decoder_pos_embed = nn.Embedding(num_patches_plus_cls_token - 1, latent_dim)

        self.encoder = PointNetEncoder(self.num_visible, latent_dim=latent_dim)

        # Encoder-Decoder：Encoder 输出的维度可能和 Decoder 要求的输入维度不一致，因此需要转换
        self.encoder2decoer = Encoder2Decoder(num_points=self.num_visible, latent_dim=latent_dim)

        self.decoder = SimpleDecoder(latent_dim=latent_dim, out_dim=num_feature)

        self.head = nn.Linear(num_feature, num_feature)

        print("Init MAE")

    def forward(self, x: Tensor):
        device = x.device
        B, N, Ts, Ts, Ts, C = x.size()
        assert self.ts == Ts
        Tsn = Ts * Ts * Ts # Tsn: texture size new
        x = x.view(B, N, -1)

        # -- Shuffle --
        shuffle_indices = torch.rand(B, N).to(device).argsort() # (b, n_x)

        mask_idx = shuffle_indices[:, : self.num_masked]
        visible_idx = shuffle_indices[:, self.num_masked :]

        batch_idx = torch.arange(B, device=device).unsqueeze(-1) # (b, 1)
        mask_x = x[batch_idx, mask_idx]
        visible_x = x[batch_idx, visible_idx]

        # -- Encode --
        pos_embeddings = self.pos_embed.repeat(B, 1, 1)[batch_idx, visible_idx + 1]
        visible_tokens = self.textuer_embed.forward(visible_x) + pos_embeddings
        encoded_tokens = self.encoder.forward(visible_tokens)

        # -- Encoder --> Decoder--
        encoded_trans_tokens = self.encoder2decoer.forward(encoded_tokens)

        # (decoder_dim)->(b, n_masked, decoder_dim)
        mask_tokens = self.mask_embed[None, None, :].repeat(B, self.num_masked, 1)
        mask_tokens += self.decoder_pos_embed(mask_idx)

        # [ B, n_patches, decoder_dim ]
        concat_tokens = torch.cat([mask_tokens, encoded_trans_tokens], dim=1)
        decoder_input_tokens = torch.empty_like(concat_tokens, device=device)
        # Un-shuffle
        decoder_input_tokens[batch_idx, shuffle_indices] = concat_tokens

        # -- Decode
        decoded_tokens = self.decoder.forward(decoder_input_tokens)

        # -- Mask pixel Prediction --
        decoder_mask_tokens = decoded_tokens[batch_idx, mask_idx, :]
        pred_masked_values = self.head(decoder_mask_tokens)

        recons_x = x.detach()
        recons_x[batch_idx, mask_idx] = pred_masked_values
        recons_x = recons_x.view(B, N, Ts, Ts, Ts, C)

        # 比较下预测值和真实值
        mse_per_patch = (pred_masked_values - mask_x).abs().mean(dim=-1)
        loss:Tensor = mse_per_patch.mean()

        return (
            recons_x,
            loss,
        )


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MAE()
    model.to(device)
    B = 1
    x = torch.randn([B, 12306, 4, 4, 4, 3]).to(device)

    recons_x,loss = model.forward(x)
