import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
from knn_cuda import KNN
try:
    from .encoder import PointNetEncoder
    from .point_mae import (
        TransformerEncoder,
    )
except ImportError as e:
    from encoder import PointNetEncoder
    from point_mae import (
        TransformerEncoder,
        MaskTransformer,
    )


def farthest_point_sample(xyz: np.ndarray, npoint: int):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    # xyz = xyz.transpose(0, 2, 1)
    B, N, C = xyz.shape

    centroids = np.zeros((B, npoint)) # 采样点矩阵（B, npoint）
    distance = np.ones((B, N)) * 1e10 # 采样点到所有点距离（B, N）

    batch_indices = np.arange(B) # batch_size 数组

    barycenter = np.sum((xyz), 1)            #计算重心坐标 及 距离重心最远的点
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.reshape(B, 1, C) #numpy中的reshape相当于torch中的view

    dist = np.sum((xyz - barycenter)**2, -1)
    farthest = np.argmax(dist, 1) #将距离重心最远的点作为第一个点，这里跟torch.max不一样

    for i in range(npoint):
        centroids[:, i] = farthest                                  # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, C) # 取出这个最远点的xyz坐标
        dist = np.sum((xyz - centroid)**2, -1)                      # 计算点集中的所有点到这个最远点的欧式距离，-1消掉了xyz那个维度
        mask = dist < distance
        distance[mask] = dist[mask]                                 # 更新distance，记录样本中每个点距离所有已出现的采样点（已采样集合中的点）的最小距离
        farthest = np.argmax(distance, -1)                          # 返回最远点索引

    return centroids


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


class Group(nn.Module): # FPS + KNN
    def __init__(self, num_group: int, group_size: int, feature_dim: int = 192):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.feature_dim = feature_dim
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz: Tensor):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        batch_idx = torch.arange(batch_size, device=xyz.device).unsqueeze(-1)
        # fps the centers out
        center_index = torch.from_numpy(farthest_point_sample(
            xyz.detach().cpu().numpy(),
            self.num_group,
        )).type(torch.long)                                    # [B, G]

        center = xyz[batch_idx, center_index, :] # [B, G, 3]

        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)    # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size,
            self.num_group,
            self.group_size,
            self.feature_dim,
        ).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class MAE(nn.Module):
    def __init__(
        self,
        texture_size: int = 4,
        num_points: int = 12306,
        num_groups: int = 800,    # divide all points to group
        group_size: int = 20,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'

        self.ts = texture_size
        self.mask_ratio = mask_ratio
        self.num_points = num_points
        self.num_groups = num_groups
        self.group_size = group_size
        self.num_visible = int((1 - self.mask_ratio) * num_points) # Divide into masked & un-masked groups
        self.num_masked = num_points - self.num_visible

        feature_dim = (texture_size**3) * 3

        latent_dim = 512
        self.textuer_embed = nn.Linear(
            self.ts * self.ts * self.ts * 3,
            latent_dim,
        )

        self.point_group = Group(self.num_groups, self.group_size, feature_dim)

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
        self.MAE_encoder = MaskTransformer(config)

        # Encoder-Decoder：Encoder 输出的维度可能和 Decoder 要求的输入维度不一致，因此需要转换
        self.encoder2decoer = Encoder2Decoder(num_points=self.num_visible, latent_dim=latent_dim)

        self.decoder = SimpleDecoder(latent_dim=latent_dim, out_dim=feature_dim)

        self.head = nn.Linear(feature_dim, feature_dim)

        print("Init MAE")

    def forward(self, x: Tensor):
        input_x = x
        device = x.device
        B, N, Ts, Ts, Ts, C = x.size()
        assert self.ts == Ts
        x = x.view(B, N, -1, C).view(B, N, -1)
        print(x.size())

        num_points_each_group = self.num_points // self.num_groups

        # centroids = farthest_point_sample(x.detach().cpu().numpy(), self.num_groups)
        neighborhood, center = self.point_group.forward(x)
        print("neighborhood ", neighborhood.size())
        print("center       ", center.size())

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

        pred_masked_values = self.head.forward(decoder_mask_tokens)

        recons_x = x.detach()
        recons_x[batch_idx, mask_idx] = pred_masked_values
        recons_x = recons_x.view(B, N, -1, C).view(B, N, Ts, Ts, Ts, C)

        # 比较下预测值和真实值
        mse_per_patch = (pred_masked_values - mask_x).abs().mean(dim=-1)
        loss = mse_per_patch.mean()

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

    recons_x, loss = model.forward(x)
