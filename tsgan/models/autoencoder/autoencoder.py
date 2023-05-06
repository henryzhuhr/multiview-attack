from typing import Tuple
import numpy as np
import torch
from torch import Tensor, optim, nn
import torch.nn.functional as F

# try:
#     from .encoder import TextureEncoder
#     from .decoder import TextureDecoder
# except ImportError as e:
#     from encoder import TextureEncoder
#     from decoder import TextureDecoder
from .encoder import TextureEncoder
from .decoder import TextureDecoder


class TextureAutoEncoder(nn.Module):
    def __init__(
        self,
        num_points: int = 12306,
        num_feature: int = 4,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = TextureEncoder(num_feature=num_feature, latent_dim=latent_dim)
        self.decoder = TextureDecoder(latent_dim=latent_dim, num_points=num_points, num_feature=num_feature)
        # self.init_weights()

    def encode(self, x: Tensor):
        z = self.encoder.forward(x)
        return z

    def decode(self, z: Tensor):
        x = self.decoder.forward(z)
        return x

    def forward(self, x: Tensor):
        x = self.decode(self.encode(x))
        return x

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
    #             nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)


class TextureEncoder(nn.Module):
    def __init__(
        self,
        num_feature: int = 4,
        latent_dim: int = 512, # lantent size
    ):
        super(TextureEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.stn_kd = STNkd(num_feature * num_feature * num_feature * 3)

        in_channel = num_feature * num_feature * num_feature * 3
        self.conv1 = nn.Conv1d(in_channel, in_channel, 1)
        self.conv2 = nn.Conv1d(in_channel, 512, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)

        # mean
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, latent_dim)

        self.act = nn.ReLU()

    def forward(self, x: Tensor):
        # Transform data format
        B, N, Ts, Ts, Ts, C = x.size()
        x = x.view(B, N, -1, C)
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)

        trans = self.stn_kd.forward(x)

        x = torch.bmm(trans, x)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.conv4(x) # [B, 1024, 1]

        x = torch.max(x, 2, keepdim=True)[0] # [B, 1024, 1]
        x = x.view(-1, 1024)

        x = self.act(self.fc1(x)) # [B, 1024]
        x = self.act(self.fc2(x)) # [B, 1024]
        x = self.fc3(x)           # [B, 1024]

        return x


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
        """ [1, latent_dim] --> [B, N, Ts, Ts, Ts, 3] """
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


class STNkd(nn.Module):
    def __init__(self, k=192):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.act = nn.LeakyReLU()

        self.k = k

    def forward(self, x: Tensor):
        B = x.size(0)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(B, 1)
        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()

        self._hyper_gate = nn.Linear(dim_ctx, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, ctx, x):
        gate = F.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextureAutoEncoder(
        num_feature=4,
        num_points=12306,
        latent_dim=512,
        device=device,
    )
    model.to(device)

    image = torch.rand([1, 3, 224, 224]).to(device)
    texture = torch.rand([1, 12306, 4, 4, 4, 3]).to(device)
    model.forward(texture, image)
