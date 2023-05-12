import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np


class TextureDecoder(nn.Module):
    def __init__(
        self,
        nt: int = 12306,
        ts: int = 4,
        dim: int = 1024,
    ):
        super().__init__()
        self.latent_dim = dim
        self.ts = ts
        self.nt = nt
        feature = ts * ts * ts * 3

        # why ConvTranspose1d https://www.zhihu.com/question/342143752 / https://arxiv.org/pdf/1609.07009v1.pdf
        self.conv1 = nn.Conv1d(1, 64, 1)
        self.unsample1 = nn.ConvTranspose1d(64, 512, 3)
        in_c = 1 * (dim - 1) + 3 - 2 * 0 + 0 # L_{out} = stride * (L_{in} - 1) + k - 2 * padding + outpadding
        self.fc1 = nn.Linear(in_c, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.conv2 = nn.Conv1d(512, 2048, 1)
        self.unsample2 = nn.ConvTranspose1d(2048, 4096, 3)
        in_c = 1 * (512 - 1) + 3 - 2 * 0 + 0 # L_{out} = stride * (L_{in} - 1) + k - 2 * padding + outpadding
        self.fc2 = nn.Linear(in_c, 512)
        self.bn2 = nn.BatchNorm1d(4096)

        self.conv_out = nn.Conv1d(4096, nt, 1)
        self.feat_out = nn.Linear(512, feature)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, latent_x: Tensor):
        """ [B, latent_dim] --> [B, N, Ts, Ts, Ts, 3] """
        bs = latent_x.shape[0]
        nt = self.nt
        ts = self.ts

        x = latent_x.unsqueeze(dim=1) # [B, C] -> [B, 1, C]
        x = self.conv1(x)
        x = self.unsample1(x)
        x = self.fc1(x)
        x = self.bn1(x)        
        x = self.act(x)

        x = self.conv2(x)
        x = self.unsample2(x)        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.conv_out(x)
        x = self.feat_out(x)
        x = (F.tanh(x) + 1) / 2
        x = x.view(bs, nt, ts, ts, ts, 3)
        return x


if __name__ == '__main__':

    latent_x = torch.rand([8, latent_dim := 1024]).cuda()

    model = TextureDecoder(nt=2756, ts=4, dim=latent_dim).cuda().train()
    y = model.forward(latent_x)
    print(y.shape)
