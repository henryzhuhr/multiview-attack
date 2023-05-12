import math
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import clip
import kornia


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

        self.act = nn.LeakyReLU(0.2)

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

        x = self.act(self.fc1(x))    # [B, 1024]
        x = self.act(self.fc2(x))    # [B, 1024]
        x = self.fc3(x) # [B, 1024]

        return x