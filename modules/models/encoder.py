from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STNkd(nn.Module):
    def __init__(self, k=192):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.k = k

    def forward(self, x: Tensor):
        B = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(B, 1)
        iden = iden.to(x.device)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class TextureEncoder(nn.Module):
    def __init__(
        self,
        texture_size=4,
        zdim=256,       # lantent size
    ):
        super(TextureEncoder, self).__init__()
        num_feature = texture_size

        self.stn_kd = STNkd(num_feature * num_feature * num_feature * 3)

        in_channel = num_feature * num_feature * num_feature * 3
        self.conv1 = nn.Conv1d(in_channel, in_channel, 1)
        self.bn1 = nn.BatchNorm1d(in_channel)
        self.conv2 = nn.Conv1d(in_channel, 512, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn4 = nn.BatchNorm1d(1024)

        # mean
        self.fc1_mean = nn.Linear(1024, 512)
        self.fc_bn1_m = nn.BatchNorm1d(512)
        self.fc2_mean = nn.Linear(512, 256)
        self.fc_bn2_m = nn.BatchNorm1d(256)
        self.fc3_mean = nn.Linear(256, zdim)

        # logvariance
        self.fc1_logvar = nn.Linear(1024, 512)
        self.fc_bn1_var = nn.BatchNorm1d(512)
        self.fc2_logvar = nn.Linear(512, 256)
        self.fc_bn2_var = nn.BatchNorm1d(256)
        self.fc3_logvar = nn.Linear(256, zdim)

    def forward(self, x: Tensor):

        # Transform data format
        B, N, Ts, Ts, Ts, C = x.size()
        x = x.view(B, N, -1, C)
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)

        trans = self.stn_kd.forward(x)

        x = torch.bmm(trans, x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x)) # [B, 1024, 1]

        x = torch.max(x, 2, keepdim=True)[0] # [B, 1024, 1]
        x = x.view(-1, 1024)

        mean: Tensor = F.relu(self.fc_bn1_m(self.fc1_mean(x))) # [B, 1024]
        mean = F.relu(self.fc_bn2_m(self.fc2_mean(mean)))      # [B, 1024]
        mean = self.fc3_mean(mean)                             # [B, 1024]

        logvar: Tensor = F.relu(self.fc_bn1_var(self.fc1_logvar(x))) # [B, 1024]
        logvar = F.relu(self.fc_bn2_var(self.fc2_logvar(logvar)))    # [B, 1024]
        logvar = self.fc3_logvar(logvar)                             # [B, 1024]

        return mean, logvar


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        num_feature=4,
        zdim=256,      # lantent size
    ):
        super(PointNetEncoder, self).__init__()

        in_channel = num_feature * num_feature * num_feature * 3
        self.conv1 = nn.Conv1d(in_channel, in_channel, 1)
        self.bn1 = nn.BatchNorm1d(in_channel)
        self.conv2 = nn.Conv1d(in_channel, 512, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn4 = nn.BatchNorm1d(1024)

        # mean
        self.fc1_mean = nn.Linear(1024, 512)
        self.fc2_mean = nn.Linear(512, 256)
        self.fc3_mean = nn.Linear(256, zdim)

        # logvariance
        self.fc1_logvar = nn.Linear(1024, 512)
        self.fc2_logvar = nn.Linear(512, 256)
        self.fc3_logvar = nn.Linear(256, zdim)

    def forward(self, x: Tensor):

        # Transform data format
        B, N, Ts, Ts, Ts, C = x.size()
        x = x.view(B, N, -1, C)
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x)) # [B, 1024, 1]

        x = torch.max(x, 2, keepdim=True)[0] # [B, 1024, 1]
        x = x.view(-1, 1024)

        mean: Tensor = F.relu(self.fc1_mean(x)) # [B, 1024]
        mean = F.relu(self.fc2_mean(mean))      # [B, 1024]
        mean = self.fc3_mean(mean)              # [B, 1024]

        logvar: Tensor = F.relu(self.fc1_logvar(x)) # [B, 1024]
        logvar = F.relu(self.fc2_logvar(logvar))    # [B, 1024]
        logvar = self.fc3_logvar(logvar)            # [B, 1024]

        return mean, logvar


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()


# class FrozenClipImageEmbedder(nn.Module):
#     """
#         Uses the CLIP image encoder.
#         """
#     def __init__(
#             self,
#             model,
#             jit=False,
#             device='cuda' if torch.cuda.is_available() else 'cpu',
#             antialias=False,
#         ):
#         super().__init__()
#         self.model, _ = clip.load(name=model, device=device, jit=jit)

#         self.antialias = antialias

#         self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
#         self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

#     def preprocess(self, x):
#         # normalize to [0,1]
#         x = kornia.geometry.resize(x, (224, 224),
#                                    interpolation='bicubic',align_corners=True,
#                                    antialias=self.antialias)
#         x = (x + 1.) / 2.
#         # renormalize according to clip
#         x = kornia.enhance.normalize(x, self.mean, self.std)
#         return x

#     def forward(self, x):
#         # x is assumed to be in range [-1,1]
#         return self.model.encode_image(self.preprocess(x))

if __name__ == '__main__':

    batch_size = 1
    n_face = 12306
    texture_size = 4
    color_channel = 3
    x = torch.rand(
        batch_size,
        n_face,
        texture_size,
        texture_size,
        texture_size,
        color_channel,
    )

    pointfeat = PointNetEncoder(num_feature=texture_size)

    mean, logvar = pointfeat.forward(x)
    print('mean', mean.size())
    print('logvar', logvar.size())