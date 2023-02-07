import torch
from torch import nn, Tensor
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, channel=4):
        super(PointNetEncoder, self).__init__()
        zdim = 256 # lantent size

        in_channel = channel * channel * channel * 3
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
        self.fc_bn1_mean = nn.BatchNorm1d(512)
        self.fc2_mean = nn.Linear(512, 256)
        self.fc_bn2_mean = nn.BatchNorm1d(256)
        self.fc3_mean = nn.Linear(256, zdim)

        # logvariance
        self.fc1_logvar = nn.Linear(1024, 512)
        self.fc_bn1_logvar = nn.BatchNorm1d(512)
        self.fc2_logvar = nn.Linear(512, 256)
        self.fc_bn2_logvar = nn.BatchNorm1d(256)
        self.fc3_logvar = nn.Linear(256, zdim)

    def forward(self, x: Tensor):
        print('x', x.size())

        B, N, Ts, Ts, Ts, C = x.size()
        x = x.view(B, N, -1, C)
        x = x.view(B, N, -1)
        x = x.permute(0, 2, 1)
        print('x', x.size())

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x)) # [B, 1024, 1]
        print('x', x.size())

        x = torch.max(x, 2, keepdim=True)[0] # [B, 1024, 1]
        x = x.view(-1, 1024)

        mean = F.relu(self.fc_bn1_mean(self.fc1_mean(x)))    # [B, 1024]
        mean = F.relu(self.fc_bn2_mean(self.fc2_mean(mean))) # [B, 1024]
        mean = self.fc3_mean(mean)                           # [B, 1024]

        logvar = F.relu(self.fc_bn1_logvar(self.fc1_logvar(x)))      # [B, 1024]
        logvar = F.relu(self.fc_bn2_logvar(self.fc2_logvar(logvar))) # [B, 1024]
        logvar = self.fc3_logvar(logvar)                             # [B, 1024]

        return mean, logvar


if __name__ == '__main__':

    batch_size = 8
    n_face = 12306
    texture_size = 6
    color_channel = 3
    x = torch.rand(
        batch_size,
        n_face,
        texture_size,
        texture_size,
        texture_size,
        color_channel,
    )

    pointfeat = PointNetEncoder(channel=texture_size)

    mean, logvar = pointfeat.forward(x)
    print('mean', mean.size())
    print('logvar', logvar.size())