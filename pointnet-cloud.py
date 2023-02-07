import torch
import torch.nn.functional as F
from torch import Tensor, nn


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x: Tensor):
        # print(x.size())
        # x [B, 2048, 3]
        x = x.transpose(1, 2)                # [B, 3, 2048]
                                             # 连续升维
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 128, 2048]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, 2048]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 256, 2048]
        x = self.bn4(self.conv4(x))          # [B, 512, 2048]
                                             # 在每个维度对所有点取 max
                                             # 得到 512 * 1 维度的全局特征
        x = torch.max(x, 2, keepdim=True)[0] # [B, 512, 1]
        x = x.view(-1, 512)                  # [B, 512]

        m = F.relu(self.fc_bn1_m(self.fc1_m(x))) # [B, 512]
        m = F.relu(self.fc_bn2_m(self.fc2_m(m))) # [B, 512]
        m = self.fc3_m(m)                        # [B, 512]
        v = F.relu(self.fc_bn1_v(self.fc1_v(x))) # [B, 512]
        v = F.relu(self.fc_bn2_v(self.fc2_v(v))) # [B, 512]
        v = self.fc3_v(v)                        # [B, 512]


        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v
