
from torch import nn
import torch

class ChannelAttention(nn.Module):
    """
    MUST-GAN: https://github.com/TianxiangMa/MUST-GAN/blob/474bbe3087147f5e7efead5b862af6643926e1bf/models/model_must.py#L147
    """
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x:torch.Tensor):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)