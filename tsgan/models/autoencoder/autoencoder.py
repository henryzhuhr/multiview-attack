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
from .vit import TextureEncoder
from .decoder import TextureDecoder


class TextureAutoEncoder(nn.Module):
    def __init__(
        self,
        npoint: int = 12306,
        sample_point: int = 1024,
        ts: int = 4,
        dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = TextureEncoder(npoint=npoint,sample_point=sample_point,ts=ts, dim=dim)
        self.decoder = TextureDecoder(dim=dim, npoint=npoint, ts=ts)
        self.init_weights()

    def encode(self, x: Tensor):
        z = self.encoder.forward(x)
        return z

    def decode(self, z: Tensor):
        x = self.decoder.forward(z)
        return x

    def forward(self, x: Tensor):
        x = self.decode(self.encode(x))
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextureAutoEncoder(
        ts=4,
        npoint=12306,
        dim=512,
        device=device,
    )
    model.to(device)

    image = torch.rand([1, 3, 224, 224]).to(device)
    texture = torch.rand([1, 12306, 4, 4, 4, 3]).to(device)
    model.forward(texture, image)
