from turtle import forward
from typing import Tuple
import torch
from torch import Tensor, optim, nn

import clip
from clip.model import CLIP

try:
    from .encoder import TextureEncoder
    from .decoder import TextureDecoder
except ImportError as e:
    from encoder import TextureEncoder
    from decoder import TextureDecoder


class TextureAutoEncoder(nn.Module):
    def __init__(
        self,
        num_feature: int = 4,
        num_points: int = 12306,
        latent_dim: int = 256,
                                 # device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()

        self.encoder = TextureEncoder(
            num_feature=num_feature,
            latent_dim=latent_dim,
        )
        # image_encoder, preprocess = clip.load("ViT-B/32")
        # self.image_encoder: CLIP = image_encoder
        # self.image_latent_dim = 256

        self.decoder = TextureDecoder(
            latent_dim=latent_dim,
            num_points=num_points,
            num_feature=num_feature,
        )

        self.criterion = nn.L1Loss()

    def encode(self, x: Tensor):
        textuer_latent, _ = self.encoder.forward(x)
        return textuer_latent

    def get_ae_loss(self, x: Tensor) -> Tuple[Tensor]:
        # [ B, N, Ts, Ts, Ts, C ]
        latent_x, _ = self.encoder(x)
        rec_x = self.decoder(latent_x)
        loss = self.criterion(x, rec_x)
        return (loss, rec_x)


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
