import os
import numpy as np
from models import *
from PIL import Image
import torch
from torch import optim
from torch.backends import cudnn
from skimage.metrics import peak_signal_noise_ratio
from modules.models.decoder import PointwiseNet
from modules.models.encoder import TextureEncoder

from modules.render.render import NeuralRenderer

cudnn.enabled = True
cudnn.benchmark = True


class Args:
    obj_model = 'data/models/vehicle-YZ.obj'
    selected_faces = 'data/models/selected_faces.txt'
    device = "cuda"
    sigma = 25
    sigma_ = sigma / 255.
    reg_noise_std = 1. / 30. # set to 1./20. for sigma=50
    lr = 0.01
    exp_weight = 0.99
    num_iter = 3000


def main():
    device = Args.device
    texture_size = 4
    with open(Args.selected_faces, 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        Args.obj_model,
        selected_faces=selected_faces,
        texture_size=texture_size,
        image_size=800,
        device=device,
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)

    encoder = TextureEncoder(num_feature=texture_size, zdim=256).to(device)
    decoder = PointwiseNet(out_dim=648, zdim=256).to(device)

    # Compute number of parameters
    total = sum([param.nelement() for param in encoder.parameters()])
    print('Number of encoder params: %.2fM' % (total / 1e6))
    total = sum([param.nelement() for param in decoder.parameters()])
    print('Number of decoder params: %.2fM' % (total / 1e6))

    # Loss
    criterion_encoder = torch.nn.MSELoss().to(device)

    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=Args.lr)
    for epoch in range(Args.num_iter):
        optimizer.zero_grad()

        # encoding
        noise_latent, _ = encoder.forward(neural_renderer.textures)
        g_noise = torch.randn(noise_latent.size()).to(noise_latent.device)
        loss_encoder = criterion_encoder.forward(noise_latent, g_noise)

        # decoding / generating
        noise_latent = noise_latent + (noise_latent.normal_() * Args.reg_noise_std)

        out = decoder.forward(noise_latent)
        print(out.size())
        exit()


if __name__ == "__main__":
    main()