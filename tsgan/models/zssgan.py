import sys
import os
# sys.path.insert(0, os.path.abspath('../'))

import torch
import torchvision.transforms as transforms

import numpy as np

from .sg2_model import Generator, Discriminator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class SG2Generator(torch.nn.Module):
    def __init__(
        self, checkpoint_path, latent_size=512, map_layers=8, img_size=256, channel_multiplier=2, device='cuda:0'
    ):
        super(SG2Generator, self).__init__()

        self.generator = Generator(img_size, latent_size, map_layers, channel_multiplier=channel_multiplier).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.generator.load_state_dict(checkpoint["g_ema"], strict=True)

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):

        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1 : 3] + list(self.get_all_layers()[4][2 : 10])
        if phase == 'shape':
            # layers 1-2
            return list(self.get_all_layers())[1 : 3] + list(self.get_all_layers()[4][0 : 2])
        if phase == 'no_fine':
            # const + layers 1-10
            return list(self.get_all_layers())[1 : 3] + list(self.get_all_layers()[4][: 10])
        if phase == 'shape_expanded':
            # const + layers 1-10
            return list(self.get_all_layers())[1 : 3] + list(self.get_all_layers()[4][0 : 3])
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers()
        else:
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1 : 3] + list(self.get_all_layers()[4][:])

    def trainable_params(self):
        params = []
        for layer in self.get_training_layers():
            params.extend(layer.parameters())

        return params

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def style(self, styles):
        '''
        Convert z codes to w codes.
        '''
        styles = [self.generator.style(s) for s in styles]
        return styles

    def get_s_code(self, styles, input_is_latent=False):
        return self.generator.get_s_code(styles, input_is_latent)

    def modulation_layers(self):
        return self.generator.modulation_layers

    #TODO Maybe convert to kwargs
    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        input_is_s_code=False,
        noise=None,
        randomize_noise=True
    ):
        return self.generator(
            styles,
            return_latents=return_latents,
            truncation=truncation,
            truncation_latent=self.mean_latent,
            noise=noise,
            randomize_noise=randomize_noise,
            input_is_latent=input_is_latent,
            input_is_s_code=input_is_s_code
        )


class SG2Discriminator(torch.nn.Module):
    def __init__(self, checkpoint_path, img_size=256, channel_multiplier=2, device='cuda:0'):
        super(SG2Discriminator, self).__init__()

        self.discriminator = Discriminator(img_size, channel_multiplier=channel_multiplier).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.discriminator.load_state_dict(checkpoint["d"], strict=True)

    def get_all_layers(self):
        return list(self.discriminator.children())

    def get_training_layers(self):
        return self.get_all_layers()

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def forward(self, images):
        return self.discriminator(images)


class ZSSGAN(torch.nn.Module):
    """ Zero-shot """
    def __init__(self, args, device="cuda"):
        super(ZSSGAN, self).__init__()

        self.device = device

        # Set up twin generators
        self.generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size).to(device)

        # freeze relevant (ALL) layers
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()