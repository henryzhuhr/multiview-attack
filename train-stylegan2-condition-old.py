import argparse
import math
import random
import os
import cv2

import numpy as np
import neural_renderer as nr
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import tsgan
from tsgan.render import NeuralRenderer

try:
    import wandb

except ImportError:
    wandb = None

from tsgan.utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from tsgan.models.op import conv2d_gradfix
from tsgan.models.non_leaking import augment, AdaptiveAugment
from tsgan.models.stylegan2 import Generator, Discriminator
from tsgan.models.classifer import resnet50

cstr = lambda s:f"\033[01;32m{s}\033[0m"


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad, = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()
    # path_penalty = F.mse_loss(path_lengths ,path_mean)

    return path_penalty, path_mean.detach(), path_lengths


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, ):
    npoint = 12306
    num_feature = 4
    args.latent_dim = 512

    # ----------------------------------------------
    #   Load Data
    # ----------------------------------------------
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
        ]
    )

    mix_train_set = tsgan.data.CroppedCOCOCarlaMixDataset(
        'configs/dataset.yaml',
        is_train=False,                                    # TODO: 测试完后，修改为训练 True
        transform=transform,
        show_detail=True,
    )

    loader = data.DataLoader(
        mix_train_set,
        batch_size=args.batch,
        num_workers=args.num_workers,
        sampler=data_sampler(mix_train_set, shuffle=True, distributed=args.distributed),
        collate_fn=tsgan.data.CroppedCOCOCarlaMixDataset.collate_fn,
        drop_last=True,
    )
    
    # ----------------------------------------------
    #   Load Neural Renderer
    # ----------------------------------------------
    with open('data/models/selected_faces.txt', 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=selected_faces,
        texture_size=num_feature,
        image_size=800,
        device=device,
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    # neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)
    print('get textures size:', neural_renderer.textures.size())

    # ----------------------------------------------
    #   Load Pretrained Autoencoder & Classifier
    # ----------------------------------------------
    encoder = tsgan.models.autoencoder.TextureEncoder(
        num_feature=num_feature,
        latent_dim=args.latent_dim,
    )
    decoder = tsgan.models.autoencoder.TextureDecoder(
        latent_dim=args.latent_dim,
        num_points=npoint,
        num_feature=num_feature,
    )
    autoencoder_pretrained = torch.load(args.autoencoder_pretrained, map_location="cpu")
    encoder.load_state_dict(autoencoder_pretrained['encoder'])
    decoder.load_state_dict(autoencoder_pretrained['decoder'])
    encoder.eval()
    decoder.eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    cond_model = resnet50()
    cond_model.fc = nn.Linear(
        cond_model.fc.in_features,
        mix_train_set.COCO_CATEGORIES_MAP.__len__(),
    )
    cond_model.load_state_dict(torch.load(args.classifier_pretrained, map_location="cpu"))
    cond_model.to(device)

    with torch.no_grad():
        real_texture = neural_renderer.textures
        texture_latent = encoder.forward(real_texture)
        batch_real_texture = real_texture.repeat(args.batch, *[1] * (len(real_texture.size()) - 1))


    generator = Generator(
        args.size,
        npoint=npoint,
        num_feature=num_feature,
        style_dim=args.latent,
        conditiom_latent_dim=cond_model.fc.in_features,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(
        args.size,
        npoint=npoint,
        num_feature=num_feature,
        style_dim=args.latent,
        conditiom_latent_dim=cond_model.fc.in_features,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0**g_reg_ratio, 0.99**g_reg_ratio))
    d_optim = optim.Adam(
        discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0**d_reg_ratio, 0.99**d_reg_ratio)
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        generator.load_state_dict(ckpt["g"], strict=False)
        discriminator.load_state_dict(ckpt["d"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    # ----------------------------------------------
    # ---> start training
    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5**(32 / (10 * 1000))
    r_t_stat = 0

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    sample_save_dir = os.path.join("tmp", args.save_dir, "sample")
    os.makedirs(sample_save_dir, exist_ok=True)
    checkpoint_save_dir = os.path.join("tmp", args.save_dir, "checkpoint")
    os.makedirs(checkpoint_save_dir, exist_ok=True)

    loader = sample_data(loader)
    for idx in pbar:
        i = idx + args.start_iter
        mix_batch_data = next(loader)
        batch_coco= mix_batch_data["coco"]
        batch_carla = mix_batch_data["carla"]

        cond_images = batch_coco["image"].to(device)
        label = batch_coco["predict_id"].to(device)

        carla_scene_image = batch_carla["image"]

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # ----------------------------------------
        # perform backward
        # ----------------------------------------
        texture_latent_mix_noise = texture_latent.repeat(args.batch, 1)
        for i_b in range(args.batch):
            noise_rotio = 0.001
            noise = torch.randn_like(texture_latent_mix_noise[i_b])
            texture_latent_mix_noise[i_b] = (1 - noise_rotio) * texture_latent_mix_noise[i_b] + noise_rotio * noise

        with torch.no_grad():
            cond_latent = cond_model.forward_latent(cond_images)

        # ----------------------------------------
        # perform backward
        # ----------------------------------------
        fake_texture = generator.forward(texture_latent_mix_noise, cond_latent) # TODO: Inference
        real_pred = discriminator.forward(batch_real_texture)
        fake_pred = discriminator.forward(fake_texture)

        # image = cv2.imread(f'tmp/data/images/Town10HD-point_0000-distance_000-direction_1.png')
        # (
        #         rgb_images,
        #         depth_images,
        #         alpha_images,
        # ) = neural_renderer.renderer.forward(
        #     neural_renderer.vertices,
        #     neural_renderer.faces,
        #     torch.tanh(fake_texture),
        # )
        # rgb_image: torch.Tensor = rgb_images[0]
        # rgb_img: np.ndarray = rgb_image.detach().cpu().numpy() * 255
        # rgb_img = rgb_img.transpose(1, 2, 0)

        # alpha_image: torch.Tensor = alpha_images[0]
        # alpha_channel: np.ndarray = alpha_image.detach().cpu().numpy()

        # render_image = np.zeros(rgb_img.shape)
        # for x in range(alpha_channel.shape[0]):
        #     for y in range(alpha_channel.shape[1]):
        #         alpha = alpha_channel[x][y]
        #         render_image[x][y] = alpha * rgb_img[x][y] + (1 - alpha) * image[x][y]
        # cv2.imwrite(os.path.join(sample_save_dir, f'{i}.png'), render_image)

        if i%args.d_loss_every==0:
            # D logistic loss
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict["d"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            # D backward
            discriminator.zero_grad()
            d_loss.backward()
            d_optim.step()

        # ----------------------------------------
        #   D regularization for every d_reg_every iterations
        # ----------------------------------------
        if i % args.d_reg_every == 0:
            batch_real_texture.requires_grad = True
            real_pred = discriminator(batch_real_texture)
            r1_loss = d_r1_loss(real_pred, batch_real_texture)
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()

        loss_dict["r1"] = r1_loss

        # frozen D
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # G backward
        fake_texture = generator.forward(texture_latent_mix_noise, cond_latent)

        fake_pred = discriminator(fake_texture)
        g_loss = g_nonsaturating_loss(fake_pred)
        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        # ----------------------------------------
        #   G regularization for every g_reg_every iterations
        # ----------------------------------------
        # if i % args.g_reg_every == 0:
        #     fake_texture, latents = generator(texture_latent_mix_noise,cond_latent, return_latents=True)
        #     path_loss, mean_path_length, path_lengths = g_path_regularize(fake_texture, latents, mean_path_length)
        #     generator.zero_grad()

        #     weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
        #     # if args.path_batch_shrink:
        #     #     weighted_path_loss += 0 * fake_texture[0, 0, 0, 0]
        #     weighted_path_loss.backward()
        #     g_optim.step()
        #     mean_path_length_avg = (reduce_sum(mean_path_length).item() / get_world_size())
        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (   
                    f"{cstr('epoch')}:{i} "
                    f"{cstr('D')}:{d_loss_val:.4f} "
                    f"{cstr('G')}:{g_loss_val:.4f} "
                    f"{cstr('r1')}:{r1_val:.4f} "
                    f"{cstr('path')}:{path_loss_val:.4f} "
                    f"{cstr('mpath')}:{mean_path_length_avg:.4f} "
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            # if i % 100 == 0:
            #     with torch.no_grad():
            #         g_ema.eval()
            #         sample = g_ema(texture_latent_mix_noise,cond_latent)
            #         # utils.save_image(
            #         #     sample,
            #         #     f"{sample_save_dir}/{str(i).zfill(6)}.png",
            #         #     nrow=int(args.n_sample**0.5),
            #         #     normalize=True,
            #         #     range=(-1, 1),
            #         # )

            if i % 500 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                    },
                    f"{checkpoint_save_dir}/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--save_dir', type=str, default='stylegan2')
    parser.add_argument("--iter", type=int, default=20000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=1, help="batch sizes for each gpus")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=64, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=1024, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)"
    )
    parser.add_argument("--d_loss_every", type=int, default=10, help="interval of the r1 regularization")
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the path length regularization")
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1"
    )
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")

    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")

    parser.add_argument('--obj_model', type=str, default='data/models/vehicle-YZ.obj')
    parser.add_argument('--autoencoder_pretrained', type=str, default='tmp/autoencoder.pt')
    parser.add_argument('--classifier_pretrained', type=str, default='tmp/classifier/resnet50-4.pt')

    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    train(args)
