import argparse
from ast import arg

import os
import random
import tempfile

import numpy as np
import torch

from torch import distributed
from torch import multiprocessing
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import tsgan
from tsgan.utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from tsgan.utils import training_stats

from tsgan.models.sg2_model import Generator, Discriminator

torch.backends.cudnn.benchmark = True # Improves training speed.


def get_args():

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    # parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=64, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)"
    )
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
    parser.add_argument(
        "--g_reg_every", type=int, default=4, help="interval of the applying path length regularization"
    )
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1"
    )
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation"
    )
    parser.add_argument(
        "--ada_target", type=float, default=0.6, help="target augmentation probability for adaptive augmentation"
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation"
    )
    parser.add_argument(
        "--ada_every", type=int, default=256, help="probability update interval of the adaptive augmentation"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    # parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def main():
    set_seed(42)

    args = get_args()
    device = "cuda"
    args.latent = 512
    args.n_mlp = 8
    args.start_iter = 0

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if get_rank() == 0:
        print("n_gpu:", n_gpu)
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0**g_reg_ratio, 0.99**g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0**d_reg_ratio, 0.99**d_reg_ratio),
    )

    if args.ckpt is not None:
        if get_rank() == 0:
            print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=False)
        discriminator.load_state_dict(ckpt["d"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        print("Distributed training")
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
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    if get_rank() == 0:
        print("Loading data")
    croppedcoco_train_set = tsgan.data.CroppedCOCO(config_file='configs/coco.yaml', is_train=False)
    croppedcoco_train_loader = data.DataLoader(
        croppedcoco_train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=data.DistributedSampler(
            croppedcoco_train_set,
            shuffle=True,
        ),
        drop_last=True,
    )

    if get_rank() == 0:
        print("Start training")
    train(
        args,
        croppedcoco_train_loader,
        generator,
        discriminator,
        g_optim,
        d_optim,
        g_ema,
        device,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)
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
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

#----------------------------------------------------------------------------

class AdaptiveAugment:
    def __init__(self, ada_aug_target, ada_aug_len, update_every, device):
        self.ada_aug_target = ada_aug_target
        self.ada_aug_len = ada_aug_len
        self.update_every = update_every

        self.ada_update = 0
        self.ada_aug_buf = torch.tensor([0.0, 0.0], device=device)
        self.r_t_stat = 0
        self.ada_aug_p = 0

    @torch.no_grad()
    def tune(self, real_pred):
        self.ada_aug_buf += torch.tensor(
            (torch.sign(real_pred).sum().item(), real_pred.shape[0]),
            device=real_pred.device,
        )
        self.ada_update += 1

        if self.ada_update % self.update_every == 0:
            self.ada_aug_buf = reduce_sum(self.ada_aug_buf)
            pred_signs, n_pred = self.ada_aug_buf.tolist()

            self.r_t_stat = pred_signs / n_pred

            if self.r_t_stat > self.ada_aug_target:
                sign = 1

            else:
                sign = -1

            self.ada_aug_p += sign * n_pred / self.ada_aug_len
            self.ada_aug_p = min(1, max(0, self.ada_aug_p))
            self.ada_aug_buf.mul_(0)
            self.ada_update = 0

        return self.ada_aug_p
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


if __name__ == "__main__":
    main()