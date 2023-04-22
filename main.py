import argparse, os, sys, datetime, glob, importlib, csv
from ast import main
from typing import List
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
# from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_info

import tsgan


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--name", type=str, default="ldm")
    parser.add_argument(
        "--resume", type=str, const=True, default="", nargs="?", help="resume from logdir or checkpoint in logdir"
    )
    # parser.add_argument(
    #     "--base",
    #     nargs="*",
    #     metavar="configs/autoencoder_kl_8x8x64.yaml",
    #     help="paths to base configs. Loaded from left-to-right. "
    #     "Parameters can be overwritten or added with command-line options of the form `--key value`.",
    #     default=list()
    # )
    parser.add_argument(
        "--base",
        type=List[str],
        default=[
            "configs/autoencoder_texture.yaml",
        ],
    )
    parser.add_argument("--train", default=False)
    parser.add_argument("--no-test", type=bool, default=False, help="disable test")
    parser.add_argument("--project", help="name of new or path to existing project")
    parser.add_argument("--debug", type=bool, default=False, help="enable post-mortem debugging")
    parser.add_argument("--logdir", type=str, default="logs", help="directory for logging dat shit")
    parser.add_argument("--scale_lr", type=bool, default=True, help="scale base-lr by ngpu * batch_size * n_accumulate")

    # SEED
    parser.add_argument("--seed", type=int, default=23, help="seed for seed_everything")
    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def main():

    sys.path.append(os.getcwd())

    parser = get_parser()

    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    if opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = f"{opt.name}-{cfg_name}"
    else:
        name = opt.name

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logdir = os.path.join(opt.logdir, f"{name}-{now}")

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        config = OmegaConf.merge(*configs)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # default to ddp
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config


        # model
        model=tsgan.models.utils.instantiate_from_config(config.model)


    except Exception:
        # if opt.debug and trainer.global_rank == 0:
        #     try:
        #         import pudb as debugger
        #     except ImportError:
        #         import pdb as debugger
        #     debugger.post_mortem()
        print("exit")
        raise


if __name__ == "__main__":
    main()