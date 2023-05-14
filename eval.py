import os
import argparse
import datetime
import tqdm
import yaml

import numpy as np
import cv2
import torch

from models.gan import TextureGenerator
from models.render import NeuralRenderer
from models.data.carladataset import CarlaDataset
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.general import non_max_suppression

cstr = lambda s: f"\033[01;32m{s}\033[0m"
logt = lambda: "\033[01;32m{%d}\033[0m" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='stylegan2')
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--mix_prob", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--lr", type=float, default=0.002)

    parser.add_argument('--obj_model', type=str, default="assets/vehicle-YZ.obj")
    parser.add_argument('--selected_faces', type=str, default="assets/faces-std.txt")
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--latent_dim', type=int, default=1024)
    # parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default="tmp/attack-dog-05131427/checkpoint/gan-98.pt")

    return parser.parse_args()


class ArgsType:
    save_dir: str
    epochs: int
    batch: int
    num_workers: int
    size: int

    mix_prob: float
    lr: float

    obj_model: str
    selected_faces: str
    texture_size: int
    latent_dim: int
    pretrained: str


def main():
    args: ArgsType = get_args()
    save_dir = "tmp/eval"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda"
    data_set = CarlaDataset(carla_root="tmp/data", categories=["dog", "car"])

    # --- Load Neural Renderer ---
    with open(args.selected_faces, 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=selected_faces,
        texture_size=args.texture_size,
        image_size=800,
        device=device,
    )
    # --- Load Texture Generator ---
    model = TextureGenerator(
        nt=len(selected_faces),
        ts=args.texture_size,
        style_dim=args.latent_dim,
        cond_dim=len(data_set.coco_ic_map),
        mix_prob=args.mix_prob
    )
    pretrained = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(pretrained["model"])
    model.to(device).eval()

    # --- Load Detector ---
    with open("configs/hyp.scratch-low.yaml", "r") as f:
        hyp: dict = yaml.safe_load(f)
    nc = 80
    detector = Model("configs/yolov5s.yaml", ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
    detector.nc = nc
    detector.hyp = hyp
    detector_loss = ComputeLoss(detector)
    ckpt = torch.load("pretrained/yolov5s.pt", map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    detector.load_state_dict(csd, strict=False)
    detector.eval()
    conf_thres, iou_thres = 0.25, 0.6

    x_t = neural_renderer.textures[:, neural_renderer.selected_faces, :] # x_{texture}
    n_r = 0.                                                             # noise ratio
    x_n = (1 - n_r) * x_t + n_r * torch.rand_like(x_t)                   # x_{texture with noise}

    pbar = tqdm.tqdm(data_set)
    for i_d, item in enumerate(pbar):
        image = item["image"].to(device)
        label = item["label"].to(device)
        r_p = {"ct": item["ct"], "vt": item["vt"], "fov": item["fov"]}

        with torch.no_grad():
            x_adv = model.decode(model.forward(x_t, label)) # x_{adv}
            render_image, _, _, render_img = render_a_image(neural_renderer, x_adv, image, r_p)

            eval_pred, train_preds = detector.forward(render_image) # real

            pred_results = non_max_suppression(eval_pred, conf_thres, iou_thres, None, False)[0]
            if len(pred_results):
                for *xyxy, conf, cls in pred_results:
                    label = '%s %.2f' % (data_set.coco_ic_map[int(cls)], conf)
                    x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                    cv2.rectangle(render_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(render_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_dir, f"eval-{i_d}.png"), render_img)


def render_a_image(
    neural_renderer: NeuralRenderer, x_texture: torch.Tensor, base_image: torch.Tensor, render_params: dict
):
    tt_adv = neural_renderer.textures
    tt_adv[:, neural_renderer.selected_faces, :] = x_texture
    neural_renderer.set_render_perspective(render_params["ct"], render_params["vt"], render_params["fov"])
    rgb_image, _, alpha_image = neural_renderer.forward(torch.tanh(tt_adv))
    render_image = alpha_image * rgb_image + (1 - alpha_image) * base_image
    render_img = np.ascontiguousarray(render_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255)
    return render_image, rgb_image, alpha_image, render_img.astype(np.uint8)


if __name__ == "__main__":
    main()