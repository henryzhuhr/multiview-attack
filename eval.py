import json
from nis import cat
import os
import argparse
import datetime
from typing import List
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
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--mix_prob", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.002)

    parser.add_argument('--world_map', type=str, default="Town10HD")
    parser.add_argument('--pretrained', type=str, default="tmp/train-person-05221615/checkpoint/_generator.pt")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


class ArgsType:
    device: str
    num_workers: int
    size: int

    mix_prob: float
    lr: float
    milestones: List[int]

    world_map: str
    pretrained: str


COLOR_MAP = [
    (0, 255, 0),   # Green
    (255, 255, 0), # Cyan
    (255, 0, 255), # Purple
    (0, 255, 255), # Yellow
    (255, 0, 0),   # Red
]


def main():
    args: ArgsType = get_args()

    pretrained = torch.load(args.pretrained, map_location='cpu')
    pargs = vars(pretrained["args"]) # pretrained args
    ts = pargs["texture_size"]
    cats: List[str] = pargs["categories"]
    obj_model = pargs["obj_model"]
    latent_dim = pargs["latent_dim"]
    mix_prob = pargs["mix_prob"]

    nowt = datetime.datetime.now().strftime("%m%d%H%M")
    base_save_dir = f"tmp/eval-{args.world_map}-{'_'.join(cats)}-{nowt}"
    os.makedirs(base_save_dir, exist_ok=True)

    device = args.device
    data_set = CarlaDataset(carla_root=f"tmp/data-maps/{args.world_map}", categories=cats, is_train=False)
    num_classes = len(data_set.coco_ic_map)

    # --- Load Neural Renderer ---
    with open(pargs["selected_faces"], 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        obj_model,
        selected_faces=selected_faces,
        texture_size=ts,
        image_size=800,
        device=args.device,
    )
    # --- Load Texture Generator ---
    model = TextureGenerator(
        nt=len(selected_faces), ts=ts, style_dim=latent_dim, cond_dim=num_classes, mix_prob=mix_prob
    )
    model.load_state_dict(pretrained["model"])
    model.to(device).eval()

    # --- Load Detector ---
    with open("configs/hyp.scratch-low.yaml", "r") as f:
        hyp: dict = yaml.safe_load(f)
    detector = Model("configs/yolov5s.yaml", ch=3, nc=num_classes, anchors=hyp.get('anchors')).to(device)
    detector.nc = num_classes
    detector.hyp = hyp
    detector_loss = ComputeLoss(detector)
    ckpt = torch.load("pretrained/yolov5s.pt", map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    detector.load_state_dict(csd, strict=False)
    detector.eval()
    conf_thres, iou_thres = 0.25, 0.6

    n_r = 0.1      # noise ratio

    x_t = (neural_renderer.textures[:, neural_renderer.selected_faces, :]).clone() # x_{texture}
    x_n = torch.rand_like(x_t)
    x_i = (1 - n_r) * x_t + n_r * x_n                                              # x_{texture with noise}

    id_name_maps = {}
    pbar = tqdm.tqdm(data_set)
    for i_d, item in enumerate(pbar):
        
        pbar.set_description(f"[{i_d}] {item['file']}")
        image = item["image"].to(device)
        r_p = {"ct": item["ct"], "vt": item["vt"], "fov": item["fov"]}

        class_concat_imgs = []
        for cat in cats:
            label = torch.tensor(data_set.coco_ci_map[cat]).unsqueeze(0).to(device)

            image_list = []
            bboxes_list = []
            for adv_type, x_ in {
                "org": x_t,
                "clean": x_t,
                "noise": x_n,
                "adv": x_i,
            }.items():
                with torch.no_grad():
                    if adv_type == "adv":
                        x_ = model.decode(model.forward(x_, label))         # x_{adv}
                    render_image, _, _, render_img = render_a_image(neural_renderer, x_, image.clone(), r_p)
                    eval_pred, train_preds = detector.forward(render_image) # real
                    pred_results = non_max_suppression(eval_pred, conf_thres, iou_thres, None, False)[0]

                atk_class = data_set.coco_ic_map[int(label.item())]
                detect_img = render_img.copy()
                bboxes = []
                w, h = detect_img.shape[: 2]
                if len(pred_results):
                    for *xyxy, conf, category in pred_results:
                        pclass = data_set.coco_ic_map[int(category)]
                        text = f'{pclass}:{conf:.2f}'
                        x1, y1, x2, y2 = [int(xy) for xy in xyxy]

                        if pclass in cats:
                            color = COLOR_MAP[1 + cats.index(pclass)]
                        elif pclass == "car":
                            color = COLOR_MAP[0]
                        else:
                            color = (255, 255, 255)
                        cv2.rectangle(detect_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(detect_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        bboxes.append(
                            [0, int(category), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
                        )
                if adv_type == "org":
                    image_list.append(render_img)
                else:
                    image_list.append(detect_img)
                if adv_type == "clean":
                    bboxes_list.append(bboxes)
            cat_render_img = cv2.vconcat(image_list)
            class_concat_imgs.append(cat_render_img)

            os.makedirs(ic_save_dir := f"{base_save_dir}/index", exist_ok=True) # index-class
            cv2.imwrite(f"{ic_save_dir}/{i_d}-{atk_class}.png", cat_render_img)

            os.makedirs(ci_save_dir := f"{base_save_dir}/class", exist_ok=True) # class-index
            cv2.imwrite(f"{ci_save_dir}/{atk_class}-{i_d}.png", cat_render_img)

            os.makedirs(cc_save_dir := f"{base_save_dir}/{atk_class}", exist_ok=True) # class
            cv2.imwrite(f"{cc_save_dir}/{i_d}.png", cat_render_img)

            os.makedirs(single_save_dir := f"{base_save_dir}/single", exist_ok=True) # class
            for i_img, img in enumerate(image_list):
                cv2.imwrite(f"{single_save_dir}/{i_d}-{atk_class}-{i_img}.png", img)

            id_name_maps[i_d] = {
                "file": item["file"],
                "bboxes": bboxes_list,
            }


        concat_img = cv2.hconcat(class_concat_imgs)
        os.makedirs(aa_save_dir := f"{base_save_dir}/concat", exist_ok=True) # class
        cv2.imwrite(f"{aa_save_dir}/{i_d}.png", concat_img)
        cv2.imwrite("tmp/__eval__.png", concat_img)

    with open(f"{base_save_dir}/id_name_maps.json", "w") as f:
        json.dump(id_name_maps, f, indent=4)

def render_a_image(
    neural_renderer: NeuralRenderer, x_texture: torch.Tensor, base_image: torch.Tensor, render_params: dict
):
    tt_adv = neural_renderer.textures.clone()
    # tt_adv =torch.zeros_like(neural_renderer.textures)
    tt_adv[:, neural_renderer.selected_faces, :] = x_texture
    neural_renderer.set_render_perspective(render_params["ct"], render_params["vt"], render_params["fov"])
    rgb_image, _, alpha_image = neural_renderer.forward(torch.tanh(tt_adv))
    render_image = alpha_image * rgb_image + (1 - alpha_image) * base_image
    render_img = np.ascontiguousarray(render_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255)
    return render_image, rgb_image, alpha_image, render_img.astype(np.uint8)


if __name__ == "__main__":
    main()