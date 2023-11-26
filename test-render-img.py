import abc
import datetime
from time import sleep
from typing import Dict, List
import os, sys
import json
import argparse
import cv2
import numpy as np

import torch
import tqdm
import yaml
from models.data.carladataset import CarlaDataset
from models.gan import TextureGenerator
from models.render import NeuralRenderer
from models.data import types
import neural_renderer as nr


from models.yolo import Model
from utils.general import non_max_suppression
from utils.loss import ComputeLoss


class TypeArgs(metaclass=abc.ABCMeta):
    obj_model: str
    world_map: str
    texture_size: int
    data_dir: str
    device: str
    pretrained: str
    nowt: str


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_model', type=str, default='assets/audi.obj')
    parser.add_argument('--data_dir', type=str, default="data/train")
    parser.add_argument('--world_map', type=str, default="Town05")
    parser.add_argument('--texture_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--pretrained', type=str, default="tmp/train/physicalAttak-dog-06211610/checkpoint/_generator.pt")
    parser.add_argument('--pretrained', type=str, default="tmp/train/physicalAttak-kite-06212302/checkpoint/_generator.pt")
    parser.add_argument('--name', type=str)
    parser.add_argument('--nowt', type=str)
    return parser.parse_args()


def main():
    args: TypeArgs = get_args()
    device = torch.device(args.device)

    if args.nowt:
        nowt = args.nowt
    else:
        nowt = datetime.datetime.now().strftime("%m%d_%H%M")
    
    base_name=f"temps/render-{args.name}/{args.world_map}-{nowt}"
    if args.name:
        os.makedirs(save_dir := f"{base_name}-{args.name}", exist_ok=True)
    else:
        os.makedirs(save_dir := f"{base_name}", exist_ok=True)

    pretrained = torch.load(args.pretrained, map_location='cpu')
    pargs = vars(pretrained["args"]) # pretrained args
    ts = pargs["texture_size"]
    cats: List[str] = pargs["categories"]
    obj_model = pargs["obj_model"]
    latent_dim = pargs["latent_dim"]
    mix_prob = pargs["mix_prob"]

    # --- Load Neural Renderer ---
    with open(pargs["selected_faces"], 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        obj_model,
        selected_faces=selected_faces,
        texture_size=ts,
        image_size=800,
        device=device,
    )
    n_r = 0.05                                                                     # noise ratio
    x_t = (neural_renderer.textures[:, neural_renderer.selected_faces, :]).clone() # x_{texture}
    x_n = torch.rand_like(x_t)
    x_i = (1 - n_r) * x_t + n_r * x_n                                              # x_{texture with noise}

    # --- Load Dataset ---
    data_set = CarlaDataset(carla_root=f"{args.data_dir}/{args.world_map}", categories=[], is_train=False)
    num_classes = len(data_set.coco_ic_map)

    # --- Load Texture Generator ---
    model = TextureGenerator(
        nt=len(neural_renderer.selected_faces), ts=ts, style_dim=latent_dim, cond_dim=num_classes, mix_prob=mix_prob
    )
    model.load_state_dict(pretrained["model"])
    model.to(device).eval()


    # --- DAS ---    
    with open("assets/DAS-faces.txt", 'r') as f:
    # with open(pargs["selected_faces"], 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    nr_DAS = NeuralRenderer(
        obj_model,
        selected_faces=selected_faces,
        texture_size=6,
        image_size=800,
        device=device
    )
    texture_DAS=torch.from_numpy(np.load("assets/DAS-texture.npy")).to(device)
    r, g, b = torch.split(texture_DAS, [1, 1, 1], dim=-1)# 交换颜色通道
    texture_DAS = torch.cat((b,g,r), dim=-1)
    DAS_texture_mask = np.zeros((nr_DAS.faces.shape[1], 6, 6, 6, 3), 'int8')
    for face_id in selected_faces:
        if face_id != '\n':
            DAS_texture_mask[int(face_id)-1, :, :, :, :] = 1
    DAS_texture_mask = torch.from_numpy(DAS_texture_mask).to(device).unsqueeze(0)
    
    # --- FCA ---    
    # with open("assets/FCA-faces.txt", 'r') as f:
    with open(pargs["selected_faces"], 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    nr_FCA = NeuralRenderer(
        obj_model,
        selected_faces=selected_faces,
        texture_size=6,
        image_size=800,
        device=device
    )
    texture_FCA=torch.from_numpy(np.load("assets/FCA-texture.npy")).to(device)
    r, g, b = torch.split(texture_FCA, [1, 1, 1], dim=-1)
    texture_FCA = torch.cat((b,g,r), dim=-1)
    FCA_texture_mask = np.zeros((nr_FCA.faces.shape[1], 6, 6, 6, 3), 'int8')
    for face_id in selected_faces:
        if face_id != '\n':
            FCA_texture_mask[int(face_id)-1, :, :, :, :] = 1
    FCA_texture_mask = torch.from_numpy(FCA_texture_mask).to(device).unsqueeze(0)

    pbar = tqdm.tqdm(data_set)
    for i_d, item in enumerate(pbar):
        # pbar.set_description(f"Processing {i_d}/{len(data_set)}")
        image = item["image"].to(device)
        name = item["name"]
        file = item["file"]
        r_p = {"ct": item["ct"], "vt": item["vt"], "fov": item["fov"]}

        render_img, [x1, y1, x2, y2] = get_bbox(neural_renderer, image, r_p)
        img = btensor2img(image)
        h, w, _ = img.shape
        box = [x1 / w, y1 / h, x2 / w, y2 / h]
        # detect_img = cv2.rectangle(render_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imwrite("temp/test.png", detect_img)
        imgs_list = {}
        if "base":
            _, _, _, render_img = render_a_image(neural_renderer, x_t, image.clone(), r_p)
            imgs_list["clean"] = render_img
            _, _, _, render_img = render_a_image(neural_renderer, x_n, image.clone(), r_p)
            imgs_list["noise"] = render_img
        if "mcc":
            for cat in cats:
                label = torch.tensor(data_set.coco_ci_map[cat]).unsqueeze(0).to(device)
                with torch.no_grad():
                    x_adv = model.decode(model.forward(x_i, label)) # x_{adv}
                _, _, _, render_img = render_a_image(neural_renderer, x_adv, image.clone(), r_p)
                imgs_list[f"mcc-{cat}"] = render_img
        if "DAS":
            t_DAS=nr_DAS.textures * (1 - DAS_texture_mask) + DAS_texture_mask * 0.5 * (torch.tanh(texture_DAS) + 1)
            nr_DAS.set_render_perspective(r_p["ct"], r_p["vt"], r_p["fov"])
            rgb_image, _, alpha_image = nr_DAS.forward(t_DAS)
            render_image = alpha_image * rgb_image + (1 - alpha_image) * image
            render_img = np.ascontiguousarray(render_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255)
            imgs_list["DAS"] = render_img
        if "FCA":
            t_FCA=nr_FCA.textures * (1 - FCA_texture_mask) + FCA_texture_mask * 0.5 * (torch.tanh(texture_FCA) + 1)

            nr_FCA.set_render_perspective(r_p["ct"], r_p["vt"], r_p["fov"])
            rgb_image, _, alpha_image = nr_FCA.forward(t_FCA)
            render_image = alpha_image * rgb_image + (1 - alpha_image) * image
            render_img = np.ascontiguousarray(render_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255)
            imgs_list["FCA"] = render_img


        imgs_file_list={}
        for n, img in imgs_list.items():
            file_name=f"{name}-{n}.png"
            cv2.imwrite(f"{save_dir}/{file_name}", img)
            imgs_file_list[n]=file_name

        with open(f"{save_dir}/{name}.json", "w") as f:
            json.dump(
                {
                    "name": name,
                    "imgs": imgs_file_list,
                    "bbox": box,
                    "size": [h, w],
                }, f, indent=4
            )




def btensor2img(image: torch.Tensor):
    """ batch tensor [1,C,W,H] to image [W,H,C]"""
    return np.ascontiguousarray(image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def get_bbox(neural_renderer: NeuralRenderer, image: torch.Tensor, render_params: dict):
    """ 获取边界框 """
    neural_renderer.set_render_perspective(render_params["ct"], render_params["vt"], render_params["fov"])
    rgb_image, _, alpha_image = neural_renderer.forward(torch.tanh(neural_renderer.textures))
    render_image = alpha_image * rgb_image + (1 - alpha_image) * image
    binary = np.ascontiguousarray(alpha_image.squeeze(0).detach().cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    find_boxes = []
    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        find_boxes.append([x, y, x + w, y + h])
    fc = np.array(find_boxes)
    box = [min(fc[:, 0]), min(fc[:, 1]), max(fc[:, 2]), max(fc[:, 3])] # [x1,y1,x2,y2]
    [x1, y1, x2, y2] = [int(b) for b in box]
    return btensor2img(render_image), [x1, y1, x2, y2]


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


if __name__ == '__main__':
    main()
