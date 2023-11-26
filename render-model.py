import abc
import os
import json
import argparse

import tqdm
import cv2
import numpy as np

import torch
from models.data.carladataset import CarlaDataset
from models.render import NeuralRenderer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj_model",
        type=str,
        default="assets/aircraft_carrier/aircraft_carrier-_YZ.obj",
    )
    parser.add_argument(
        "--selected_faces", type=str, default="assets/faces-audi-std.txt"
    )
    parser.add_argument("--texture_size", type=int, default=25)
    parser.add_argument("--scence_dir", type=str, default="data/samples")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


class TypeArgs(metaclass=abc.ABCMeta):
    __metaclass__ = abc.ABCMeta
    obj_model: str
    selected_faces: str
    texture_size: int
    scence_image: str
    scence_label: str


def main():
    args: TypeArgs = get_args()
    # Load Neural Renderer
    ts = args.texture_size
    with open(args.selected_faces, "r") as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split("\n")]
    neural_renderer = NeuralRenderer(
        args.obj_model,
        selected_faces=None,
        texture_size=ts,
        image_size=640,
        device=args.device,
    )
    os.makedirs("images", exist_ok=True)
    scence_dir: str = args.scence_dir
    for file in tqdm.tqdm(os.listdir(scence_dir)):
        if file.endswith(".json"):
            label_file = os.path.join(scence_dir, file)

            with open(label_file, "r") as f:
                label_dict = json.load(f)
            name = label_dict["name"]
            image_file = os.path.join(os.path.dirname(label_file), name + ".png")
            if os.path.exists(image_file):
                img = render_once(neural_renderer, image_file, label_dict)
                h,s,v = cv2.split(cv2.cvtColor(img,cv2.COLOR_BGR2HSV))
                v1=np.clip(cv2.add(1*v,60),0,255)
                img = cv2.cvtColor(np.uint8(cv2.merge((h,s,v1))),cv2.COLOR_HSV2BGR)
                cv2.imwrite(os.path.join("images", name + "-render_bg-ts_4.png"), img)


def render_once(neural_renderer: NeuralRenderer, image_file: str, label_dict: str):
    image = cv2.imread(image_file)
    image = cv2.imread("data/background.jpg")
    image = cv2.resize(image, (640, 640))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vehicle_transform = CarlaDataset.convert_dict_transform(label_dict["vehicle"])
    camera_transform = CarlaDataset.convert_dict_transform(label_dict["camera"])
    fov = label_dict["camera"]["fov"]

    neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)

    x_t = neural_renderer.textures  # [:, selected_faces, :]
    img = render_a_image(neural_renderer, image, x_t)
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("images/render.png", img)
    return img


def render_a_image(neural_renderer: NeuralRenderer, image: cv2.Mat, x: torch.Tensor):
    # x_full = torch.zeros_like(neural_renderer.textures)
    x_full = neural_renderer.textures
    # x_full[:, neural_renderer.selected_faces, :] = x

    rgb_images, _, alpha_images = neural_renderer.renderer.forward(
        neural_renderer.vertices, neural_renderer.faces, torch.tanh(x_full)
    )
    rgb_img: cv2.Mat = (
        (rgb_images[0].detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    )
    alpha_channel: cv2.Mat = alpha_images[0].detach().cpu().numpy()

    render_image = np.zeros(rgb_img.shape)
    for x in range(alpha_channel.shape[0]):
        for y in range(alpha_channel.shape[1]):
            alpha = alpha_channel[x][y]
            render_image[x][y] = alpha * rgb_img[x][y] + (1 - alpha) * image[x][y]
    return render_image


if __name__ == "__main__":
    main()
