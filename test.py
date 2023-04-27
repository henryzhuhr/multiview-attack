import os
from pprint import pprint
import cv2

import numpy as np
from tsgan.data.mixdataset import CroppedCOCOCarlaMixDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import torch
from tsgan.models.stylegan2 import Discriminator
from tsgan.utils import logheader
import tsgan
from tsgan.render import NeuralRenderer
if __name__ == "__main__":
    device = torch.device("cuda")
    dataset = CroppedCOCOCarlaMixDataset(
        "configs/dataset.yaml",
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
    )
    print("category detail:", dataset.categories)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=CroppedCOCOCarlaMixDataset.collate_fn
    )
    batch_size = 4

    # sample_batch = next(iter(dataloader))
    # print()
    # print("sample batch:")
    # pprint(sample_batch)

    with open('data/models/selected_faces.txt', 'r') as f:
        selected_faces = [int(face_id) for face_id in f.read().strip().split('\n')]
    neural_renderer = NeuralRenderer(
        'data/models/vehicle-YZ.obj',
        selected_faces=selected_faces,
        texture_size=4,
        image_size=800,
        device=device,
    )
    neural_renderer.to(neural_renderer.textures.device)
    neural_renderer.renderer.to(neural_renderer.textures.device)
    # neural_renderer.set_render_perspective(camera_transform, vehicle_transform, fov)
    print('get textures size:', neural_renderer.textures.size())

    with torch.no_grad():
        real_texture = torch.rand_like(neural_renderer.textures)
        real_texture = real_texture * neural_renderer.textures_mask
        print(real_texture.size())
        # for face in real_texture[0]:
        #     if (~torch.gt(torch.zeros_like(face).float(),face)).all().bool()==False:
        #         print(face.size())
        batch_real_texture = real_texture.repeat(batch_size, *[1] * (len(real_texture.size()) - 1))

    for batch_data in dataloader:
        cond_images = batch_data["coco"]["image"]
        coco_label = batch_data["coco"]["predict_id"]

        carla_scene_images = batch_data["carla"]["image"]
        carla_render_params = batch_data["carla"]["render_param"]

        for i_b in range(batch_real_texture.size(0)):
            carla_scene_image = carla_scene_images[i_b]
            fake_texture = batch_real_texture[i_b].unsqueeze(0)
            render_texture = fake_texture * neural_renderer.textures_mask + neural_renderer.textures * (
                1 - neural_renderer.textures_mask
            )
            carla_render_param = carla_render_params[i_b]

            neural_renderer.set_render_perspective(
                carla_render_param["camera_transform"],
                carla_render_param["vehicle_transform"],
                carla_render_param["fov"],
            )
            (
                rgb_images,
                depth_images,
                alpha_images,
            ) = neural_renderer.renderer.forward(
                neural_renderer.vertices,
                neural_renderer.faces,
                torch.tanh(render_texture),
            )
            rgb_image: torch.Tensor = rgb_images[0]
            rgb_img: np.ndarray = rgb_image.detach().cpu().numpy() * 255
            rgb_img = rgb_img.transpose(1, 2, 0)

            alpha_image: torch.Tensor = alpha_images[0]
            alpha_channel: np.ndarray = alpha_image.detach().cpu().numpy()

            render_image = np.zeros(rgb_img.shape)
            for x in range(alpha_channel.shape[0]):
                for y in range(alpha_channel.shape[1]):
                    alpha = alpha_channel[x][y]
                    render_image[x][y] = alpha * rgb_img[x][y] + (1 - alpha) * carla_scene_image[x][y]
            cv2.imwrite(os.path.join("tmp/stylegan2-cropcoco_car-0423_1438/sample", f'render.png'), render_image)
