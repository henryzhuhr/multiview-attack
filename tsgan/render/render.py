from typing import List
import numpy as np
import torch
from torch import Tensor, nn

import neural_renderer
import math

from ..types.carla import (Location, Rotation, Transform)


class NeuralRenderer(nn.Module):
    """
    This is the core pytorch function to call.
    Every torch NMR has a chainer NMR.
    Only fwd/bwd once per iteration.
    """
    def __init__(self, obj_model, selected_faces: List[int] = None, texture_size=6, image_size=800, device="cuda"):
        super(NeuralRenderer, self).__init__()

        # 加载模型，返回 顶点, 面, 纹理
        (
            vertices,                  # torch.Size([8795, 3])
            faces,                     # torch.Size([12306, 3])
            textures,                  # torch.Size([12306, 6, 6, 6, 3])
        ) = neural_renderer.load_obj(
            obj_model,                 # obj 模型路径
            texture_size=texture_size, # 渲染纹理尺寸 越大渲染纹理精度越高
            load_texture=True,
        )
        vertices: Tensor = vertices.to(device)
        faces: Tensor = faces.to(device)
        textures: Tensor = textures.to(device)
        self.vertices = vertices.unsqueeze(0).to(device)
        self.faces = faces.unsqueeze(0).to(device)
        self.textures = textures.unsqueeze(0).to(device)

        self.selected_faces = selected_faces
        if selected_faces is not None:
            textures_mask = torch.zeros(textures.size()).to(device)
            for face_id in selected_faces:
                textures_mask[face_id - 1, :, :, :, :] = 1
            render_textures = textures * textures_mask
        else:
            render_textures = textures
        self.render_textures = nn.Parameter(render_textures.unsqueeze(0)).to(device) # 待渲染的纹理 (优化参数)

        # 初始化渲染器
        self.renderer = neural_renderer.Renderer(
            image_size=image_size,                # 渲染输出图像大小
            camera_mode='look',                   # 相机模式
            light_intensity_ambient=0.5,          # 常数化光场，即各个方向的光照均等，因此直接对光源进行增扩即可，即物体每个面片得到的光照均为color_ambient
            light_intensity_directional=0.5,      # 平行光，即有方向的光场，就需要考虑到光场和面片的相互作用了。面片是具有法向和位置的，光场是平行的，因此就是向量的点乘，不是像常数化光场一样的张量乘。
            light_color_ambient=[1, 1, 1],        # 灯光 white
            light_color_directional=[1, 1, 1],    # 灯光 white
            light_direction=[0, 0, 1],            # 灯光 up-to-down
            viewing_angle=45,                     # 观察角度
        ).to(device)

    def forward(self, textures=None):
        ''' Renders masks.
        Args:
            vertices: B X N X 3 numpy array2
            faces: B X F X 3 numpy array
            textures: B X F X T X T X T X 3 numpy array
        Returns:
            images: B X 3 x 256 X 256 numpy array
        '''
        image, _, _ = self.renderer.forward(self.vertices, self.faces, torch.tanh(self.render_textures))
        return image

    def set_render_perspective(
        self,
        camera_transform: Transform,  #
        vehicle_transform: Transform,
        fov: int
    ):
        """
        设置渲染的视角
        ===
        通过 carla world 中的 camera_transform 和 vehicle_transform 可以反推出拍摄视角，并准确放置车辆。
        carla world 中采集数据集是有了车的位置，再去设置相机位置；而渲染的时候刚好相反

        >>> camera_transform   [[ -6.53281482  -6.53281482   3.82683432]    # 位置参数
        >>>                     [-22.5         45.           0.        ]]   # 旋转参数

        >>> vehicle_transform  [[109.92987823 -14.68801212   0.59999877]    # 位置参数
        >>>                     [  0.         -89.60925293   0.        ]]   # 旋转参数        
        """

        view_scale = 0.54 * 1.3                                   # 超参数： 数值越小 模型越大
        scale = (self.renderer.viewing_angle / fov) * view_scale # 视角缩放系数，FOV相关

        # 距离
        camera_t_list = [
            camera_transform.location.x, #
            camera_transform.location.y,
            camera_transform.location.z
        ]
        eye = [l * scale for l in camera_t_list]

        # calc camera_direction and camera_up
        # 角度转化为弧度
        pitch = np.radians(camera_transform.rotation.pitch)
        yaw = np.radians(camera_transform.rotation.yaw)
        roll = np.radians(camera_transform.rotation.roll)

        # 需不需要确定下范围？？？
        cam_direct = [
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch) * 1.25,
        ]
        cam_up = [
            np.cos(np.pi / 2 + pitch) * np.cos(yaw),
            np.cos(np.pi / 2 + pitch) * np.sin(yaw),
            np.sin(np.pi / 2 + pitch),
        ]

        self.renderer.eye = eye                     #[3, 3, 0]               #
        self.renderer.camera_direction = cam_direct #[0, 0, 0] #
        self.renderer.camera_up = cam_up

        # print(' scl', scale)
        # print(' loc', camera_t_list)
        # print(' eye', self.renderer.eye)
        # print(' dir', self.renderer.camera_direction)
        # print(' up ', self.renderer.camera_up)

        # return

        # 如果物体也有旋转，则需要调整相机位置和角度，和物体旋转方式一致
        # 先实现最简单的绕Z轴旋转
        p_cam = eye
        p_dir = [eye[0] + cam_direct[0], eye[1] + cam_direct[1], eye[2] + cam_direct[2]]
        p_up = [eye[0] + cam_up[0], eye[1] + cam_up[1], eye[2] + cam_up[2]]
        p_l = [p_cam, p_dir, p_up]
        trans_p = []

        for p in p_l:
            if np.sqrt(p[0]**2 + p[1]**2) == 0:
                cosfi = 0
                sinfi = 0
            else:
                cosfi = p[0] / np.sqrt(p[0]**2 + p[1]**2)
                sinfi = p[1] / np.sqrt(p[0]**2 + p[1]**2)

            cossum = cosfi * np.cos(np.radians(vehicle_transform.rotation.yaw)
                                   ) + sinfi * np.sin(np.radians(vehicle_transform.rotation.yaw))
            sinsum = np.cos(np.radians(vehicle_transform.rotation.yaw)
                           ) * sinfi - np.sin(np.radians(vehicle_transform.rotation.yaw)) * cosfi
            trans_p.append([np.sqrt(p[0]**2 + p[1]**2) * cossum, np.sqrt(p[0]**2 + p[1]**2) * sinsum, p[2]])

        eye = [
            trans_p[0][0],                 #
            trans_p[0][1],
            trans_p[0][2],
        ]
        cam_direct = [
            trans_p[1][0] - trans_p[0][0], #
            trans_p[1][1] - trans_p[0][1],
            trans_p[1][2] - trans_p[0][2]
        ]
        cam_up = [
            trans_p[2][0] - trans_p[0][0], #
            trans_p[2][1] - trans_p[0][1],
            trans_p[2][2] - trans_p[0][2]
        ]
                                           # 修改渲染视角参数
        self.renderer.eye = eye
        self.renderer.camera_direction = cam_direct
        self.renderer.camera_up = cam_up

        # print(' eye', self.renderer.eye)
        # print(' dir', self.renderer.camera_direction)
        # print(' up ', self.renderer.camera_up)
