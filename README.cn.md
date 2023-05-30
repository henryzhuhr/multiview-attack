

## Requirement
[openai/CLIP](https://github.com/openai/CLIP)
```sh
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

```sh
cd neural_renderer
python3 setup.py install
```

## Train

训练纹理编解码器
```bash
python3 train-AE.py
```

# 测试流程

1. `run.sh` 训练完成后，训练结果保存在 `tmp/train`。在 `tmp/train/train-<NAMME>/checkpoint/_generator.pt` 有权重
2. `test-render-img` 将各个方法转化成图片，保存在 `tmp/test/{world_map}-{nowt}` 下 (`nowt`为当前时间)
3. `metrics-yolo.py` 推理生成的 `tmp/test/{world_map}-{nowt}`。结果保存在 `tmp/test/{world_map}-{nowt}-det`




````python
camera_distances = [ # train
     [5, 4, 2, 90], # [x, y, z, fov]
     [5, 5, 2.5, 90],
     [6, 4, 3, 90],
     [7, 4, 2, 90],
]
````
在python 中根据 `camera_distances` 绘制出在 3D 坐标的位置

如下要求 
1. 首先，对 `camera_distances` 的坐标进行扩充为 8 个点 例如 `[distance z, fov]` 中只考虑扩充 `x,y`。4个点在坐标轴上，`[x,0]`, `[-x,0]`, `[0,y]`, `[0,-y]`, 另外四个点在圆上，`[cos45,sin45]\sqrt(x*x+y*y)`,其余几个在相应位置 
2. `x` `y` 反转成正负值，`z` 保持不变。例如 `[5, 4, 2, 90]` 扩充为 `[5, 4, 2, 90] [-5, 4, 2, 90] [-5, -4, 2, 90] [5, -4, 2, 90]`，该组点所在的平面，需要用一个虚线连接起来表示一个平面，并且连接围成的形状是一个圆形，而不是直接把各个点连接起来
3. 每一个点的坐标 `[x,y,z]` 标注出来，每一个点与原点连线，线的粗细可以自己调节
4. 点的颜色和连线的颜色保持一致，并且颜色可以自己修改
5. 取消网格线显示，和坐标背景，只显示绘制的部分
6. 我还希望自己设置观察角度，请给出该部分的代码
7. 保存为 png 和 pdf 两种格式


对坐标进行扩充为 8 个点 例如 `[d z, fov]` 中只考虑扩充 `d`。
- 4个点在坐标轴上，`[d,0]`, `[-d,0]`, `[0,d]`, `[0,-d]`, 
- 另外四个点在圆上，`[cos45,sin45]*d`,其余几个在相应位置 