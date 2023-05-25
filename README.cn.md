

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
2. 