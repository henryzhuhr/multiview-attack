





eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt

# cd libs/stylegan2-pytorch
# python generate.py \
#     --sample 8 \
#     --pics 2 \
#     --ckpt /home/zhr/Project/diffusion-attack/pretrained/550000.pt


export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  train-AEstyle.py
python -m torch.distributed.launch --nproc_per_node=2 train-stylegan2.py \
    --ckpt tmp/550000.pt

# n_gpus=2
# python  train-AEstyle.py \
#     --autoencoder_pretrained tmp/autoencoder/autoencoder-5500.pt
# python -m torch.distributed.launch --nproc_per_node=$n_gpus --nnodes=1  train-AEstyle.py