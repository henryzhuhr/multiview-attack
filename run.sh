eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt



export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1  train-AEstyle.py
# python -m torch.distributed.launch --nproc_per_node=2 train-stylegan2.py \
#     --ckpt tmp/550000.pt

export CUDA_VISIBLE_DEVICES=0,1
# -m torch.distributed.launch --nproc_per_node=1 
python train-stylegan2-condition.py  \
    --save_dir stylegan2-cropcoco_car \
    --autoencoder_pretrained tmp/autoencoder/autoencoder-3000.pt \
    --batch 32 --iter 800000

