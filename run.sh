eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt



export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

# python train-AE.py
# python -m torch.distributed.launch --nproc_per_node=2 train-stylegan2.py \
#     --ckpt tmp/550000.pt

export CUDA_VISIBLE_DEVICES=0

python train-stylegan2-condition.py  \
    --save_dir tsgan \
    --autoencoder_pretrained tmp/nAE/autoencoder.pt \
    --batch 8 --epochs 2000 --lr 0.002 \
    --d_loss_every 2 --d_reg_every 4 \
    --g_reg_every 4 --g_det_every 1

