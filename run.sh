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

python train-tgan.py  \
    --save_dir lsgan \
    --obj_model data/models/vehicle-YZ.obj \
    --selected_faces data/models/faces-std.txt \
    --batch 8 --epochs 2000 --lr 0.02
