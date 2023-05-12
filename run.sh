eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt




export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES=0

# rm -rf tmp/autoencoder
# python train-AE.py \
#     --obj_model data/models/vehicle-YZ.obj \
#     --selected_faces data/models/faces-std.txt \
#     --latent_dim 1024 

python train-tgan.py  \
    --save_dir generator \
    --obj_model data/models/vehicle-YZ.obj \
    --selected_faces data/models/faces-std.txt \
    --pretrained tmp/autoencoder/autoencoder.pt \
    --batch 8 --epochs 2000 --lr 0.1
