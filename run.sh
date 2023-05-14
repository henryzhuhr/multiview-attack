eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt




export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES=0

# rm -rf tmp/autoencoder
# python train-AE.py \
#     --obj_model assets/vehicle-YZ.obj \
#     --selected_faces assets/faces-std.txt \
#     --latent_dim 1024 

python train-tgan.py  \
    --save_dir attack-dog \
    --obj_model assets/vehicle-YZ.obj \
    --selected_faces assets/faces-std.txt \
    --pretrained tmp/autoencoder/autoencoder.pt \
    --categories "dog" \
    --batch 8 --epochs 200 --lr 0.1 --milestones 50 100

# python train-tgan.py  \
#     --save_dir attack-bowl \
#     --obj_model assets/vehicle-YZ.obj \
#     --selected_faces assets/faces-std.txt \
#     --pretrained tmp/autoencoder/autoencoder.pt \
#     --categories "bowl" \
#     --batch 8 --epochs 200 --lr 0.1 --milestones 50 100

python train-tgan.py  \
    --save_dir attack-dog_bowl \
    --obj_model assets/vehicle-YZ.obj \
    --selected_faces assets/faces-std.txt \
    --pretrained tmp/autoencoder/autoencoder.pt \
    --categories "dog" "bowl" \
    --batch 8 --epochs 500 --lr 0.1 --milestones 100 300
