eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt




export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES=0

# python train-tgan.py  \
#     --save_dir train \
#     --obj_model assets/vehicle-YZ.obj \
#     --selected_faces assets/faces-std.txt \
#     --pretrained tmp/autoencoder/ae-std.pt \
#     --categories "dog" \
#     --batch 8 --epochs 200 --lr 0.1 --milestones 150
    
python train-tgan.py  \
    --save_dir train \
    --obj_model assets/vehicle-YZ.obj \
    --selected_faces assets/faces-std.txt \
    --pretrained tmp/autoencoder/ae-std.pt \
    --categories "person" \
    --batch 8 --epochs 100 --lr 0.1 --milestones 100


python train-tgan.py  \
    --save_dir train \
    --obj_model assets/vehicle-YZ.obj \
    --selected_faces assets/faces-std.txt \
    --pretrained tmp/autoencoder/ae-std.pt \
    --categories "fork" \
    --batch 8 --epochs 150 --lr 0.1 --milestones 150


# python train-tgan.py  \
#     --save_dir train \
#     --obj_model assets/vehicle-YZ.obj \
#     --selected_faces assets/faces-std.txt \
#     --pretrained tmp/autoencoder/ae-std.pt \
#     --categories "dog" \
#     --batch 8 --epochs 200 --lr 0.1 --milestones 150

# python train-tgan.py  \
#     --save_dir attack-bowl \
#     --obj_model assets/vehicle-YZ.obj \
#     --selected_faces assets/faces-std.txt \
#     --pretrained tmp/autoencoder/ae-std.pt \
#     --categories "dog" "apple" \
#     --batch 8 --epochs 300 --lr 0.1 --milestones 100 150

# python train-tgan.py  \
#     --save_dir attack_l1 \
#     --obj_model assets/vehicle-YZ.obj \
#     --selected_faces assets/faces-less.txt \
#     --pretrained tmp/autoencoder/ae-less.pt \
#     --categories "dog" "bowl" "apple" "airplane" \
#     --batch 8 --epochs 400 --lr 0.1 --milestones 200 300

