eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt




export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES=0

# python train-pacg.py  \
#     --save_dir train \
#     --obj_model assets/audi.obj \
#     --selected_faces assets/faces-audi-std.txt \
#     --pretrained pretrained/ae-audi-std-6000.pt \
#     --categories "dog" \
#     --batch 8 --epochs 100 --lr 0.1 --milestones 150
    
# python train-pacg.py  \
#     --save_dir train \
#     --obj_model assets/audi.obj \
#     --selected_faces assets/faces-audi-std.txt \
#     --pretrained pretrained/ae-audi-std-6000.pt \
#     --categories "dog" "person" \
#     --batch 8 --epochs 100 --lr 0.1 --milestones 100

python train-pacg.py  \
    --save_dir train \
    --obj_model assets/audi.obj \
    --selected_faces assets/faces-audi-std.txt \
    --pretrained pretrained/ae-audi-std-6000.pt \
    --categories "dog" "bowl" "apple" "airplane" \
    --batch 8 --epochs 300 --lr 0.1 --milestones 200 300

