eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt




export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES=0

# for cls in "dog" "kite" "skateboard"
# do
#     python train.py \
#         --save_dir newtrain \
#         --obj_model assets/audi.obj \
#         --selected_faces assets/faces-audi-std.txt \
#         --pretrained pretrained/ae-audi-std-6000.pt \
#         --categories $cls \
#         --batch 8 --epochs 400 --lr 0.1 --milestones 200
# done

python train.py \
    --save_dir whuJ \
    --obj_model assets/audi.obj \
    --selected_faces assets/faces-audi-std.txt \
    --pretrained pretrained/ae-audi-std-6000.pt \
    --categories "person" \
    --batch 8 --epochs 500 --lr 0.1 --milestones 100 200 300

python train.py \
    --save_dir whuJ \
    --obj_model assets/audi.obj \
    --selected_faces assets/faces-audi-std.txt \
    --pretrained pretrained/ae-audi-std-6000.pt \
    --categories "bird" \
    --batch 8 --epochs 500 --lr 0.1 --milestones 100 200 300
    
python train.py \
    --save_dir whuJ \
    --obj_model assets/audi.obj \
    --selected_faces assets/faces-audi-std.txt \
    --pretrained pretrained/ae-audi-std-6000.pt \
    --categories "cup" \
    --batch 8 --epochs 500 --lr 0.1 --milestones 100 200 300

    
# python train-pacg.py \
#     --save_dir newtrain \
#     --obj_model assets/audi.obj \
#     --selected_faces assets/faces-audi-std.txt \
#     --pretrained tmp/AE/ae_b-audi-std-5000.pt \
#     --categories dog \
#     --batch 8 --epochs 300 --lr 0.1 --milestones 200

# python train-pacg.py \
#     --save_dir RealWorld \
#     --obj_model assets/audi.obj \
#     --selected_faces assets/faces-audi-std.txt \
#     --pretrained pretrained/ae-audi-std-6000.pt \
#     --categories dog --resume "tmp/train/train_std-dog-06130953/checkpoint/_generator.pt" \
#     --batch 4 --epochs 100 --lr 0.01 --milestones 200


# python train-pacg.py \
#     --save_dir train_std \
#     --obj_model assets/audi.obj \
#     --selected_faces assets/faces-audi-std.txt \
#     --pretrained pretrained/ae-audi-std-6000.pt \
#     --categories "dog" "kite" \
#     --batch 8 --epochs 300 --lr 0.1 --milestones 150

# python train-pacg.py  \
#     --save_dir train_std \
#     --obj_model assets/audi.obj \
#     --selected_faces assets/faces-audi-std.txt \
#     --pretrained pretrained/ae-audi-std-6000.pt \
#     --categories "apple" "skateboard" \
#     --batch 8 --epochs 300 --lr 0.1 --milestones 150


# python train-pacg.py \
#     --save_dir train_std \
#     --obj_model assets/audi.obj \
#     --selected_faces assets/faces-audi-std.txt \
#     --pretrained pretrained/ae-audi-std-6000.pt \
#     --categories "dog" "kite" "skateboard" \
#     --batch 8 --epochs 300 --lr 0.1 --milestones 250 



# python train-pacg.py  \
#     --save_dir train \
#     --obj_model assets/audi.obj \
#     --selected_faces assets/faces-audi-std.txt \
#     --pretrained pretrained/ae-audi-std-6000.pt \
#     --categories "dog" "bowl" "apple" "airplane" \
#     --batch 8 --epochs 300 --lr 0.1 --milestones 200 300



scp -r -P 27 zhr@$ANTIS_PUBLIC_IP:/home/zhr/Project/diffusion-attack/temps/render-bird/Town05-0716_1246-bird-yolo .
scp -r -P 27 zhr@$ANTIS_PUBLIC_IP:/home/zhr/Project/diffusion-attack/temps/render-cup/Town05-0716_1210-cup-yolo .
scp -r -P 27 zhr@$ANTIS_PUBLIC_IP:/home/zhr/Project/diffusion-attack/temps/render-person/Town05-0712_1145-person-yolo .