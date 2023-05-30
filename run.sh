eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt




export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=8

export CUDA_VISIBLE_DEVICES=0

for cls in "dog" "kite" "skateboard"
do
    python train-pacg.py \
        --save_dir train_std \
        --obj_model assets/audi.obj \
        --selected_faces assets/faces-audi-std.txt \
        --pretrained pretrained/ae-audi-std-6000.pt \
        --categories $cls \
        --batch 8 --epochs 200 --lr 0.1 --milestones 150
done

python train-pacg.py \
    --save_dir train_std \
    --obj_model assets/audi.obj \
    --selected_faces assets/faces-audi-std.txt \
    --pretrained pretrained/ae-audi-std-6000.pt \
    --categories "dog" "kite" \
    --batch 8 --epochs 300 --lr 0.1 --milestones 150

python train-pacg.py  \
    --save_dir train_std \
    --obj_model assets/audi.obj \
    --selected_faces assets/faces-audi-std.txt \
    --pretrained pretrained/ae-audi-std-6000.pt \
    --categories "apple" "skateboard" \
    --batch 8 --epochs 300 --lr 0.1 --milestones 150


python train-pacg.py \
    --save_dir train_std \
    --obj_model assets/audi.obj \
    --selected_faces assets/faces-audi-std.txt \
    --pretrained pretrained/ae-audi-std-6000.pt \
    --categories "dog" "kite" "skateboard" \
    --batch 8 --epochs 400 --lr 0.1 --milestones 250 



# python train-pacg.py  \
#     --save_dir train \
#     --obj_model assets/audi.obj \
#     --selected_faces assets/faces-audi-std.txt \
#     --pretrained pretrained/ae-audi-std-6000.pt \
#     --categories "dog" "bowl" "apple" "airplane" \
#     --batch 8 --epochs 300 --lr 0.1 --milestones 200 300

