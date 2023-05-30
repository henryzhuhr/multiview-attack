eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

pretrained="tmp/train/train-dog_person/checkpoint/_generator.pt"
world_map="Town10HD"
nowt=$(date "+%m%d_%H%M")

python test-render-img.py \
    --pretrained $pretrained \
    --world_map $world_map \
    --nowt $nowt

python test-metrics-yolo.py \
    --data_dir "tmp/test/$world_map-$nowt"
