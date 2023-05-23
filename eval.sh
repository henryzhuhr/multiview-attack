eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

for map in "Town01" "Town02" "Town03" "Town10HD"
do
    python eval.py --pretrained "tmp/train-person-05221615/checkpoint/_generator.pt" --map $map
done
