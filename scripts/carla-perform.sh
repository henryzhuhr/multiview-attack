eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"


for i in {1..200}
do
    python scripts/carla-preform.py
done