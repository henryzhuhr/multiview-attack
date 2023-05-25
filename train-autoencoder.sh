eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt

# "audi" 
for model_name in "vehicle"
do
    for type in "less" "std" "full"
    do
        python train-autoencoder.py \
            --obj_model assets/${model_name}.obj \
            --latent_dim 1024 \
            --selected_faces assets/faces-${model_name}-${type}.txt \
            --save_name ae-${model_name}-${type}
    done
done