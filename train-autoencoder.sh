eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt


python train-autoencoder.py \
    --obj_model assets/audi.obj \
    --latent_dim 1024 \
    --selected_faces assets/faces-audi-std.txt \
    --save_name ae_b-audi-std

# "audi" 
for model_name in "audi"
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