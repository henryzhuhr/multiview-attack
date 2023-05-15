eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt




rm -rf tmp/autoencoder
python train-autoencoder.py \
    --obj_model assets/vehicle-YZ.obj \
    --latent_dim 1024 \
    --selected_faces assets/faces-less.txt \
    --save_name ae-less

python train-autoencoder.py \
    --obj_model assets/vehicle-YZ.obj \
    --latent_dim 1024 \
    --selected_faces assets/faces-std.txt \
    --save_name ae-std

python train-autoencoder.py \
    --obj_model assets/vehicle-YZ.obj \
    --latent_dim 1024 \
    --selected_faces assets/faces-full.txt \
    --save_name ae-full
