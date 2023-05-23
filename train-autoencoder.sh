eval "$(conda shell.bash hook)"
conda activate tsgan
echo -e "\n\033[01;36m[ENV]$(which python) \033[0m\n"

# python -m pip install requirements.txt




python train-autoencoder.py \
    --obj_model assets/vehicle.obj \
    --latent_dim 1024 \
    --selected_faces assets/faces-less.txt \
    --save_name ae-less

python train-autoencoder.py \
    --obj_model assets/vehicle.obj \
    --latent_dim 1024 \
    --selected_faces assets/faces-std.txt \
    --save_name ae-std

python train-autoencoder.py \
    --obj_model assets/vehicle.obj \
    --latent_dim 1024 \
    --selected_faces assets/faces-full.txt \
    --save_name ae-full



# export name=Town10HD-point_0014-distance_000-direction_0
# cp tmp/data/scenes/$name.png images/carla-scene.png
# cp tmp/data/labels/$name.json images/carla-scene.json
python sample.py --selected_faces assets/faces-less.txt --epoches 1