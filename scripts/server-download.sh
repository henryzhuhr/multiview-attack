SOURCE_DIR="zhr@$N4090_SERVER:~/Project/diffusion-attack/.git"
TARGET_DIR="./"
mkdir -p $TARGET_DIR
echo "$SOURCE_DIR -> $TARGET_DIR"

rsync -av $SOURCE_DIR $TARGET_DIR \
    --link-dest $TARGET_DIR \
    --exclude="__pycache__/" \
    --exclude="*.egg-info" \
    --exclude="tmp" 