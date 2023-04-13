mkdir -p libs
cd libs
git clone https://github.com/CompVis/taming-transformers.git taming-transformers\
    -b master
git clone https://github.com/openai/CLIP.git clip \
    -b main
git clone https://github.com/ultralytics/yolov5.git \
    -b v6.1