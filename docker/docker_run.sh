#!/bin/bash

# prepare /datasets, /pretrained_models and /output folders as explained in the main README.md
xhost +
docker run \
--gpus all \
-it \
--shm-size=8gb --env="DISPLAY" \
--volume="/home/hoenig/BOP/gdrnpp_bop2022:/gdrn" \
--volume="/hdd/datasets_bop:/gdrn/datasets/BOP_DATASETS" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix" \
--name=gdrnppv0 gdrn_base