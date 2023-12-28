#!/bin/bash

#lmo_variants=("lmo" "lmo_random_texture_all")
lmo_variants=("lmo")

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pwd

perturbations=("1" "2" "3" "4" "5")

noise_types=("gaussian_noise" "shot_noise" "motion_blur" "brightness" "gaussian_blur")

for noise_type in "${noise_types[@]}"
do
    for perturbation in "${perturbations[@]}"
    do
        command="mv ./datasets/BOP_DATASETS/lmo/perturbations/${noise_type[@]}/lmo_test_${noise_type[@]}_${perturbation[@]}/000002 ./datasets/BOP_DATASETS/lmo/test/000002" 
        eval "$command"

        for lmo_variant in "${lmo_variants[@]}"
        do       
            command="./det/yolox/tools/test_yolox.sh configs/yolox/bop_pbr/yolox_${lmo_variant[@]}.py 0 output/yolox/bop_pbr/yolox_${lmo_variant[@]}/model_final.pth"
            eval "$command"
        done

        # Move test image folder back to its location
        command="mv ./datasets/BOP_DATASETS/lmo/test/000002 ./datasets/BOP_DATASETS/lmo/perturbations/${noise_type[@]}/lmo_test_${noise_type[@]}_${perturbation[@]}/000002"
        eval "$command"

        # copy result files to specific test dataset folder
        for i in "${!lmo_variants[@]}"
        do
            lmo_variant="${lmo_variants[i]}"

            command="mv ./output/yolox/bop_pbr/yolox_${lmo_variant[@]}/inference/lmo_bop_test/map_result.json ./datasets/BOP_DATASETS/lmo/perturbations/${noise_type[@]}/lmo_test_${noise_type[@]}_${perturbation[@]}/map_result_${lmo_variant[@]}.json"
            eval "$command"
        done
    done
done



