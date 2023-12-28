#!/bin/bash

#tless_variants=("tless" "tless_random_texture")
tless_variants=("tless")

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pwd

perturbations=("2")
#perturbations=("1" "2" "3" "4" "5")

noise_types=("gaussian_noise")
#noise_types=("gaussian_noise" "shot_noise" "motion_blur" "brightness" "gaussian_blur")

for noise_type in "${noise_types[@]}"
do
    for perturbation in "${perturbations[@]}"
    do
        command="mv ./datasets/BOP_DATASETS/tless/perturbations/${noise_type[@]}/test_primesense_${noise_type[@]}_${perturbation[@]} ./datasets/BOP_DATASETS/tless/test_primesense" 
        eval "$command"

        for tless_variant in "${tless_variants[@]}"
        do       
            command="./det/yolox/tools/test_yolox.sh configs/yolox/bop_pbr/yolox_${tless_variant[@]}.py 0 output/yolox/bop_pbr/yolox_${tless_variant[@]}/model_final.pth"
            eval "$command"
        done

        # Move test image folder back to its location
        command="mv ./datasets/BOP_DATASETS/tless/test_primesense ./datasets/BOP_DATASETS/tless/perturbations/${noise_type[@]}/test_primesense_${noise_type[@]}_${perturbation[@]}"
        eval "$command"

        # copy result files to specific test dataset folder
        for i in "${!tless_variants[@]}"
        do
            tless_variant="${tless_variants[i]}"

            command="mv ./output/yolox/bop_pbr/yolox_${tless_variant[@]}/inference/tless_bop_test_primesense/map_result.json ./datasets/BOP_DATASETS/tless/perturbations_add_0.1/${noise_type[@]}/test_primesense_${noise_type[@]}_${perturbation[@]}/map_result_${tless_variant[@]}.json"
            eval "$command"
        done
    done
done



