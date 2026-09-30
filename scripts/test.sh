#!/bin/bash


if [ $# -eq 0 ]
then
    echo "Warning: The configuration file should be specified according to the device being used!"
    exit 1
fi

config_file="$1"

script_name=$(basename "$0" .sh)

source "../configs/${script_name}/${config_file}.sh"


CUDA_VISIBLE_DEVICES=$cuda_devices python ../src/${script_name}.py \
    --lang_encoder_path "$lang_encoder_path" \
    --tokenizer_path "$tokenizer_path" \
    --pretrained_visual_encoder "$pretrained_visual_encoder" \
    --pretrained_adapter "$pretrained_adapter" \
    --ckpt_path "$ckpt_path" \
    --data_folder "$data_folder" \
    --mask_folder "$mask_folder" \
    --report_file "$report_file" \
    --monai_cache_dir "$monai_cache_dir" \
    --result_path "$result_path"
