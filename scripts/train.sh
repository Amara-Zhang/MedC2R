#!/bin/bash

# 
if [ $# -eq 0 ]
then
    echo "Warning: The configuration file should be specified according to the device being used!"
    exit 1
fi


config_file="$1"


script_name=$(basename "$0" .sh)


source "../configs/${script_name}/${config_file}.sh"


nproc_per_node=$(echo $cuda_devices | tr ',' '\n' | wc -l)

CUDA_VISIBLE_DEVICES=$cuda_devices torchrun --nproc_per_node=$nproc_per_node --master-port=$master_port  ../src/${script_name}.py \
    --bf16 $bf16 \
    --lang_encoder_path "$lang_encoder_path" \
    --tokenizer_path "$tokenizer_path" \
    --pretrained_visual_encoder "$pretrained_visual_encoder" \
    --pretrained_adapter "$pretrained_adapter" \
    --data_folder "$data_folder" \
    --mask_folder "$mask_folder" \
    --report_file "$report_file" \
    --monai_cache_dir "$monai_cache_dir" \
    --output_dir "$output_dir" \
    --deepspeed "$deepspeed_config" \
    --per_device_train_batch_size $per_device_train_batch_size \
    --num_train_epochs $num_train_epochs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "$evaluation_strategy" \
    --save_strategy "$save_strategy" \
    --save_total_limit $save_total_limit \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --warmup_steps $warmup_steps \
    --lr_scheduler_type "$lr_scheduler_type" \
    --dataloader_num_workers $dataloader_num_workers \
    --run_name "$experiment_name" \
    --logging_steps $logging_steps
