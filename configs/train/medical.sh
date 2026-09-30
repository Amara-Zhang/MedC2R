
# Device settings
cuda_devices="0"  

# Torchrun settings
master_port=12345

# Paths
lang_encoder_path="./pretrained_models/Llama-2-7b-chat-hf"
tokenizer_path="./pretrained_models/Llama-2-7b-chat-hf"
pretrained_visual_encoder="./pretrained_models/RadFM_vit3d.pth"
pretrained_adapter="./pretrained_models/RadFM_perceiver_fc.pth"


data_folder="./data/global"
mask_folder="./data/region"
report_file="./data/reports.csv"
result_path="./results/reports.csv"

output_dir="./output"
deepspeed_config="./ds_configs/stage2.json"

# Training settings
learning_rate=5e-5
per_device_train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=8
evaluation_strategy="no"
save_strategy="epoch"
save_total_limit=3
weight_decay=0.0
warmup_steps=20
lr_scheduler_type="constant_with_warmup"
dataloader_num_workers=8
logging_steps=1
