
DATA_PATH="/home/caiyunfeng/python_code/ollma_code/data_my_align_0621_Geo3k_1.json"
IMAGE_PATH="/home/caiyunfeng/python_code/data_all/data_all"
MODEL_MAX_LENGTH=4096
OUTPUT_DIR="/media/caiyunfeng/Expansion/cuixiaoteng/checkpoints/llava_factory/custom-finetune-TinyLLaVA-OpenELM-450M-SigLIP-0.89B-full"

deepspeed --include localhost:0,1 --master_port 29501 tinyllava/train/custom_finetune.py \
    --deepspeed ./scripts/zero3.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version phi \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 True \
    --training_recipe common \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --lora_r 128 \
    --lora_alpha 256 \
    --group_by_modality_length False \
    --pretrained_model_path "jiajunlong/TinyLLaVA-OpenELM-450M-SigLIP-0.89B" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name custom-finetune-TinyLLaVA-OpenELM-450M-SigLIP-0.89B-full
