
###################### e.g.: coco training ######################################

export MODEL_NAME="/root/paddlejob/workspace/env_run/output/move2t7/stable-diffusion-v1-4"
export dataset_name="/root/paddlejob/workspace/env_run/output/move2t7/coco_dataset/coco_ori_train"

CUDA_VISIBLE_DEVICES=0 nohup /root/paddlejob/workspace/env_run/qjy_dataset_env/diffusion/bin/accelerate launch  --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$dataset_name \
  --use_ema \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=50000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="coco_m_real" \
  > log.txt & # --dataset_name=$dataset_name \