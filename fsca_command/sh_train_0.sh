


tmp_my_name=${0##*/}
my_name=${tmp_my_name%.*}
echo $my_name

output_dir=$my_name
export CUDA_VISIBLE_DEVICES=0


python diffusion_mia.py \
    --num_generated_images 64 \
    --batch_size 50 \
    --epochs 100 \
    --output_dir ./checkpoints/output_dir \
    --datadir diffusion_inference_feature \
    --lr 0.001 \




