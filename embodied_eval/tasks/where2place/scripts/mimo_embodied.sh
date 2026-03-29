#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/'

# 启用像素坐标智能处理（适配MiMo-Embodied模型）
export NORMALIZE_PIXEL_COORDS=true

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
#     --model mimo_embodied \
#     --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=8,use_flash_attention_2=false \
#     --evaluator eqa \
#     --tasks where2place-bbox \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/where2place/mimo_embodied_bbox

CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model mimo_embodied \
    --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=8,use_flash_attention_2=false \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --output_path /your/path/to/embodied-eval-main/logs/where2place/mimo_embodied_point