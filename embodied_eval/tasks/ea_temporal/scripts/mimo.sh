#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

# NCCL超时设置（增加到30分钟，默认10分钟）
export NCCL_TIMEOUT=1800
# NCCL调试选项（可选，用于排查问题）
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=2,3,4,7 accelerate launch --num_processes=4 --main_process_port=$PORT -m embodied_eval \
    --model mimo_embodied \
    --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=8 \
    --evaluator eqa \
    --tasks ea-temporal \
    --batch_size 1 \
    --output_path /your/path/to/embodied-eval-main/logs/ea-temporal/mimo_embodied
