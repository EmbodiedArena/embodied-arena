#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

# 设置模型参数
MODEL_NAME="wall_oss"
MODEL_PATH="/your/path/to/embodied-eval-main/embodied_eval/data/wall-oss-fast"

export TOKENIZERS_PARALLELISM=false

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=0,1,2,4 python -m embodied_eval \
    --model ${MODEL_NAME} \
    --model_args model_name_or_path=${MODEL_PATH},max_num_frames=32, \
    --evaluator eqa \
    --tasks emspatial-bench \
    --batch_size 1 \
    --limit 1 \
    --output_path /your/path/to/embodied-eval-main/logs/emspatial-bench/wall-oss-fast
    