#!/bin/bash
# ！！使用时切换conda环境，名称为cambrian
# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit


# 设置模型参数
MODEL_NAME="pelican_vl"
MODEL_PATH="X-Humanoid/Pelican1.0-VL-7B"

export TOKENIZERS_PARALLELISM=false

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes=1  --main_process_port=$PORT -m embodied_eval \
    --model ${MODEL_NAME} \
    --model_args model_name_or_path=${MODEL_PATH},max_num_frames=32,attn_implementation=flash_attention_2, \
    --evaluator eqa \
    --tasks emspatial-bench \
    --batch_size 4 \
    --output_path /your/path/to/embodied-eval-main/logs/emspatial-bench/pelican_vl
    