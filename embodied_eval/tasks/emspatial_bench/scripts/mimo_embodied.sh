#!/bin/bash
# ！！使用时切换conda环境，名称为cambrian
# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit
# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

# 设置模型参数
MODEL_NAME="mimo_embodied"
MODEL_PATH="/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/"

export TOKENIZERS_PARALLELISM=false

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1  --main_process_port=$PORT -m embodied_eval \
    --model ${MODEL_NAME} \
    --model_args model_name_or_path=${MODEL_PATH},max_num_frames=32,attn_implementation=flash_attention_2, \
    --evaluator eqa \
    --tasks emspatial-bench \
    --batch_size 4 \
    --output_path /your/path/to/embodied-eval-main/logs/emspatial-bench/MiMo-Embodied-7B
    