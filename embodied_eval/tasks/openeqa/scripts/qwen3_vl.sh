#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/../../../.." || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

export TOKENIZERS_PARALLELISM=false

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# --limit限制测试样本数量
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes=4  --main_process_port=$PORT -m embodied_eval \
    --model qwen3_vl \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Qwen3-VL-8B-Instruct/,max_num_frames=8,attn_implementation=flash_attention_2, \
    --evaluator eqa \
    --tasks openeqa-emeqa \
    --batch_size 4 \
    --output_path /your/path/to/embodied-eval-main/logs/openeqa/qwen3-vl-8b-instruct
    