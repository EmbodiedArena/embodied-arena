#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 确保末尾带 v1

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model embodied_brain \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B,max_num_frames=8,max_new_tokens=256,max_pixels=50176,use_flash_attention_2=True \
    --evaluator eqa \
    --tasks openeqa-emeqa \
    --batch_size 1 \
    --output_path ./logs/openeqa/embodied_brain