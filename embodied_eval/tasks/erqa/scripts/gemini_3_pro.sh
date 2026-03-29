#!/bin/bash
# Gemini 3 Pro ERQA 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (Gemini 通过 OpenAI 兼容接口调用)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://openai.arnotho.com/v1'

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 使用 accelerate launch 开启多进程数据并行
# 这里 --num_processes 设为 8，你可以根据需求调大
accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gemini-3-pro-preview,max_frames_num=10,max_retries=1,max_new_tokens=2048 \
    --evaluator eqa \
    --tasks erqa \
    --batch_size 1 \
    --inference_only \
    --output_path ./logs/erqa/gemini_3_pro
