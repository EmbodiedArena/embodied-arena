#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 确保末尾带 v1

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=1,4,5 accelerate launch --num_processes=3 --main_process_port=$PORT -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gemini-3-pro,max_frames_num=8,max_retries=1,max_new_tokens=2048 \
    --evaluator eqa \
    --tasks openeqa-emeqa \
    --batch_size 3 \
    --save_results \
    --output_path  ./logs/openeqa-emeqa/gemini-3-pro/