#!/bin/bash

cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gpt-5.2,max_frames_num=8,max_retries=3,max_new_tokens=2048 \
    --evaluator eqa \
    --tasks openeqa-emeqa \
    --batch_size 1 \
    --save_results \
    --output_path /your/path/to/embodied-eval-main/logs/openeqa/gpt-5.2