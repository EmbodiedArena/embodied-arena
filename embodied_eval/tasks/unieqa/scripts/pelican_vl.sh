#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/' # 可选

# 2. 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 3. 运行评测
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes=2 --main_process_port=$PORT -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B,max_num_frames=1,use_flash_attention_2=False \
    --evaluator eqa \
    --tasks unieqa \
    --batch_size 2 \
    --output_path ./logs/unieqa/pelican_vl
