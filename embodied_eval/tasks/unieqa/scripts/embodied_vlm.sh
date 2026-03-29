#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 2. 解决 CUDA 内存碎片化问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. 禁用 tokenizer 并行
export TOKENIZERS_PARALLELISM=false

# 4. 运行评测（单卡先跑通）
CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model embodied_vlm \
    --model_args model_name_or_path=IffYuan/Embodied-VLM-8B-RFT-0307,max_num_frames=8 \
    --evaluator eqa \
    --tasks unieqa \
    --batch_size 1 \
    --output_path ./logs/unieqa/embodied_vlm
