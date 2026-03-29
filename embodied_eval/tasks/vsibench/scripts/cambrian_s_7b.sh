#!/bin/bash
# EmbodiedBrain-7B VSI-Bench 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 确保末尾带 v1

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=3,5,6 accelerate launch --num_processes=3 --main_process_port=$PORT -m embodied_eval \
    --model cambrian \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B,max_num_frames=32,use_flash_attention_2=True \
    --evaluator eqa \
    --tasks vsibench \
    --batch_size 2 \
    --output_path ./logs/vsibench/cambrian_s_7b

!/bin/bash

# # 2. 运行评测
# python -m embodied_eval \
#     --model embodied_brain \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B,max_num_frames=16,use_flash_attention_2=True \
#     --evaluator eqa \
#     --tasks vsibench \
#     --batch_size 2 \
#     --output_path ./logs/vsibench/embodied_brain



