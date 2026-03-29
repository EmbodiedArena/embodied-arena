#!/bin/bash
# gemini3-pro VSI-Bench 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 确保末尾带 v1

export NCCL_TIMEOUT=1800

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,4,7 python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gpt-5.2,max_frames_num=32,max_retries=3,max_new_tokens=2048 \
    --evaluator eqa \
    --tasks vsibench \
    --batch_size 1 \
    --save_results \
    --output_path /your/path/to/embodied-eval-main/logs/vsibench/gpt5_2

!/bin/bash

# # 2. 运行评测
# python -m embodied_eval \
#     --model embodied_brain \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B,max_num_frames=16,use_flash_attention_2=True \
#     --evaluator eqa \
#     --tasks vsibench \
#     --batch_size 2 \
#     --output_path ./logs/vsibench/embodied_brain



