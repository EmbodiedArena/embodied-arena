#!/bin/bash
# RoboVQA 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../.." || exit

# API密钥配置（用于LLM评估）
# export OPENAI_API_KEY='your-api-key'
# export OPENAI_API_BASE='your-api-base'  # 可选

# export OPENAI_API_KEY='your-api-key'

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 运行评测
# CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
#     --model qwen3_vl \
#    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Qwen3-VL-8B-Instruct/,max_num_frames=32,fps=2,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks robovqa \
#     --batch_size 1 \
#     --output_path ./logs/robovqa/qwen3_vl_8b


# CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
#     --model internvl3_5 \
#    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/InternVL3_5-8B,max_num_frames=32,fps=2,use_flash_attention_2=True \
#     --evaluator eqa \
#     --tasks robovqa \
#     --batch_size 1 \
#     --output_path ./logs/robovqa/internvl3.5-8B

# CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
#     --model pelican_vl \
#    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/pelican-vl/models/Pelican1.0-VL-7B,max_num_frames=32,fps=2,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks robovqa \
#     --batch_size 1 \
#     --output_path ./logs/robovqa/pelican_vl

CUDA_VISIBLE_DEVICES=6 python -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Qwen3-VL-4B-Instruct,max_num_frames=32,use_flash_attention_2=False \
    --evaluator eqa \
    --tasks robovqa \
    --batch_size 1 \
    --output_path ./logs/robovqa/cambrain









