#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

# 2. 解决 CUDA 内存碎片化问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. 禁用 tokenizer 并行
export TOKENIZERS_PARALLELISM=false

# 4. 运行评测（单卡先跑通）
CUDA_VISIBLE_DEVICES=2 python -m embodied_eval \
    --model embodied_vlm \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Embodied-VLM-8B-RFT-0307,max_num_frames=8 \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --output_path ./logs/where2place/embodied_vlm/point

CUDA_VISIBLE_DEVICES=2 python -m embodied_eval \
    --model embodied_vlm \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Embodied-VLM-8B-RFT-0307,max_num_frames=8 \
    --evaluator eqa \
    --tasks where2place-bbox \
    --batch_size 1 \
    --output_path ./logs/where2place/embodied_vlm/bbox