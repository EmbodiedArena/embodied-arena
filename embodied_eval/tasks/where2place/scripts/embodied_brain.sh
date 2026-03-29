#!/bin/bash
# EmbodiedBrain-7B Where2Place 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 确保末尾带 v1

# 2. 配置坐标缩放（Embodied_Brain 输出 0-100 范围的百分比格式）
export USE_PERCENTAGE_COORDS=true

# 2. 运行评测
CUDA_VISIBLE_DEVICES=2,4,5,6 python -m embodied_eval \
    --model embodied_brain \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B,max_num_frames=32,use_flash_attention_2=False \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --output_path ./logs/where2place/embodied_brain
