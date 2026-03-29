#!/bin/bash
# EmbodiedBrain-7B UniEQA 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 确保末尾带 v1

# 2. 运行评测
# 注意：请根据实际情况修改以下参数：
# - model_name_or_path: EmbodiedBrain-7B 模型路径
# - max_num_frames: 视频采样帧数（推荐 8-32）
# - output_path: 结果输出路径

python -m embodied_eval \
    --model embodied_brain \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B,max_num_frames=15,use_flash_attention_2=True \
    --evaluator eqa \
    --tasks unieqa \
    --batch_size 1 \
    --output_path ./logs/unieqa/embodied_brain
