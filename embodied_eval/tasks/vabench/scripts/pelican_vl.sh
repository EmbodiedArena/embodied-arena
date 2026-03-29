#!/usr/bin/env bash
# VABench-P Pelican-VL 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit 1

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m embodied_eval \
  --model pelican_vl \
  --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B,max_num_frames=1,use_flash_attention_2=False \
  --evaluator eqa \
  --tasks vabench \
  --batch_size 1 \
  --output_path ./logs/vabench/pelican_vl
