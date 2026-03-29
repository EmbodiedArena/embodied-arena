#!/usr/bin/env bash
# VABench-P MiMo-Embodied-7B 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit 1

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m embodied_eval \
  --model mimo_embodied \
  --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=1,use_flash_attention_2=false \
  --evaluator eqa \
  --tasks vabench \
  --batch_size 1 \
  --output_path ./logs/vabench/mimo_embodied
