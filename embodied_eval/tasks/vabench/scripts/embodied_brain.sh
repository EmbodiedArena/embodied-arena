#!/usr/bin/env bash
# VABench-P EmbodiedBrain-7B 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit 1

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} python -m embodied_eval \
  --model embodied_brain \
  --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B,max_num_frames=1,use_flash_attention_2=False \
  --evaluator eqa \
  --tasks vabench \
  --batch_size 1 \
  --output_path ./logs/vabench/embodied_brain
