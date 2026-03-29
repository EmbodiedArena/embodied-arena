#!/usr/bin/env bash
# VABench-P InternVL3.5-8B

cd "$(dirname "$0")/../../../../" || exit 1

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python -m embodied_eval \
  --model internvl3_5 \
  --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/InternVL3_5-8B,max_num_frames=1,use_flash_attention_2=True \
  --evaluator eqa \
  --tasks vabench \
  --batch_size 1 \
  --output_path ./logs/vabench/internvl3_5_8b
