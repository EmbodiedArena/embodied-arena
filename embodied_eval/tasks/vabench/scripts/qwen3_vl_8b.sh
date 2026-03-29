#!/usr/bin/env bash
# VABench-P Qwen3-VL-8B-Instruct

cd "$(dirname "$0")/../../../../" || exit 1

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m embodied_eval \
  --model qwen3_vl \
  --model_args model_name_or_path=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=1,use_flash_attention_2=True \
  --evaluator eqa \
  --tasks vabench \
  --batch_size 1 \
  --output_path ./logs/vabench/qwen3_vl_8b
