#!/usr/bin/env bash
# VABench-P Cosmos-Reason1-7B

cd "$(dirname "$0")/../../../../" || exit 1

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} python -m embodied_eval \
  --model cosmos_reason1 \
  --model_args model_name_or_path=nvidia/Cosmos-Reason1-7B,max_num_frames=1,use_flash_attention_2=False \
  --evaluator eqa \
  --tasks vabench \
  --batch_size 1 \
  --output_path ./logs/vabench/cosmos_reason1_7b
