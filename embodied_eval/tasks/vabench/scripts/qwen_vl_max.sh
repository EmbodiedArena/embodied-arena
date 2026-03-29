#!/usr/bin/env bash
# VABench-P Qwen-VL-Max（OpenAI 兼容 API）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "$SCRIPT_DIR/../../../../" && pwd)" || exit 1

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m embodied_eval \
  --model openai_async_compatible \
  --model_args model_name_or_path=qwen-vl-max,max_frames_num=1,max_retries=3,max_new_tokens=2048 \
  --evaluator eqa \
  --tasks vabench \
  --batch_size 1 \
  --output_path ./logs/vabench/qwen_vl_max
