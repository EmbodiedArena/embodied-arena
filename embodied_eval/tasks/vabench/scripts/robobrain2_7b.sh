#!/usr/bin/env bash
# VABench-P RoboBrain2.0-7B

cd "$(dirname "$0")/../../../../" || exit 1

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

# Avoid importing TensorFlow/JAX backends via transformers (can trigger CUDA/XLA conflicts)
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export USE_TORCH=1

# Make native crashes more diagnosable
export PYTHONFAULTHANDLER=1

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m embodied_eval \
  --model robobrain2 \
  --model_args model_name_or_path=BAAI/RoboBrain2.0-7B,max_num_frames=1,use_flash_attention_2=False \
  --evaluator eqa \
  --tasks vabench_robobrain2 \
  --batch_size 1 \
  --output_path ./logs/vabench/robobrain2_7b
