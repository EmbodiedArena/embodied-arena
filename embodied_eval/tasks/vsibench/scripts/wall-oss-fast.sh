#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/../../../.." || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m embodied_eval \
  --model wall_oss \
  --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/wall-oss-fast,max_num_frames=32,use_flash_attention_2=false \
  --evaluator eqa \
  --tasks vsibench \
  --batch_size 1 \
  --limit 1 \
  --output_path /your/path/to/embodied-eval-main/logs/vsibench/wall-oss-fast \
  2>&1 | tee /your/path/to/embodied-eval-main/embodied_eval/tasks/vsibench/debug_run.log