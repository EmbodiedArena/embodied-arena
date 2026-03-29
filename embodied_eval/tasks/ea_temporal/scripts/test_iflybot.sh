cd "$(dirname "$0")/../../../.." || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 2. 解决 CUDA 内存碎片化问题，避免 Segmentation fault
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. 禁用 tokenizer 并行，避免潜在的冲突
export TOKENIZERS_PARALLELISM=false

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=5 python -m embodied_eval \
  --model iflybot_vlm \
  --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/iFlyBotVLM,max_num_frames=32 \
  --evaluator eqa \
  --tasks ea-temporal \
  --batch_size 1 \
  --output_path /your/path/to/embodied-eval-main/logs/ea-temporal/iflybot_vlm \