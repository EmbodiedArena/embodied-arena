#!/bin/bash
# Change to project root directory
cd "$(dirname "$0")/../../.." || exit

# Set API KEY and BASE URL
export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct,max_num_frames=8 \
    --evaluator eqa \
    --tasks erqa \
    --batch_size 1 \
    --output_path /path/to/your/output/path
    