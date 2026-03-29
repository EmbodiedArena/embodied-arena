#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/../../../.." || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api-gpt-ge.apifox.cn/'  # 可选

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 运行评测
CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model internvl3_5 \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/InternVL3_5-8B,max_num_frames=8,fps=8 \
    --evaluator eqa \
    --tasks openeqa-emeqa \
    --batch_size 2 \
    --output_path ./logs/openeqa/internvl3_5