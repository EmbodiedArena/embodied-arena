#!/bin/bash
# UniEQA 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../.." || exit

# API 密钥配置 (LLM 评估必需)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='your-api-base' # 可选

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 运行评测
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model qwen3_vl \
    --model_args model_name_or_path=Qwen/Qwen3-VL-4B-Instruct,max_num_frames=1 \
    --evaluator eqa \
    --tasks unieqa \
    --batch_size 1 \
    --output_path ./logs/unieqa/results
