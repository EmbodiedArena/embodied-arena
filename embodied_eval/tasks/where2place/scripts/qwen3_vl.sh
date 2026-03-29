#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 可选

# 2. 运行评测
# 自动分配可用端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")


# 使用 accelerate launch 开启多卡并行
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 --main_process_port=$PORT -m embodied_eval \
    --model qwen3_vl \
    --model_args model_name_or_path=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=32,max_pixels=50176,use_flash_attention_2=True \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 4 \
    --output_path ./logs/where2place/qwen3_vl
