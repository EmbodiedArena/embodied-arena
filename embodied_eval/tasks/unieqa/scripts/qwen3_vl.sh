#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 可选

# 2. 运行评测
# 自动分配可用端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 计算显卡数量
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=1
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

# 使用 accelerate launch 开启多卡并行
accelerate launch --num_processes=$NUM_GPUS --main_process_port=$PORT -m embodied_eval \
    --model qwen3_vl \
    --model_args model_name_or_path=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=10,use_flash_attention_2=True \
    --evaluator eqa \
    --tasks unieqa \
    --batch_size 1 \
    --inference_only \
    --output_path ./logs/unieqa/qwen3_vl
