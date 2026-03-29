#!/bin/bash

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 可选

# 1.5. 优化显存分配（减少碎片）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 获取随机端口和 GPU 数量
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
# 强制单进程单卡（InternVL3.5 当前实现对多进程不友好）
NUM_GPUS=1

# 3. 运行评测
# 使用多卡并行，并开启 Flash Attention
accelerate launch --num_processes=$NUM_GPUS --main_process_port=$PORT -m embodied_eval \
    --model internvl3_5 \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/InternVL3_5-8B,max_num=1,max_num_frames=15,use_flash_attention_2=True \
    --evaluator eqa \
    --tasks unieqa \
    --batch_size 1 \
    --inference_only \
    --output_path ./logs/unieqa/internvl3_5-8B
