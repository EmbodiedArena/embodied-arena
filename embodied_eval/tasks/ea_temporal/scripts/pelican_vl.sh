#!/bin/bash
# vsi-bench 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# API密钥配置（如需要）
# export OPENAI_API_KEY='your-api-key'
# export OPENAI_API_BASE='your-api-base'  # 可选

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/'  # 可选

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

#运行时间1min

# 运行评测
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B,max_num_frames=1,use_flash_attention_2=False \
    --evaluator eqa \
    --tasks ea-temporal \
    --batch_size 1 \
    --output_path /your/path/to/embodied-eval-main/logs/ea-temporal/pelican_vl