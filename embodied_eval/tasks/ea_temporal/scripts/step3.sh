#!/bin/bash
# Step3-VL-10B-Base OpenEQA 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

# 2. 解决 CUDA 内存碎片化问题，避免 Segmentation fault
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. 禁用 tokenizer 并行，避免潜在的冲突
export TOKENIZERS_PARALLELISM=false

export NCCL_TIMEOUT=1800

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
source /home/tanghyyy/miniconda3/etc/profile.d/conda.sh

# 4. 运行评测
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
    --model step3_vl \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Step3-VL-10B,max_num_frames=32 \
    --evaluator eqa \
    --tasks ea-temporal \
    --batch_size 1 \
    --output_path /your/path/to/embodied-eval-main/logs/ea-temporal/step3
