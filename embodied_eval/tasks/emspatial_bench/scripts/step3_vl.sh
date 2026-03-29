#!/bin/bash
# Step3-VL-10B-Base EmbSpatial-Bench 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 设置模型参数
MODEL_NAME="step3_vl"
MODEL_PATH="/your/path/to/embodied-eval-main/embodied_eval/data/Step3-VL-10B"

# 1. 配置 API 密钥 (某些任务在导入时需要，即使不运行该任务)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

# 2. 解决 CUDA 内存碎片化问题，避免 Segmentation fault
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. 禁用 tokenizer 并行，避免潜在的冲突
export TOKENIZERS_PARALLELISM=false

# 4. 运行评测
CUDA_VISIBLE_DEVICES=4 python -m embodied_eval \
    --model ${MODEL_NAME} \
    --model_args model_name_or_path=${MODEL_PATH} \
    --evaluator eqa \
    --tasks emspatial-bench \
    --batch_size 1 \
    --limit 10 \
    --output_path ./logs/emspatial-bench/${MODEL_NAME}
