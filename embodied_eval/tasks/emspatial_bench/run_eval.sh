#!/bin/bash

# EmbSpatial-Bench 评测脚本示例
# 使用方法: bash embodied_eval/tasks/emspatial_bench/run_eval.sh

# 切换到项目根目录
cd "$(dirname "$0")/../../.." || exit

export CUDA_VISIBLE_DEVICES=7

# 设置模型参数
MODEL_NAME="cambrian"  # 或其他支持的模型
MODEL_PATH="/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B" # 模型路径（HuggingFace ID 或本地路径）

# 运行评测
python -m embodied_eval \
    --model ${MODEL_NAME} \
    --model_args model_name_or_path=${MODEL_PATH},max_num_frames=32,use_flash_attention_2=False \
    --evaluator eqa \
    --tasks emspatial-bench \
    --batch_size 1 \
    --output_path ./logs/emspatial-bench/${MODEL_NAME}


