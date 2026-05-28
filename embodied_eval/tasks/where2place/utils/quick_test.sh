#!/bin/bash
# 快速测试脚本 - 仅评估少量样本以验证模型是否正常工作
# 使用方法：bash quick_test.sh

cd "$(dirname "$0")/../../../.." || exit

# 配置
MODEL_PATH="X-Humanoid/Pelican1.0-VL-7B"  # 修改为你的模型路径
GPU_ID=0
LIMIT=5  # 仅测试 5 个样本

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

echo "========================================"
echo "PelicanVL 快速测试"
echo "========================================"
echo "模型: $MODEL_PATH"
echo "样本数: $LIMIT"
echo "========================================"

CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch \
    --num_processes=1 \
    --main_process_port=$PORT \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=$MODEL_PATH,max_num_frames=32,fps=2 \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --limit $LIMIT \
    --output_path ./logs/where2place/quick_test

echo ""
echo "========================================"
echo "快速测试完成！"
echo "结果保存在: ./logs/where2place/quick_test"
echo "========================================"
