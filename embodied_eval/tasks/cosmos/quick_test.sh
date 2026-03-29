#!/bin/bash
# Cosmos 快速测试脚本 - 用于测试单个子任务

# 切换到项目根目录
cd "$(dirname "$0")/../../.." || exit

# ============ 配置参数 ============
MODEL_PATH="X-Humanoid/Pelican1.0-VL-7B"
MODEL_TYPE="pelican_vl"
GPU_ID=0
BATCH_SIZE=1
MAX_NUM_FRAMES=32
FPS=2

# 默认测试 bridgev2（可以通过命令行参数修改）
TASK=${1:-bridgev2}

OUTPUT_BASE="./logs/cosmos/pelican_vl_7b"

# ============ 运行评估 ============
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

echo "========================================"
echo "Cosmos 快速测试"
echo "========================================"
echo "测试任务: $TASK"
echo "模型: $MODEL_PATH"
echo "模型类型: $MODEL_TYPE"
echo "GPU: $GPU_ID"
echo "输出: ${OUTPUT_BASE}/${TASK}"
echo "========================================"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch \
    --num_processes=1 \
    --main_process_port=$PORT \
    -m embodied_eval \
    --model $MODEL_TYPE \
    --model_args model_name_or_path=$MODEL_PATH,max_num_frames=$MAX_NUM_FRAMES,fps=$FPS \
    --evaluator eqa \
    --tasks cosmos-$TASK \
    --batch_size $BATCH_SIZE \
    --output_path ${OUTPUT_BASE}/${TASK}

echo ""
echo "========================================"
echo "测试完成！"
echo "结果保存在: ${OUTPUT_BASE}/${TASK}"
echo "========================================"







