#!/bin/bash
# Where2Place 评估脚本 - PelicanVL 模型

# 切换到项目根目录
cd "$(dirname "$0")/../../.." || exit

# ============ 配置参数 ============
# 模型路径（修改为你的实际路径）
MODEL_PATH="X-Humanoid/Pelican1.0-VL-7B"
# 或使用本地路径：
# MODEL_PATH="/data/models/Pelican1.0-VL-7B"

# GPU 设置
GPU_ID=0

# 输出路径
OUTPUT_BASE="./logs/where2place/pelican_vl_7b"

# 批次大小
BATCH_SIZE=1

# 视频处理参数
MAX_NUM_FRAMES=32
FPS=2

# ============ 运行评估 ============
# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

echo "========================================"
echo "Where2Place 评估 - PelicanVL"
echo "========================================"
echo "模型: $MODEL_PATH"
echo "GPU: $GPU_ID"
echo "输出: $OUTPUT_BASE"
echo "========================================"
echo ""

# 任务 1: Point 检测
echo "[1/2] 运行 Point 检测任务..."
CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch \
    --num_processes=1 \
    --main_process_port=$PORT \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=$MODEL_PATH,max_num_frames=$MAX_NUM_FRAMES,fps=$FPS \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size $BATCH_SIZE \
    --output_path ${OUTPUT_BASE}_point

echo ""
echo "[1/2] Point 任务完成"
echo ""

# # 任务 2: BBox 检测
# echo "[2/2] 运行 BBox 检测任务..."
# CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch \
#     --num_processes=1 \
#     --main_process_port=$PORT \
#     -m embodied_eval \
#     --model pelican_vl \
#     --model_args model_name_or_path=$MODEL_PATH,max_num_frames=$MAX_NUM_FRAMES,fps=$FPS \
#     --evaluator eqa \
#     --tasks where2place-bbox \
#     --batch_size $BATCH_SIZE \
#     --output_path ${OUTPUT_BASE}_bbox

# echo ""
# echo "[2/2] BBox 任务完成"
# echo ""
# echo "========================================"
# echo "所有任务完成！"
# echo "结果保存在:"
# echo "  - Point: ${OUTPUT_BASE}_point"
# echo "  - BBox:  ${OUTPUT_BASE}_bbox"
# echo "========================================"
