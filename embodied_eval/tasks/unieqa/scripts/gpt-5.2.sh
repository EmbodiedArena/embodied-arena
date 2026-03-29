#!/bin/bash
# GPT-5.2 Chat Latest UniEQA 评测脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

export CUDA_VISIBLE_DEVICES=3,4,5,6
# 2. 创建日志目录
LOG_DIR="./logs/unieqa/gpt-5.2"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/gpt-5.2_$(date +%Y%m%d_%H%M%S).log"

# 3. 直接运行评测 (不使用 accelerate launch 以节省显存)
nohup python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gpt-5.2,max_frames_num=15,max_retries=3,max_new_tokens=2048 \
    --evaluator eqa \
    --tasks unieqa \
    --batch_size 1 \
    --inference_only \
    --output_path "$LOG_DIR" \
    > "$LOG_FILE" 2>&1 &

# 4. 输出进程信息
PID=$!
echo "GPT-5.2 Chat Latest 评测已启动，进程 ID: $PID"
echo "日志文件: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
echo "停止进程: kill $PID"
