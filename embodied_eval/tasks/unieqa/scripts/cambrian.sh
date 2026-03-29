#!/bin/bash

# ！！使用时切换conda环境，名称为cambrian
# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

# 2. 创建日志目录
LOG_DIR="./logs/unieqa/cambrian"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/cambrian_$(date +%Y%m%d_%H%M%S).log"

# 3. 使用 nohup 后台运行评测
nohup python -m embodied_eval \
    --model cambrian \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B,max_num_frames=10,use_flash_attention_2=True \
    --evaluator eqa \
    --tasks unieqa \
    --batch_size 1 \
    --inference_only \
    --output_path "$LOG_DIR" \
    > "$LOG_FILE" 2>&1 &

# 4. 输出进程信息
PID=$!
echo "Cambrian 评测已启动，进程 ID: $PID"
echo "日志文件: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
echo "停止进程: kill $PID"
