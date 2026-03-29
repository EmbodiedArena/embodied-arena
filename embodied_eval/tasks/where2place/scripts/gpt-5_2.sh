# export OPENAI_API_KEY='your-api-key'
# export OPENAI_API_BASE='https://api.gpt.ge/v1/'
# export OPENAI_API_BASE='https://api.gpt.ge/v1/responses'
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://yunwu.ai/v1'

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

LOG_DIR="./logs/where2place/gpt-5.2"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/gpt-5.2_$(date +%Y%m%d_%H%M%S).log"

python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gpt-5.2,max_retries=2 \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --save_results \
    --output_path "$LOG_DIR" \