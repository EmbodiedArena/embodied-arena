# ！！使用时切换conda环境，名称为cambrian
# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

# export OPENAI_API_KEY='your-api-key'
# export OPENAI_API_BASE='https://openai.arnotho.com/v1'

# NCCL超时设置（增加到30分钟，默认10分钟）
export NCCL_TIMEOUT=1800
# NCCL调试选项（可选，用于排查问题）
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")


CUDA_VISIBLE_DEVICES=3,4,5,6 python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gpt-5.2,max_frames_num=8,max_retries=3,max_new_tokens=2048 \
    --evaluator eqa \
    --tasks ea-temporal \
    --batch_size 1 \
    --save_results \
    --output_path /your/path/to/embodied-eval-main/logs/ea-temporal/gpt-5.2
