
# ！！使用时切换conda环境，名称为cambrian
# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

# 2. 运行评测
CUDA_VISIBLE_DEVICES=5 python -m embodied_eval \
    --model cambrian \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B,max_num_frames=32,use_flash_attention_2=False \
    --evaluator eqa \
    --tasks robovqa \
    --batch_size 1 \
    --output_path ./logs/robovqa/cambrain
