# export OPENAI_API_KEY='your-api-key'
# export OPENAI_API_BASE='https://openai.arnotho.com/v1'
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 
# 2. 运行评测
# 使用 openai_async_compatible 模型类
CUDA_VISIBLE_DEVICES=3 python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gpt-5.2,max_frames_num=10,max_retries=1,max_new_tokens=2048 \
    --evaluator eqa \
    --tasks robovqa \
    --batch_size 1 \
    --output_path ./logs/robovqa/gpt-5.2