# export OPENAI_API_KEY="your-api-key"
# export OPENAI_API_BASE="https://openai.arnotho.com/v1"
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/'

export Nuoyin_API_BASE="http://101.132.143.105:5069/v1"

python -m embodied_eval \
    --model nuoyin \
    --model_args model_name_or_path=KnowinBrain,max_retries=2,timeout=400 \
    --evaluator eqa \
    --tasks robovqa \
    --batch_size 1 \
    --save_results \
    --output_path /your/path/to/embodied-eval-main/logs/robovqa/nuoyin