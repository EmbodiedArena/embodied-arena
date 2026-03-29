export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/'

export Nuoyin_API_BASE="http://101.132.143.105:5069/v1/chat/completions"

python -m embodied_eval \
    --model nuoyin \
    --model_args model_name_or_path=KnowinBrain,max_retries=2,max_new_tokens=1024 \
    --evaluator eqa \
    --tasks ea-temporal \
    --batch_size 1 \
    --output_path /your/path/to/embodied-eval-main/logs/ea-temporal/nuoyin