export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/'

export Nuoyin_API_BASE="http://101.132.143.105:5069/v1/"
# export Nuoyin_API_BASE="http://101.132.143.105:5069/v1/chat/completions"

python -m embodied_eval \
    --model nuoyin \
    --model_args model_name_or_path=KnowinBrain,max_retries=1,timeout=200 \
    --evaluator eqa \
    --tasks unieqa \
    --batch_size 1 \
    --output_path /your/path/to/embodied-eval-main/logs/unieqa/nuoyin