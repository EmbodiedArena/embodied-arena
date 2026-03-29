export Nuoyin_API_BASE="http://101.132.143.105:5069/v1/chat/completions"

python -m embodied_eval \
    --model nuoyin \
    --model_args model_name_or_path=KnowinBrain,max_retries=2,max_new_tokens=1024 \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --output_path /your/path/to/embodied-eval-main/logs/where2place/nuoyin