export Nuoyin_API_BASE="http://101.132.143.105:5069/v1"

python -m embodied_eval \
    --model nuoyin \
    --model_args model_name_or_path=KnowinBrain,max_retries=2,timeout=300 \
    --evaluator eqa \
    --tasks openeqa-emeqa \
    --batch_size 1 \
    --output_path /your/path/to/embodied-eval-main/logs/openeqa/nuoyin