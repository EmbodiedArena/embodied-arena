export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 确保末尾带 v1

python -m embodied_eval.tasks.unieqa.process \
    --base_dir /your/path/to/embodied-eval-main/logs/unieqa/nuoyin/20260121_173103