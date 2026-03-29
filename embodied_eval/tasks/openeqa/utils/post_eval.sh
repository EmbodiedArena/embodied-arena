#!/usr/bin/env bash
set -euo pipefail

# 直接在此处配置你的 OpenAI 网关与 Key
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'  # 必须有/v1后缀，/v1/也可，无区别

# 直接在此处配置要重打分的日志目录（里面应包含 samples_openeqa-emeqa.json / results_openeqa-emeqa.json）
BASE_DIR="/your/path/to/embodied-eval-main/logs/openeqa/qwen2_5-vl-7b-instruct/20260125_211253/"

python -m embodied_eval.tasks.openeqa.process \
  --base_dir "${BASE_DIR}" \
  --openai_model "gpt-4o" \
  --openai_temperature 0.2 
