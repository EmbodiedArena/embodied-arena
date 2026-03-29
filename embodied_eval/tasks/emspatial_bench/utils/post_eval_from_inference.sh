#!/usr/bin/env bash
set -euo pipefail

# 直接在此处配置你的 OpenAI 网关与 Key
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'  # 必须有/v1后缀，/v1/也可，无区别

# 直接在此处配置要重打分的日志目录（里面应包含 samples_emspatial-bench.json / results_emspatial-bench.json）
BASE_DIR="/your/path/to/embodied-eval-main/logs/emspatial-bench/MiMo-Embodied-7B/20260128_031703/"

python -m embodied_eval.tasks.emspatial_bench.utils.postprocess \
  --input_file "$BASE_DIR/inference_emspatial-bench.json" \
  --output_dir "$BASE_DIR"
