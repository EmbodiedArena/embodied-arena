#!/usr/bin/env bash
set -euo pipefail

# 这个脚本用于：对某个 emspatial-bench 的 log 目录进行“空回答回填”
# - 只会修改 samples_emspatial-bench.json 里空的 resps（例如 [[""]]）
# - 不会改动任何已有非空回答

cd "$(dirname "$0")/../../../../" || exit

# ====== API 配置（按需修改）======
# 推荐：在运行前通过环境变量传入，而不是把 key 写进脚本：
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 
: "${OPENAI_API_KEY:?Need OPENAI_API_KEY}"
: "${OPENAI_API_BASE:?Need OPENAI_API_BASE}"

# ====== 数据集路径（可选覆盖）======
# 默认会从 embodied_eval/tasks/emspatial_bench/emspatial-bench.yaml 读取 dataset_path
# 你也可以在这里强制指定：
export EMSPATIAL_DATASET_PATH="${EMSPATIAL_DATASET_PATH:-}"

# ====== 回填模型配置（可按需修改）======
# 注意：这里的 model 名称需要与你的 OPENAI_API_BASE 网关所支持的模型名一致
export EMSPATIAL_REFILL_MODEL="${EMSPATIAL_REFILL_MODEL:-gemini-2.5-pro}"
export EMSPATIAL_REFILL_MAX_NEW_TOKENS="${EMSPATIAL_REFILL_MAX_NEW_TOKENS:-512}"
export EMSPATIAL_REFILL_TEMPERATURE="${EMSPATIAL_REFILL_TEMPERATURE:-0}"
export EMSPATIAL_REFILL_TIMEOUT="${EMSPATIAL_REFILL_TIMEOUT:-60}"
export EMSPATIAL_REFILL_MAX_RETRIES="${EMSPATIAL_REFILL_MAX_RETRIES:-3}"
export EMSPATIAL_REFILL_BATCH_SIZE="${EMSPATIAL_REFILL_BATCH_SIZE:-8}"
export EMSPATIAL_REFILL_RETRY_ON_BLANK="${EMSPATIAL_REFILL_RETRY_ON_BLANK:-1}"

# ====== log 目录配置（只保留这一种方式：直接修改本文件）======
# 把下面这一行改成你要处理的日志目录（目录内应包含 samples_emspatial-bench.json）
LOG_DIR="/your/path/to/embodied-eval-main/logs/emspatial-bench/gemini-2_5-pro/20260203_052537/"

if [[ ! -d "${LOG_DIR}" ]]; then
  echo "ERROR: LOG_DIR not found or not a directory: ${LOG_DIR}" >&2
  exit 1
fi

# ====== Python 解释器选择 ======


python -m embodied_eval.tasks.emspatial_bench.utils.refill_empty_resps \
  --log_dir "${LOG_DIR}" \
  --model "${EMSPATIAL_REFILL_MODEL}" \
  --max_new_tokens "${EMSPATIAL_REFILL_MAX_NEW_TOKENS}" \
  --temperature "${EMSPATIAL_REFILL_TEMPERATURE}" \
  --timeout "${EMSPATIAL_REFILL_TIMEOUT}" \
  --max_retries "${EMSPATIAL_REFILL_MAX_RETRIES}" \
  --batch_size "${EMSPATIAL_REFILL_BATCH_SIZE}" \
  --retry_on_blank "${EMSPATIAL_REFILL_RETRY_ON_BLANK}"
