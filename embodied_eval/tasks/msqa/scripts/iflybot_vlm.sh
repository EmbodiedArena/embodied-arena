#!/usr/bin/env bash
set -euo pipefail

# MSQA - iFlyBotVLM

cd "$(dirname "$0")/../../../../" || exit 1

MODEL_TYPE=${MODEL_TYPE:-"iflybot_vlm"}
MODEL_PATH=${MODEL_PATH:-"/home/tanghyyy/data/iFlyBotVLM"}

GPU_ID=${GPU_ID:-"${CUDA_VISIBLE_DEVICES:-0}"}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUTPUT_BASE=${OUTPUT_BASE:-"./logs/msqa/iflybot_vlm"}
RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}

BATCH_SIZE=${BATCH_SIZE:-1}
NUM_FRAME=${NUM_FRAME:-32}
MAX_NUM=${MAX_NUM:-12}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-False}

PORT=$(${PYTHON_BIN:-python} -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
OUT_DIR="${OUTPUT_BASE}/${RUN_ID}"
mkdir -p "$OUT_DIR"

accelerate launch \
  --num_processes=1 \
  --main_process_port="$PORT" \
  -m embodied_eval \
  --model "$MODEL_TYPE" \
  --model_args "model_name_or_path=$MODEL_PATH,num_frame=$NUM_FRAME,max_num=$MAX_NUM,use_flash_attn=$USE_FLASH_ATTN" \
  --evaluator eqa \
  --tasks msqa \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUT_DIR" 2>&1 | tee "${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

