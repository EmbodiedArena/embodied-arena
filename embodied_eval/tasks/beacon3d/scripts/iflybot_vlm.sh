#!/usr/bin/env bash
set -euo pipefail

# Beacon3D - iFlyBotVLM
#
# Runs all Beacon3D tasks:
# - beacon3d-grounding
# - beacon3d-qa / beacon3d-qa_scannet / beacon3d-qa_3rscan / beacon3d-qa_multiscan

cd "$(dirname "$0")/../../../../" || exit 1

MODEL_TYPE=${MODEL_TYPE:-"iflybot_vlm"}
MODEL_PATH=${MODEL_PATH:-"/home/tanghyyy/data/iFlyBotVLM"}

GPU_ID=${GPU_ID:-"${CUDA_VISIBLE_DEVICES:-0}"}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUTPUT_BASE=${OUTPUT_BASE:-"./logs/beacon3d/iflybot_vlm"}
RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}

BATCH_SIZE=${BATCH_SIZE:-1}
NUM_FRAME=${NUM_FRAME:-32}
MAX_NUM=${MAX_NUM:-12}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-False}

LIMIT=${LIMIT:-""} # e.g. LIMIT=10 for quick test

PORT=$(${PYTHON_BIN:-python} -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

TASKS="beacon3d-grounding,beacon3d-qa,beacon3d-qa_scannet,beacon3d-qa_3rscan,beacon3d-qa_multiscan"
OUT_DIR="${OUTPUT_BASE}/${RUN_ID}"
mkdir -p "$OUT_DIR"

echo "========================================"
echo "Beacon3D - iFlyBotVLM"
echo "GPU=$CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH=$MODEL_PATH"
echo "TASKS=$TASKS"
echo "OUT=$OUT_DIR"
echo "========================================"

extra_args=()
if [ -n "$LIMIT" ]; then
  extra_args+=(--limit "$LIMIT")
fi

accelerate launch \
  --num_processes=1 \
  --main_process_port="$PORT" \
  -m embodied_eval \
  --model "$MODEL_TYPE" \
  --model_args "model_name_or_path=$MODEL_PATH,num_frame=$NUM_FRAME,max_num=$MAX_NUM,use_flash_attn=$USE_FLASH_ATTN" \
  --evaluator eqa \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUT_DIR" "${extra_args[@]}" 2>&1 | tee "${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

