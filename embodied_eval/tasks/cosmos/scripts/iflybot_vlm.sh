#!/usr/bin/env bash
set -euo pipefail

# Cosmos Reasoning Benchmark - iFlyBotVLM
#
# Usage:
#   GPUS="0,1,2,3,4,5,6,7" MODEL_PATH="/home/tanghyyy/data/iFlyBotVLM" bash embodied_eval/tasks/cosmos/scripts/iflybot_vlm.sh
#
# Notes:
# - This script does NOT set OPENAI_API_KEY / OPENAI_API_BASE.
# - By default it runs 5 subtasks in parallel batches across the GPUs in $GPUS.

cd "$(dirname "$0")/../../../../" || exit 1

MODEL_TYPE=${MODEL_TYPE:-"iflybot_vlm"}
MODEL_PATH=${MODEL_PATH:-"/home/tanghyyy/data/iFlyBotVLM"}

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

OUTPUT_BASE=${OUTPUT_BASE:-"./logs/cosmos/iflybot_vlm"}
RUN_ID=${RUN_ID:-"$(date +%Y%m%d_%H%M%S)"}

BATCH_SIZE=${BATCH_SIZE:-1}
NUM_FRAME=${NUM_FRAME:-32}
MAX_NUM=${MAX_NUM:-12}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-False}

PARALLEL_MODE=${PARALLEL_MODE:-"auto"} # auto|parallel|sequential

TASKS=("bridgev2" "robovqa" "agibot" "holoassist" "robofail")
TASK_COUNT=${#TASKS[@]}

get_port() {
  ${PYTHON_BIN:-python} -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

run_task() {
  local task="$1"
  local gpu_id="$2"
  local task_num="$3"
  local port
  port="$(get_port)"

  local out_dir="${OUTPUT_BASE}/${RUN_ID}/${task}"
  mkdir -p "${OUTPUT_BASE}/${RUN_ID}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$task_num/$TASK_COUNT] GPU=$gpu_id task=cosmos-$task"
  CUDA_VISIBLE_DEVICES="$gpu_id" accelerate launch \
    --num_processes=1 \
    --main_process_port="$port" \
    -m embodied_eval \
    --model "$MODEL_TYPE" \
    --model_args "model_name_or_path=$MODEL_PATH,num_frame=$NUM_FRAME,max_num=$MAX_NUM,use_flash_attn=$USE_FLASH_ATTN" \
    --evaluator eqa \
    --tasks "cosmos-$task" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$out_dir" 2>&1 | tee "${OUTPUT_BASE}/${RUN_ID}/${task}_$(date +%Y%m%d_%H%M%S).log"
}

echo "========================================"
echo "Cosmos - iFlyBotVLM"
echo "MODEL_PATH=$MODEL_PATH"
echo "GPUS=$GPUS (count=$NUM_GPUS)"
echo "OUTPUT_BASE=$OUTPUT_BASE"
echo "RUN_ID=$RUN_ID"
echo "PARALLEL_MODE=$PARALLEL_MODE"
echo "num_frame=$NUM_FRAME max_num=$MAX_NUM batch_size=$BATCH_SIZE"
echo "========================================"
echo ""

use_parallel=false
if [ "$PARALLEL_MODE" = "parallel" ]; then
  use_parallel=true
elif [ "$PARALLEL_MODE" = "sequential" ]; then
  use_parallel=false
else
  if [ "$NUM_GPUS" -gt 1 ]; then
    use_parallel=true
  fi
fi

if [ "$use_parallel" = true ]; then
  echo "Parallel mode: run subtasks in GPU batches"
  echo ""
  declare -a PIDS=()
  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    task_num=$((i + 1))
    gpu="${GPU_ARRAY[$((i % NUM_GPUS))]}"
    run_task "$task" "$gpu" "$task_num" &
    PIDS+=("$!")
    if [ $(( (i + 1) % NUM_GPUS )) -eq 0 ] && [ "$i" -lt $((TASK_COUNT - 1)) ]; then
      for pid in "${PIDS[@]}"; do
        wait "$pid"
      done
      PIDS=()
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
else
  echo "Sequential mode: run subtasks one by one"
  echo ""
  gpu="${GPU_ARRAY[0]}"
  for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    task_num=$((i + 1))
    run_task "$task" "$gpu" "$task_num"
    echo ""
  done
fi

echo ""
echo "✅ Cosmos finished. Outputs under: ${OUTPUT_BASE}/${RUN_ID}/"

