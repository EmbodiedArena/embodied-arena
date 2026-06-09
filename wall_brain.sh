#!/bin/bash
# WALL_BRAIN 本地评测（非 DLC）

cd "$(dirname "$0")" || exit

# =============================================================================
# 可调参数
# =============================================================================

MODEL_NAME="wall_brain_preview"
MODEL_PATH="./path/to/model"

GPUS="0,1,2,3,4,5,6,7"
NUM_PROCESSES=8

# LLM-as-judge 任务（unieqa / openeqa-emeqa 等）需要配置
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://dashscope.aliyuncs.com/compatible-mode/v1'

TASKS=(
    "where2place-point"
    "erqa"
    "cosmos-agibot"
    "cosmos-bridgev2"
    "cosmos-robovqa"
    "cosmos-holoassist"
    "cosmos-robofail"
    "vsibench"
    "emspatial-bench"
    "unieqa"
    "openeqa-emeqa"
    "robovqa"
    "vabench"
)

# =============================================================================
# 运行
# =============================================================================

export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

get_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

echo "================================"
echo "WALL_BRAIN 本地评测"
echo "================================"
echo "模型: ${MODEL_PATH}"
echo "GPU:  ${GPUS} (num_processes=${NUM_PROCESSES})"
echo "任务: ${#TASKS[@]} 个"
echo "================================"
echo ""

for TASK in "${TASKS[@]}"; do
    [ -z "$TASK" ] && continue
    TASK_DIR="${TASK%%-*}"
    OUTPUT_PATH="./logs/${MODEL_NAME}/${TASK_DIR}"
    PORT=$(get_port)

    echo "运行任务: ${TASK}"
    echo "  输出目录: ${OUTPUT_PATH}"

    CUDA_VISIBLE_DEVICES=${GPUS} accelerate launch \
        --num_processes=${NUM_PROCESSES} \
        --main_process_port=${PORT} \
        -m embodied_eval \
        --model ${MODEL_NAME} \
        --model_args model_name_or_path=${MODEL_PATH} \
        --evaluator eqa \
        --tasks ${TASK} \
        --output_path ${OUTPUT_PATH}

    echo ""
done

echo "================================"
echo "所有任务完成"
echo "================================"
