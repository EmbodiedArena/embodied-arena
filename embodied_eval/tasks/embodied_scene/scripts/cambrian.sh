#!/usr/bin/env bash
# EmbodiedScene — Cambrian-S-7B（推理 + LLM-as-Judge 分步；需 cambrian conda 环境）
# 与 all.sh 中对应小节流程一致

cd "$(dirname "$0")/../../../../" || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'
export NCCL_TIMEOUT=1800

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/cambrian

source /home/tanghyyy/miniconda3/etc/profile.d/conda.sh
conda activate cambrian

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
    --model cambrian \
    --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B,max_num_frames=32,use_flash_attention_2=False \
    --evaluator eqa \
    --tasks embodied-scene \
    --batch_size 1 \
    --inference_only \
    --output_path $INFER_OUTPUT

INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
echo "Running post-evaluation on: $INFER_RUN_DIR"
python -m embodied_eval.tasks.embodied_scene.process \
    --base_dir $INFER_RUN_DIR \
    --openai_model gpt-4o-mini
