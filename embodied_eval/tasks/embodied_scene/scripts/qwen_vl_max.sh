#!/usr/bin/env bash
# EmbodiedScene — Qwen-VL-Max（API，推理 + LLM-as-Judge 分步）
# 与 all.sh 中对应小节流程一致

cd "$(dirname "$0")/../../../../" || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'
export NCCL_TIMEOUT=1800

INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/qwen-vl-max

python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=qwen-vl-max,max_frames_num=8,max_retries=3,max_new_tokens=2048 \
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
