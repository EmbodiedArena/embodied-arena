#!/bin/bash
set -e

cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' # 确保末尾带 v1

export NCCL_TIMEOUT=1800

# 获取随机端口
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# =========================
# 配置
# =========================
export PROJECT_ROOT=/your/path/to/embodied-eval-main
export DATA_ROOT=${PROJECT_ROOT}/embodied_eval/data/vsi-bench-data/vsibench_hf_splits16
export OUTPUT_ROOT=${PROJECT_ROOT}/logs/vsibench/gpt5_2
export MODEL_NAME=gpt-5.2
export NUM_SPLITS=16

mkdir -p "$OUTPUT_ROOT"

# =========================
# 并行运行四个子集
# =========================
for i in $(seq 0 $((NUM_SPLITS-1)))
do
(
  echo ">>> Launch split ${i}/${NUM_SPLITS}"

  SUBSET="vsibench${i}"
  OUT_DIR=${OUTPUT_ROOT}/split_${i}
  mkdir -p "$OUT_DIR"

  python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=$MODEL_NAME,max_frames_num=32,max_retries=3,max_new_tokens=2048 \
    --evaluator eqa \
    --tasks "$SUBSET" \
    --batch_size 1 \
    --save_results \
    --output_path "$OUT_DIR"
) &
done

wait
echo "✅ All splits finished"
