#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/../../../.." || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 2. 解决 CUDA 内存碎片化问题，避免 Segmentation fault
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 3. 禁用 tokenizer 并行，避免潜在的冲突
export TOKENIZERS_PARALLELISM=false

export PROJECT_ROOT=/your/path/to/embodied-eval-main
export DATA_ROOT=${PROJECT_ROOT}/embodied_eval/data/vsi-bench-data/vsibench_hf_splits4
export OUTPUT_ROOT=${PROJECT_ROOT}/logs/vsibench/step3
export MODEL_NAME=/your/path/to/embodied-eval-main/embodied_eval/data/Step3-VL-10B
export NUM_SPLITS=4

export PYTHONUNBUFFERED=1

# python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_NAME', fix_mistral_regex=True)"

for i in $(seq 0 $((NUM_SPLITS-1)))
do
(
  echo ">>> Launch split ${i}/${NUM_SPLITS}"

  SUBSET="vsibench${i}"
  OUT_DIR=${OUTPUT_ROOT}/split_${i}

  CUDA_VISIBLE_DEVICES="${i}" python -m embodied_eval \
    --model step3_vl \
    --model_args model_name_or_path=$MODEL_NAME,max_frames_num=32 \
    --evaluator eqa \
    --tasks "$SUBSET" \
    --batch_size 1 \
    --save_results \
    --output_path "$OUT_DIR"
) &
done

wait
echo "✅ All splits finished"
