#!/bin/bash
# 切换到项目根目录
cd "$(dirname "$0")/../../../.." || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")


# CUDA_VISIBLE_DEVICES=0 python -m embodied_eval \
#     --model mimo_embodied \
#     --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=1,torch_dtype=float16 \
#     --evaluator eqa \
#     --tasks vsibench \
#     --batch_size 1 \
#     --output_path ./logs/vsibench/mimo_embodied

CUDA_VISIBLE_DEVICES=0,1 python -m embodied_eval \
  --model mimo_embodied \
  --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=32,torch_dtype=float16,device_map="auto" \
  --evaluator eqa \
  --tasks vsibench \
  --batch_size 1 \
  --output_path ./logs/vsibench/mimo_embodied


# #!/bin/bash
# cd "$(dirname "$0")/../../../../" || exit

# export OPENAI_API_KEY='your-api-key'
# export OPENAI_API_BASE='https://api.gpt.ge/v1'

# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU_GROUPS=("0,1" "2,3" "4,5" "6,7")

# for i in 0 1 2 3; do
#   CUDA_VISIBLE_DEVICES=${GPU_GROUPS[$i]} \
#   python -m embodied_eval \
#     --model mimo_embodied \
#     --model_args model_name_or_path=/path/to/MiMo-Embodied-7B,max_num_frames=8,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks vsibench \
#     --batch_size 1 \
#     --data_shard_idx $i \
#     --num_data_shards 4 \
#     --output_path ./logs/vsibench/mimo_shard_$i \
#     > logs/mimo_$i.log 2>&1 &
# done

# wait
# echo "All shards finished."

    