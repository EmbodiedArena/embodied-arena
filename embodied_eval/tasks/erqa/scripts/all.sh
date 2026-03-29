# ！！使用时切换conda环境，名称为cambrian
# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# 1. 配置 API 密钥 (用于 LLM-as-Judge 评分)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

# export OPENAI_API_KEY='your-api-key'
# export OPENAI_API_BASE='https://openai.arnotho.com/v1'

# NCCL超时设置（增加到30分钟，默认10分钟）
export NCCL_TIMEOUT=1800
# NCCL调试选项（可选，用于排查问题）
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model mimo_embodied \
#     --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=8 \
#     --evaluator eqa \
#     --tasks erqa \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/erqa/mimo_embodied

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model internvl3_5 \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/InternVL3_5-8B,max_num_frames=32,fps=2 \
#     --evaluator eqa \
#     --tasks erqa \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/erqa/internvl3_5-8B

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model embodied_brain \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B,max_num_frames=32,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks erqa \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/erqa/embodied_brain

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model pelican_vl \
#     --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B,max_num_frames=1,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks erqa \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/erqa/pelican_vl

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model qwen3_vl \
#     --model_args model_name_or_path=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=10,use_flash_attention_2=True \
#     --evaluator eqa \
#     --tasks erqa \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/erqa/qwen3_vl


# source /home/tanghyyy/miniconda3/etc/profile.d/conda.sh
# conda activate cambrian

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model cambrian \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B,max_num_frames=32,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks erqa \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/erqa/cambrain

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m embodied_eval \
    --model openai_async_compatible \
    --model_args model_name_or_path=gpt-5.2-2025-12-11,max_frames_num=8,max_retries=3,max_new_tokens=2048 \
    --evaluator eqa \
    --tasks erqa \
    --batch_size 1 \
    --save_results \
    --output_path /your/path/to/embodied-eval-main/logs/erqa/gpt-5.2

