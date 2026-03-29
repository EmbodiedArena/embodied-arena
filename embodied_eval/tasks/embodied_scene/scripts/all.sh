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
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/embodied_scene/mimo_embodied

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model internvl3_5 \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/InternVL3_5-8B,max_num_frames=32,fps=2 \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/embodied_scene/internvl3_5-8B

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model embodied_brain \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B,max_num_frames=32,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/embodied_scene/embodied_brain

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model pelican_vl \
#     --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B,max_num_frames=1,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/embodied_scene/pelican_vl

# source /home/tanghyyy/miniconda3/etc/profile.d/conda.sh
# conda activate cambrian

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model cambrian \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B,max_num_frames=32,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --output_path /your/path/to/embodied-eval-main/logs/embodied_scene/cambrian

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m embodied_eval \
#     --model openai_async_compatible \
#     --model_args model_name_or_path=gpt-5.2-2025-12-11,max_frames_num=8,max_retries=3,max_new_tokens=2048 \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --save_results \
#     --output_path /your/path/to/embodied-eval-main/logs/embodied_scene/gpt-5.2

# ============================================================
# Qwen3-VL-8B-Instruct (API 模型，推理+评分分开)
# ============================================================

INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/qwen3_vl

# Step 1: 只跑推理，保存原始响应（避免多卡同步崩溃丢失结果）
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=7 --main_process_port=$PORT -m embodied_eval \
#     --model qwen3_vl \
#     --model_args model_name_or_path=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=10,use_flash_attention_2=True \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# Step 2: 推理完成后，单独跑 LLM-as-Judge 评估（不需要 GPU，不涉及多卡同步）
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# InternVL3.5-8B (本地模型，推理+评分分开)
# ============================================================

INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/internvl3_5-8B

# Step 1: 只跑推理，保存原始响应（避免多卡同步崩溃丢失结果）
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=7 --main_process_port=$PORT -m embodied_eval \
#     --model internvl3_5 \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/InternVL3_5-8B,max_num_frames=32,fps=2 \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# Step 2: 推理完成后，单独跑 LLM-as-Judge 评估（不需要 GPU，不涉及多卡同步）
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# Gemini-2.5-Pro (API 模型，推理+评分分开)
# ============================================================
# INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/gemini-2.5-pro

# # Step 1: 只跑推理
# python -m embodied_eval \
#     --model openai_async_compatible \
#     --model_args model_name_or_path=gemini-2.5-pro,max_frames_num=8,max_retries=3,max_new_tokens=2048 \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# # Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# Mimo-Embodied-7B (本地模型，推理+评分分开)
# ============================================================
INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/mimo_embodied

# Step 1: 只跑推理
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=7 --main_process_port=$PORT -m embodied_eval \
    --model mimo_embodied \
    --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=8 \
    --evaluator eqa \
    --tasks embodied-scene \
    --batch_size 1 \
    --inference_only \
    --output_path $INFER_OUTPUT

# Step 2: LLM-as-Judge 评估
INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
echo "Running post-evaluation on: $INFER_RUN_DIR"
python -m embodied_eval.tasks.embodied_scene.process \
    --base_dir $INFER_RUN_DIR \
    --openai_model gpt-4o-mini

# ============================================================
# GPT-5.2 (API 模型，推理+评分分开)
# ============================================================
# INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/gpt-5.2

# # Step 1: 只跑推理
# python -m embodied_eval \
#     --model openai_async_compatible \
#     --model_args model_name_or_path=gpt-5.2-2025-12-11,max_frames_num=8,max_retries=3,max_new_tokens=2048 \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# # Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# PelicanVL-7B (本地模型，推理+评分分开)
# ============================================================
# INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/pelican_vl

# # Step 1: 只跑推理
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model pelican_vl \
#     --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B,max_num_frames=1,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# # Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# Qwen-VL-Max (API 模型，推理+评分分开)
# ============================================================
# INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/qwen-vl-max

# # Step 1: 只跑推理
# python -m embodied_eval \
#     --model openai_async_compatible \
#     --model_args model_name_or_path=qwen-vl-max,max_frames_num=8,max_retries=3,max_new_tokens=2048 \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# # Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# EmbodiedBrain-7B (本地模型，推理+评分分开)
# ============================================================
INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/embodied_brain

# Step 1: 只跑推理
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=7 --main_process_port=$PORT -m embodied_eval \
#     --model embodied_brain \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/EmbodiedBrain-7B,max_num_frames=32,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# RoboBrain2.0-7B (本地模型，推理+评分分开)
# ============================================================
INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/robobrain2

# Step 1: 只跑推理
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=7 --main_process_port=$PORT -m embodied_eval \
#     --model robobrain2 \
#     --model_args model_name_or_path=BAAI/RoboBrain2.0-7B,max_num_frames=32,use_flash_attention_2=True \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# Cosmos-Reason-1-7B (本地模型，推理+评分分开)
# ============================================================
INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/cosmos_reason1

# Step 1: 只跑推理
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes=7 --main_process_port=$PORT -m embodied_eval \
#     --model cosmos_reason1 \
#     --model_args model_name_or_path=nvidia/Cosmos-Reason1-7B,max_num_frames=32,use_flash_attention_2=True \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# O3 (API 模型，推理+评分分开)
# ============================================================
# INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/o3

# # Step 1: 只跑推理
# python -m embodied_eval \
#     --model openai_async_compatible \
#     --model_args model_name_or_path=o3,max_frames_num=8,max_retries=3,max_new_tokens=2048 \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# # Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# IflyBot-VLM-8B (本地模型，推理+评分分开)
# ============================================================
INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/iflybot_vlm

# Step 1: 只跑推理
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=7 --main_process_port=$PORT -m embodied_eval \
#     --model iflybot_vlm \
#     --model_args model_name_or_path=/home/tanghyyy/data/iFlyBotVLM,max_num_frames=8,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# thinker-4B (本地模型，推理+评分分开)
# ============================================================
INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/thinker_vl

# Step 1: 只跑推理
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=7 --main_process_port=$PORT -m embodied_eval \
#     --model thinker_vl \
#     --model_args model_name_or_path=UBTECH-Robotics/Thinker-4B,max_num_frames=32,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini

# ============================================================
# RynnBrain-8B (本地模型，推理+评分分开)
# ============================================================
# INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/rynnbrain

# Step 1: 只跑推理
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=7 --main_process_port=$PORT -m embodied_eval \
#     --model rynnbrain \
#     --model_args model_name_or_path=/home/tanghyyy/data/RynnBrain-8B,max_num_frames=32,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini


# ============================================================
# Cambrian-S-7B (本地模型，推理+评分分开，需切换 cambrian 环境)
# ============================================================
# INFER_OUTPUT=/your/path/to/embodied-eval-main/logs/embodied_scene/cambrian

# # Step 1: 只跑推理（需先 source conda 并 activate cambrian）
# source /home/tanghyyy/miniconda3/etc/profile.d/conda.sh
# conda activate cambrian
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --num_processes=8 --main_process_port=$PORT -m embodied_eval \
#     --model cambrian \
#     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B,max_num_frames=32,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks embodied-scene \
#     --batch_size 1 \
#     --inference_only \
#     --output_path $INFER_OUTPUT

# # Step 2: LLM-as-Judge 评估
# INFER_RUN_DIR=$(ls -td $INFER_OUTPUT/*/ | head -1)
# echo "Running post-evaluation on: $INFER_RUN_DIR"
# python -m embodied_eval.tasks.embodied_scene.process \
#     --base_dir $INFER_RUN_DIR \
#     --openai_model gpt-4o-mini
