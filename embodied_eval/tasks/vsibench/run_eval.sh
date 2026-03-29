# #!/bin/bash
# # vsi-bench 评测脚本

# # 切换到项目根目录
# cd "$(dirname "$0")/../../.." || exit

# # API密钥配置（如需要）
# # export OPENAI_API_KEY='your-api-key'
# # export OPENAI_API_BASE='your-api-base'  # 可选

# export OPENAI_API_KEY='your-api-key'
# export OPENAI_API_BASE='https://api.gpt.ge/'  # 可选

# # 获取随机端口
# PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # 运行评测
# CUDA_VISIBLE_DEVICES=0,7 accelerate launch --num_processes=2 --main_process_port=$PORT -m embodied_eval \
#     --model mimo_embodied \
#     --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=16,fps=1,use_flash_attention_2=False \
#     --evaluator eqa \
#     --tasks vsibench \
#     --batch_size 1 \
#     --output_path ./logs/vsibench/mimo_embodied

# # CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
# #     --model pelican_vl \
# #     --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/pelican-vl,max_num_frames=32,fps=2 \
# #     --evaluator eqa \
# #     --tasks vsibench \
# #     --batch_size 1 \
# #     --output_path ./logs/vsibench/pelican_vl_7b

#!/bin/bash

cd "$(dirname "$0")/../../.." || exit

# ===== 关键：解决 CUDA 碎片化 =====
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU
export CUDA_VISIBLE_DEVICES=0,2

PORT=$(python - << 'EOF'
import socket
s=socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
EOF
)

accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --main_process_port=$PORT \
    --mixed_precision=no \
    -m embodied_eval \
    --model mimo_embodied \
    --model_args model_name_or_path=/home/tanghyyy/embodied-arena/embodied_eval/model/XiaomiMiMo/MiMo-Embodied-7B/,max_num_frames=16,fps=1,dtype=bf16,use_flash_attention_2=False,device_map=auto \
    --evaluator eqa \
    --tasks vsibench \
    --batch_size 1 \
    --output_path ./logs/vsibench/mimo_embodied
