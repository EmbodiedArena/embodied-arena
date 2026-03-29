#!/usr/bin/env bash

# VABench-P 通用评测脚本（参考 RoboVQA）

# 切换到项目根目录
cd "$(dirname "$0")/../../.." || exit 1

######################
# 1. 可选：配置 LLM-as-Judge API（如果评估器需要）
######################
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1' 

######################
# 2. 运行评测
######################

# 随机找一个空闲端口（如果使用 accelerate，可复用）
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# 示例：使用本地/HF 上的多模态模型（请按实际情况修改）
# 这里示例使用 qwen2_5_vl，你也可以替换为 internvl3_5、mimo_embodied、embodied_brain 等，
# 只需对应修改 --model 和 --model_args。

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m embodied_eval \
  --model cambrian \
  --model_args model_name_or_path=/your/path/to/embodied-eval-main/embodied_eval/data/Cambrian-S-7B,max_num_frames=32,use_flash_attention_2=False \
  --evaluator eqa \
  --tasks vabench \
  --batch_size 1 \
  --output_path ./logs/vabench/cambrian

# 如果你想和 RoboVQA 一样，为不同模型建独立脚本：
#   - 复制本文件为 scripts/cambrian.sh / scripts/embodied_brain.sh 等
#   - 修改 --model / --model_args / --output_path 即可

