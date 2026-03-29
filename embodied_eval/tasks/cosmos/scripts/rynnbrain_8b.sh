#!/bin/bash
# Cosmos Reasoning Benchmark - RynnBrain-8B 评估脚本

# 切换到项目根目录
cd "$(dirname "$0")/../../../../" || exit

# API密钥配置（如需要）
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/'

# 配置GPU
GPUS="0"

# 获取随机端口
get_port() {
    python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

# 模型配置
MODEL_PATH="/your/path/to/embodied-eval-main/embodied_eval/data/RynnBrain-8B"
MODEL_TYPE="rynnbrain"
OUTPUT_BASE="./logs/cosmos/rynnbrain"

# 任务列表
TASKS=("bridgev2" "robovqa" "agibot" "holoassist" "robofail")

echo "================================"
echo "Cosmos - RynnBrain-8B 评估"
echo "================================"
echo "模型: $MODEL_PATH"
echo "GPU: $GPUS"
echo "================================"
echo ""

# 顺序运行所有任务
for task in "${TASKS[@]}"; do
    echo "运行任务: $task"
    PORT=$(get_port)
    
    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
        --num_processes=1 \
        --main_process_port=$PORT \
        -m embodied_eval \
        --model $MODEL_TYPE \
        --model_args model_name_or_path=$MODEL_PATH,max_num_frames=32,use_flash_attention_2=false \
        --evaluator eqa \
        --tasks cosmos-$task \
        --batch_size 1 \
        --output_path ${OUTPUT_BASE}/${task}
    
    echo ""
done

echo ""
echo "================================"
echo "所有任务完成！"
echo "================================"
echo ""

# ============ 聚合结果 ============
echo "正在聚合评估结果..."
echo ""

python - << EOF
import json
import os
from pathlib import Path

output_base = "${OUTPUT_BASE}"
tasks = ["bridgev2", "robovqa", "agibot", "holoassist", "robofail"]

all_results = {}
task_accuracies = []

print("正在读取各子任务结果...")
print("-" * 50)

for task in tasks:
    task_dir = Path(output_base) / task
    if not task_dir.exists():
        print(f"⚠️  {task:12s}: 目录不存在")
        continue
    
    # 查找最新的结果文件
    timestamp_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir()], 
                           key=lambda x: x.name, reverse=True)
    
    if not timestamp_dirs:
        print(f"⚠️  {task:12s}: 未找到结果目录")
        continue
    
    latest_dir = timestamp_dirs[0]
    result_file = latest_dir / f"results_cosmos-{task}.json"
    
    if not result_file.exists():
        print(f"⚠️  {task:12s}: 结果文件不存在")
        continue
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # 提取准确率
    accuracy = None
    for key in [f"{task}_accuracy", "overall_accuracy", "overall"]:
        if key in results:
            accuracy = results[key]
            break
    
    if accuracy is not None:
        all_results[f"{task}_accuracy"] = accuracy
        task_accuracies.append(accuracy)
        print(f"✓  {task:12s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print(f"⚠️  {task:12s}: 未找到准确率")

print("-" * 50)

if task_accuracies:
    avg_accuracy = sum(task_accuracies) / len(task_accuracies)
    all_results["accuracy_average"] = avg_accuracy
    all_results["overall"] = avg_accuracy
    
    print("")
    print("=" * 50)
    print(f"平均准确率: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"完成任务数: {len(task_accuracies)}/{len(tasks)}")
    print("=" * 50)
    
    # 保存聚合结果
    output_file = Path(output_base) / "aggregated_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 聚合结果已保存到: {output_file}")
else:
    print("\n⚠️  没有找到有效的结果")

EOF

echo ""
echo "================================"
echo "评估完成！"
echo "结果保存在: ${OUTPUT_BASE}/"
echo "================================"
