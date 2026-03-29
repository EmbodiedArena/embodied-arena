#!/bin/bash
# Cosmos Reasoning Benchmark 评估脚本 - 支持多GPU并行

# 切换到项目根目录
cd "$(dirname "$0")/../../.." || exit

# ============ 配置参数 ============
# 模型路径
MODEL_PATH="X-Humanoid/Pelican1.0-VL-7B"
MODEL_TYPE="pelican_vl"

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'

# GPU设置 - 可用的GPU列表（逗号分隔）
# 通过环境变量覆盖: GPUS="0,1" bash run_eval.sh
GPUS=${GPUS:-"2,3,4,7"}

# 将GPU字符串转换为数组
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# 输出路径
OUTPUT_BASE="./logs/cosmos/pelican_vl_7b"

# 模型参数
BATCH_SIZE=1
MAX_NUM_FRAMES=32
FPS=2

# 并行模式: auto(自动), parallel(强制并行), sequential(强制顺序)
PARALLEL_MODE=${PARALLEL_MODE:-"auto"}

# ============ 显示配置信息 ============
echo "========================================"
echo "Cosmos Reasoning Benchmark 评估"
echo "========================================"
echo "模型: $MODEL_PATH"
echo "模型类型: $MODEL_TYPE"
echo "可用GPU: $GPUS (共 $NUM_GPUS 个)"
echo "并行模式: $PARALLEL_MODE"
echo "输出路径: $OUTPUT_BASE"
echo "批次大小: $BATCH_SIZE"
echo "视频帧数: $MAX_NUM_FRAMES"
echo "========================================"
echo ""

# 任务列表
TASKS=("bridgev2" "robovqa" "agibot" "holoassist" "robofail")
TASK_COUNT=${#TASKS[@]}

# ============ 定义运行函数 ============
run_task() {
    local TASK=$1
    local GPU_ID=$2
    local TASK_NUM=$3
    
    # 获取随机端口避免冲突
    local PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$TASK_NUM/$TASK_COUNT] 在GPU $GPU_ID 上启动 $TASK 任务..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch \
        --num_processes=1 \
        --main_process_port=$PORT \
        -m embodied_eval \
        --model $MODEL_TYPE \
        --model_args model_name_or_path=$MODEL_PATH,max_num_frames=$MAX_NUM_FRAMES,fps=$FPS \
        --evaluator eqa \
        --tasks cosmos-$TASK \
        --batch_size $BATCH_SIZE \
        --output_path ${OUTPUT_BASE}/${TASK} 2>&1 | tee ${OUTPUT_BASE}/${TASK}_$(date +%Y%m%d_%H%M%S).log
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$TASK_NUM/$TASK_COUNT] ✓ $TASK 任务完成（GPU $GPU_ID）"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$TASK_NUM/$TASK_COUNT] ✗ $TASK 任务失败（GPU $GPU_ID），退出码: $EXIT_CODE"
    fi
    
    return $EXIT_CODE
}

# ============ 决定运行模式 ============
USE_PARALLEL=false

if [ "$PARALLEL_MODE" = "parallel" ]; then
    USE_PARALLEL=true
    echo "使用并行模式运行（强制）"
elif [ "$PARALLEL_MODE" = "sequential" ]; then
    USE_PARALLEL=false
    echo "使用顺序模式运行（强制）"
elif [ "$PARALLEL_MODE" = "auto" ]; then
    if [ $NUM_GPUS -gt 1 ]; then
        USE_PARALLEL=true
        echo "检测到多个GPU，使用并行模式"
    else
        USE_PARALLEL=false
        echo "检测到单个GPU，使用顺序模式"
    fi
fi
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# ============ 执行任务 ============
if [ "$USE_PARALLEL" = true ]; then
    # ------------ 并行模式 ------------
    echo "开始并行运行 $TASK_COUNT 个任务..."
    echo ""
    
    declare -a PIDS=()
    declare -a TASK_INFO=()
    
    # 启动所有任务
    for i in "${!TASKS[@]}"; do
        TASK=${TASKS[$i]}
        TASK_NUM=$((i + 1))
        
        # 循环分配GPU
        GPU_INDEX=$((i % NUM_GPUS))
        GPU_ID=${GPU_ARRAY[$GPU_INDEX]}
        
        # 后台运行
        run_task "$TASK" "$GPU_ID" "$TASK_NUM" &
        PID=$!
        
        PIDS+=($PID)
        TASK_INFO+=("$TASK:GPU$GPU_ID:PID$PID")
        
        echo "[启动] $TASK (GPU $GPU_ID, PID: $PID)"
        
        # 如果任务数超过GPU数，分批运行
        if [ $(( (i + 1) % NUM_GPUS )) -eq 0 ] && [ $i -lt $((TASK_COUNT - 1)) ]; then
            echo ""
            echo "等待当前批次（${PIDS[@]}）完成..."
            
            for pid in "${PIDS[@]}"; do
                wait $pid
            done
            
            PIDS=()
            TASK_INFO=()
            echo ""
            echo "当前批次完成，继续下一批..."
            echo ""
        fi
    done
    
    # 等待最后一批完成
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo ""
        echo "等待最后一批任务完成..."
        echo "运行中的任务: ${TASK_INFO[@]}"
        echo ""
        
        FAILED=()
        for i in "${!PIDS[@]}"; do
            pid=${PIDS[$i]}
            info=${TASK_INFO[$i]}
            task=$(echo $info | cut -d: -f1)
            
            if wait $pid; then
                echo "✓ $task 成功"
            else
                echo "✗ $task 失败"
                FAILED+=("$task")
            fi
        done
        
        echo ""
        if [ ${#FAILED[@]} -gt 0 ]; then
            echo "⚠️  警告: ${#FAILED[@]} 个任务失败: ${FAILED[*]}"
        else
            echo "✅ 所有任务成功完成！"
        fi
    fi
    
else
    # ------------ 顺序模式 ------------
    echo "开始顺序运行 $TASK_COUNT 个任务..."
    echo ""
    
    GPU_ID=${GPU_ARRAY[0]}
    FAILED=()
    
    for i in "${!TASKS[@]}"; do
        TASK=${TASKS[$i]}
        TASK_NUM=$((i + 1))
        
        if ! run_task "$TASK" "$GPU_ID" "$TASK_NUM"; then
            FAILED+=("$TASK")
        fi
        echo ""
    done
    
    if [ ${#FAILED[@]} -gt 0 ]; then
        echo "⚠️  警告: ${#FAILED[@]} 个任务失败: ${FAILED[*]}"
    else
        echo "✅ 所有任务成功完成！"
    fi
fi

echo ""
echo "========================================"
echo "所有任务执行完毕！"
echo "结果保存在:"
for TASK in "${TASKS[@]}"; do
    echo "  - $TASK: ${OUTPUT_BASE}/${TASK}"
done
echo "========================================"

# ============ 聚合结果 ============
echo ""
echo "正在聚合所有子任务结果..."
echo ""

python - << 'EOF'
import json
import os
from pathlib import Path

output_base = "./logs/cosmos/pelican_vl_7b"
tasks = ["bridgev2", "robovqa", "agibot", "holoassist", "robofail"]

all_results = {}
task_accuracies = []
task_results = {}

print("正在读取各子任务结果...")
for task in tasks:
    task_dir = Path(output_base) / task
    if not task_dir.exists():
        print(f"⚠️  {task} 目录不存在，跳过")
        continue
    
    # 查找最新的结果文件
    timestamp_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir()], 
                           key=lambda x: x.name, reverse=True)
    
    if not timestamp_dirs:
        print(f"⚠️  {task} 没有找到结果目录，跳过")
        continue
    
    latest_dir = timestamp_dirs[0]
    result_file = latest_dir / f"results_cosmos-{task}.json"
    
    if not result_file.exists():
        print(f"⚠️  {result_file} 不存在，跳过")
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
        task_results[task] = accuracy
        print(f"  ✓ {task:12s}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print(f"  ⚠️  {task}: 未找到准确率")

print("")
if task_accuracies:
    avg_accuracy = sum(task_accuracies) / len(task_accuracies)
    all_results["accuracy_average"] = avg_accuracy
    all_results["overall"] = avg_accuracy
    
    print(f"{'='*50}")
    print(f"平均准确率: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"完成任务数: {len(task_accuracies)}/{len(tasks)}")
    print(f"{'='*50}")
    
    # 保存聚合结果
    output_file = Path(output_base) / "aggregated_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 聚合结果已保存到: {output_file}")
else:
    print("⚠️  没有找到有效的结果")

EOF

echo ""
echo "========================================"
echo "评估完成！"
echo "========================================"
