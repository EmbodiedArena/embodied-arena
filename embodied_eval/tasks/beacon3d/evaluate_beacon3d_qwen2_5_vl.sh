#!/bin/bash

# Beacon3D Evaluation Script for Qwen2.5-VL Model
# This script provides complete evaluation commands for Beacon3D tasks

set -e  # Exit on any error

echo "=================================================="
echo "Beacon3D Evaluation with Qwen2.5-VL Model"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "embodied_eval/models/qwen2_5_vl.py" ]; then
    echo "Error: Please run this script from the embodied-eval-main directory"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: embodied-eval"
conda activate embodied-eval

# Set environment variables
export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

# Check CUDA availability
CUDA_AVAILABLE=$(python -c "import torch; print('1' if torch.cuda.is_available() else '0')")
if [ "$CUDA_AVAILABLE" = "1" ]; then
    echo "✓ CUDA is available - using GPU acceleration"
else
    echo "⚠ CUDA not available - using CPU (this will be slow)"
fi

# Get a random port
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Using port: $PORT"

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""
echo "Choose evaluation type:"
echo "1) Quick test (5 samples each task)"
echo "2) Full Beacon3D-QA evaluation"
echo "3) Full Beacon3D-Grounding evaluation"
echo "4) Combined evaluation (both tasks)"
echo "5) Custom evaluation"
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "Running quick test..."
        echo "This will evaluate 5 samples from each Beacon3D task"
        echo ""

        # Quick test for both tasks
        accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
            --model qwen2_5_vl \
            --model_args model_name_or_path=/data/arena/models/Qwen2_5-VL-3B-Instruct,max_num_frames=8 \
            --tasks beacon3d-qa,beacon3d-grounding \
            --batch_size 1 \
            --output_path ./logs/beacon3d_quick_test \
            --limit 5
        ;;

    2)
        echo ""
        echo "Running full Beacon3D-QA evaluation..."
        echo "This may take several hours depending on your hardware"
        echo ""

        accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
            --model qwen2_5_vl \
            --model_args model_name_or_path=/data/arena/models/Qwen2_5-VL-3B-Instruct,max_num_frames=8 \
            --tasks beacon3d-qa \
            --batch_size 1 \
            --output_path ./logs/beacon3d_qa_full
        ;;

    3)
        echo ""
        echo "Running full Beacon3D-Grounding evaluation..."
        echo "Note: IoU scores will be 0 for 2D models (expected behavior)"
        echo "This may take several hours depending on your hardware"
        echo ""

        accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
            --model qwen2_5_vl \
            --model_args model_name_or_path=/data/arena/models/Qwen2_5-VL-3B-Instruct,max_num_frames=8 \
            --tasks beacon3d-grounding \
            --batch_size 1 \
            --output_path ./logs/beacon3d_grounding_full
        ;;

    4)
        echo ""
        echo "Running combined Beacon3D evaluation..."
        echo "This will evaluate both QA and Grounding tasks"
        echo "Note: Grounding IoU scores will be 0 for 2D models (expected)"
        echo ""

        read -p "Enter sample limit (press Enter for full evaluation): " limit
        if [ -z "$limit" ]; then
            limit_arg=""
        else
            limit_arg="--limit $limit"
        fi

        accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
            --model qwen2_5_vl \
            --model_args model_name_or_path=/data/arena/models/Qwen2_5-VL-3B-Instruct,max_num_frames=8 \
            --tasks beacon3d-qa,beacon3d-grounding \
            --batch_size 1 \
            --output_path ./logs/beacon3d_combined \
            $limit_arg
        ;;

    5)
        echo ""
        echo "Custom evaluation mode"
        echo ""

        read -p "Enter tasks (e.g., beacon3d-qa or beacon3d-qa,beacon3d-grounding): " tasks
        read -p "Enter batch size (default 1): " batch_size
        batch_size=${batch_size:-1}
        read -p "Enter sample limit (press Enter for all): " limit
        read -p "Enter output path (default ./logs/beacon3d_custom): " output_path
        output_path=${output_path:-./logs/beacon3d_custom}

        cmd="accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
            --model qwen2_5_vl \
            --model_args model_name_or_path=/data/arena/models/Qwen2_5-VL-3B-Instruct,max_num_frames=8 \
            --tasks $tasks \
            --batch_size $batch_size \
            --output_path $output_path"

        if [ -n "$limit" ]; then
            cmd="$cmd --limit $limit"
        fi

        echo "Running command:"
        echo "$cmd"
        echo ""

        eval $cmd
        ;;

    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Evaluation completed!"
echo "=================================================="
echo ""
echo "Results are saved in the logs directory:"
echo "- Results: Check results_*.json for metrics"
echo "- Samples: Check samples_*.json for detailed outputs"
echo ""
echo "For troubleshooting, see: Qwen2_5_VL_Troubleshooting_Guide.md"
