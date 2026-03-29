#!/bin/bash
# Official Beacon3D Evaluation Script for Qwen2.5-VL
# Uses official ChatGPT LLM evaluation to match benchmark results exactly

set -e  # Exit on any error

echo "=========================================="
echo "Official Beacon3D Evaluation for Qwen2.5-VL"
echo "=========================================="

# Check if API key is provided
if [ -z "$OPENAI_API_KEY" ] && [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "❌ ERROR: Please set your OpenAI API key!"
    echo "   Option 1: export OPENAI_API_KEY='your-key-here'"
    echo "   Option 2: export AZURE_OPENAI_API_KEY='your-key-here'"
    echo "   Option 3: ./evaluate_beacon3d_official_qwen.sh your-api-key-here"
    exit 1
fi

# If API key provided as argument, use it
if [ $# -eq 1 ]; then
    export OPENAI_API_KEY="$1"
    echo "Using provided API key"
fi

# Environment setup
cd /home/arena/embodiedeval/embodied-eval-main
conda activate embodied-eval

# Set other environment variables
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}"

# Get random port
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Using port: $PORT"

# Create output directory
OUTPUT_DIR="./logs/official_beacon3d_qwen_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo "Starting evaluation..."

# Run evaluation
accelerate launch --num_processes=1 --main_process_port="$PORT" \
    -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=/data/arena/models/Qwen2_5-VL-3B-Instruct,max_num_frames=8 \
    --evaluator official_beacon3d \
    --tasks beacon3d-qa \
    --batch_size 1 \
    --output_path "$OUTPUT_DIR" \
    --limit 10  # Remove this line to evaluate all samples

echo ""
echo "=========================================="
echo "✅ Official Beacon3D evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key results:"
if [ -f "$OUTPUT_DIR/results_beacon3d-qa.json" ]; then
    echo "📊 Final Scores:"
    python -c "
import json
with open('$OUTPUT_DIR/results_beacon3d-qa.json') as f:
    results = json.load(f)
    em_scores = [item['em'] for item in results]
    gpt_scores = [item['score'] for item in results]
    em_mean = sum(em_scores) / len(em_scores) * 100
    gpt_mean = (sum(gpt_scores) / len(gpt_scores) - 1) / 4 * 100
    print(f'  EM Score: {em_mean:.2f}%')
    print(f'  GPT Score: {gpt_mean:.2f}%')
"
else
    echo "❌ Results file not found"
fi
echo "=========================================="
