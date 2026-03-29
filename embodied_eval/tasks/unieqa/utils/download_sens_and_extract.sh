#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 从 utils 目录往上 4 级到项目根目录
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# 设置路径
UNIEQA_DATA_DIR="$PROJECT_ROOT/embodied_eval/data/unieqa/111/UniEQA/data"
OUTPUT_DIR="$PROJECT_ROOT/embodied_eval/data/unieqa/111/UniEQA/data/Part1/images/scannet-v0"

# 验证路径
if [ ! -d "$UNIEQA_DATA_DIR" ]; then
    echo "❌ 错误: UniEQA 数据目录不存在: $UNIEQA_DATA_DIR"
    exit 1
fi

echo "📁 项目根目录: $PROJECT_ROOT"
echo "📁 UniEQA 数据目录: $UNIEQA_DATA_DIR"
echo "📁 输出目录: $OUTPUT_DIR"
echo ""
echo "🚀 步骤 1/2: 检查并下载缺失的 ScanNet .sens 文件..."
cd "$SCRIPT_DIR"
if ! python "$SCRIPT_DIR/download_needed_scenes.py" "$SCRIPT_DIR" "$UNIEQA_DATA_DIR"; then
    echo "❌ 下载步骤失败，请检查错误信息"
    exit 1
fi

echo ""
echo "🎬 步骤 2/2: 提取需要的场景图片帧..."
if ! python "$SCRIPT_DIR/extract_specific_scenes.py" \
    --scannet-root "$SCRIPT_DIR" \
    --output-directory "$OUTPUT_DIR" \
    --unieqa-data-dir "$UNIEQA_DATA_DIR" \
    --rgb-only \
    --max-num-frames 600; then
    echo "❌ 提取步骤失败，请检查错误信息"
    exit 1
fi

echo ""
echo "✅ 全部完成！RGB 帧已提取到: $OUTPUT_DIR"
