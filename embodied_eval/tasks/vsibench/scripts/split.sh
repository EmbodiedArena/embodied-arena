#!/bin/bash
set -e

# =========================
# 配置
# =========================
export DATASET=/your/path/to/embodied-eval-main/embodied_eval/data/vsi-bench-data/test
export OUT_ROOT=/your/path/to/embodied-eval-main/embodied_eval/data/vsi-bench-data/vsibench_hf_splits8
export SPLITS=8

# =========================
# 创建输出目录
# =========================
mkdir -p "$OUT_ROOT"

# =========================
# 运行 Python 分割数据集
# =========================
python3 - <<EOF
from datasets import load_from_disk, DatasetDict
import os

DATASET = "$DATASET"
OUT_ROOT = "$OUT_ROOT"
SPLITS = $SPLITS

# 直接加载 Dataset（不是 DatasetDict）
ds = load_from_disk(DATASET)
N = len(ds)
chunk = (N + SPLITS - 1) // SPLITS

for i in range(SPLITS):
    start = i * chunk
    end = min((i + 1) * chunk, N)
    sub = ds.select(range(start, end))
    out = os.path.join(OUT_ROOT, f"part_{i}")
    os.makedirs(out, exist_ok=True)
    DatasetDict({"test": sub}).save_to_disk(out)
    print(f"Saved part {i}: [{start}, {end})")
EOF