# RoboVQA utils

This directory contains helper scripts for RoboVQA.

## Files

### `download.py`

Dataset download and preparation (standardized interface).

**Features:**

- Check for missing video files
- Automatically download missing videos (supports `gsutil` and the Python GCS client)
- Verify dataset integrity

**CLI:**

```bash
# Check for missing videos
python -m embodied_eval.tasks.robovqa.utils.download \
    --dataset_path /path/to/RoboVQA_TF2HF \
    --video_dir /path/to/videos \
    --check_only

# Auto-download missing videos
python -m embodied_eval.tasks.robovqa.utils.download \
    --dataset_path /path/to/RoboVQA_TF2HF \
    --video_dir /path/to/videos \
    --auto_download
```

**Python API:**

```python
from embodied_eval.tasks.robovqa.utils.download import prepare_dataset, download_robovqa_videos

# Prepare dataset (check and download as needed)
success = prepare_dataset(
    dataset_path="/path/to/RoboVQA_TF2HF",
    video_dir="/path/to/videos",
    auto_download=True
)
```

### `preprocess.py`

One-off data preprocessing script.

**Purpose:** Convert raw RoboVQA JSON into Hugging Face `Dataset` format.

**CLI:**

```bash
# Run from repo root
cd /path/to/embodied-eval-main
python embodied_eval/tasks/robovqa/utils/preprocess.py
```

**Features:**

- Parse `<task:...>` tags in the raw data
- Extract questions, answers, and task types
- Convert rows into a standard Hugging Face `Dataset`
- Optional upload to the Hugging Face Hub

**Notes:**

- This is a **one-time data preparation** tool
- Use it only when you need to regenerate processed data from raw JSON
- The evaluation stack expects the **already processed** Hugging Face dataset; you do **not** need this script for normal evaluation

## General notes

- These scripts are for data prep and helpers
- Prefer the standardized flow described in the parent [README.md](../README.md)
- If paths misbehave, run commands from the repository root
