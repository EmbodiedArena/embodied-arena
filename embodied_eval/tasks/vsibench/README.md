# VSI-Bench evaluation guide

## Contents

1. [Data preparation](#1-data-preparation)
2. [Running evaluation](#2-running-evaluation)
3. [Reading results](#3-reading-results)

---

## 1. Data preparation

### 1.1 Download the dataset

```bash
python -m embodied_eval.tasks.vsibench.utils.download --local_dir /path/to/prime-vsi-bench-data
```

The download typically includes: (1) `scannet.zip` (ScanNet videos), (2) `scannetpp.zip` (ScanNet++ videos), (3) `arkitscenes.zip` (ARKitScenes videos), and (4) `test.jsonl` (test questions and answers).

### 1.2 Preprocess

```bash
# Full pipeline (extract videos + convert to evaluation format)
python -m embodied_eval.tasks.vsibench.utils.preprocess \
    --raw_dir /path/to/prime-vsi-bench-data \
    --output_dir /path/to/vsi-bench-data

# Extract videos only
python -m embodied_eval.tasks.vsibench.utils.preprocess \
    --raw_dir /path/to/prime-vsi-bench-data \
    --output_dir /path/to/vsi-bench-data \
    --extract_only

# Convert JSONL to Hugging Face Dataset only
python -m embodied_eval.tasks.vsibench.utils.preprocess \
    --raw_dir /path/to/prime-vsi-bench-data \
    --output_dir /path/to/vsi-bench-data \
    --convert_only
```

After extraction, the layout should look like:

```text
output_dir/
├── test/                    # Hugging Face Dataset
│   ├── dataset_info.json
│   └── ...
├── scannet/                 # videos
│   ├── scene0015_00.mp4
│   └── ...
├── scannetpp/               # videos
│   └── ...
└── arkitscenes/             # videos
    └── ...
```

### 1.3 Point the task at your data

Edit `vsibench.yaml`:

```yaml
dataset_path: /path/to/vsi-bench-data
dataset_kwargs:
  video_dir: /path/to/vsi-bench-data
```

---

## 2. Running evaluation

### 2.1 Evaluation script

```bash
# 1. API keys for LLM-based judging
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='your-api-base'  # optional

# 2. Edit and run
bash embodied_eval/tasks/vsibench/run_eval.sh
```

**Typical edits inside the script:**

- `--model`: registry name (e.g. `qwen3_vl`, `internvl3`, `llava_onevision`)
- `--model_args`: checkpoint and runtime knobs
  - `model_name_or_path`: Hugging Face model id or local path
  - `max_num_frames`: frames sampled from video (often 8–32)
  - `fps`: sampling rate (often 2–4)
- `--output_path`: where logs are written

### 2.2 Loading base vs. LoRA checkpoints

**Base model:**

```bash
--model_args model_name_or_path=OpenGVLab/InternVL3_5-8B,max_num_frames=32,fps=2
```

**LoRA / SFT adapter:**

```bash
--model_args model_name_or_path=/path/to/lora/sft/,max_num_frames=32,fps=2
```

The LoRA folder must include `adapter_config.json`.

---

## 3. Reading results

### 3.1 Output files

Each run writes under `<output_path>/YYYYMMDD_HHMMSS/`:

```text
├── configs_vsibench.json    # run configuration
├── samples_vsibench.json    # per-sample details
└── results_vsibench.json    # aggregated metrics (main file)
```

### 3.2 Metrics

Example `results_vsibench.json`:

```json
{
    "obj_appearance_order_accuracy": 0.5760517799352751,
    "object_abs_distance_MRA:.5:.95:.05": 0.4226618705035971,
    "object_counting_MRA:.5:.95:.05": 0.6989380530973451,
    "object_rel_distance_accuracy": 0.5535211267605634,
    "object_size_estimation_MRA:.5:.95:.05": 0.6719832109129066,
    "room_size_estimation_MRA:.5:.95:.05": 0.6125,
    "route_planning_accuracy": 0.34536082474226804,
    "object_rel_direction_accuracy": 0.49670450401229616,
    "accuracy_average": 0.4929095588626007,
    "MRA:.5:.95:.05_average": 0.6015207836284622,
    "overall": 0.5472151712455315
}
```

**Scoring:** VSI-Bench mixes counting items and multiple-choice items. Counting uses **accuracy** in \[0, 1\]; choice-style items use **mean relative accuracy (MRA)** in \[0, 1\] (see the original paper for definitions). `overall` combines both families, also in \[0, 1\].

**Website scale (0–100):** **website score = raw score × 100**.

---

## Appendix

### A. Troubleshooting

**OOM / not enough GPU memory:** If you have multiple GPUs, set `CUDA_VISIBLE_DEVICES` in `run_eval.sh` and add `device_map="auto"` inside `--model_args`. Otherwise lower `max_num_frames` and `batch_size`.

**Too slow:** With several GPUs, raise `--num_processes` to match GPU count for parallel runs (some model stacks may not support this). You can also increase `batch_size`, lower `max_num_frames`, or shard the dataset and evaluate in chunks.

**Model download issues:** Avoid `git clone` for model weights (e.g. InternVL3.5-8B); prefer `huggingface-cli download`.

When comparing models on the same benchmark, keep hyperparameters (e.g. `max_num_frames`) consistent—do not tune them per model only for speed.

### B. Links

- **Official VSI-Bench:** [vision-x-nyu/thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space)
- **Dataset:** [nyu-visionx/VSI-Bench on Hugging Face](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)

### C. Dataset splits

Besides `vsibench.yaml` (full split), other YAML files in this folder configure **subsets** of VSI-Bench. You normally do not need them; use them only when full evaluation is too slow and you want a smaller slice for testing.
