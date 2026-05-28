# Where2Place evaluation guide

## Contents

- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Examples](#examples)
- [Advanced usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Outputs and scoring](#outputs-and-scoring)
- [Utils scripts](#utils-scripts)
- [Further reading](#further-reading)

---

## Quick start

### 0. Data

- **Source:** Hugging Face `FlagEval/Where2Place`
- **Task:** 100 real-world images for free-space referral via spatial relations, collected from cluttered scenes. Each image has a sentence describing usable space and a mask for the target region.
- **Output format:** Different models use different point/bbox conventions; adjust the expected output format in the task configuration so parsing matches your model.

### 1. Shell script (recommended)

```bash
# Task directory
cd ./embodied-arena/embodied_eval/tasks/where2place

# Edit the script and set the model path
nano run_eval.sh
# Set MODEL_PATH to your checkpoint

# Run
bash utils/run_eval.sh
```

### 2. Manual CLI

```bash
# Repo root
cd embodied-arena

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --main_process_port=29500 \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B,max_num_frames=32,fps=2 \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --output_path ./logs/where2place/pelican_vl_7b_point
```

### 3. Python API

```python
from embodied_eval.models.pelican_vl import PelicanVL

model = PelicanVL(
    model_name_or_path="X-Humanoid/Pelican1.0-VL-7B",
    device="cuda",
    device_map="cuda",
    max_num_frames=32,
    fps=2,
    batch_size=1,
)

context = "Where can I place the object?"
visuals = ["path/to/image.jpg"]
response = model.respond(context, visuals)
print(response)
```

---

## CLI reference

### Core flags

| Flag | Description | Example |
|------|-------------|---------|
| `--model` | Registered model name | `pelican_vl` |
| `--model_args` | Comma-separated kwargs | `model_name_or_path=...,fps=2` |
| `--evaluator` | Evaluator backend | `eqa` (QA) or `nav` (navigation) |
| `--tasks` | Task id | `where2place-point` or `where2place-bbox` |
| `--batch_size` | Batch size | `1` |
| `--output_path` | Log directory | `./logs/where2place/results` |

### `model_args` fields

Pass as a single comma-separated string (no spaces after commas):

```bash
--model_args model_name_or_path=/path/to/model,\
max_num_frames=32,\
fps=2,\
max_new_tokens=1024,\
temperature=0,\
do_sample=false,\
top_p=1.0,\
num_beams=1,\
use_flash_attention_2=false,\
min_pixels=3126,\
max_pixels=200704,\
batch_size=1
```

| Field | Meaning |
|-------|---------|
| `model_name_or_path` | Hugging Face id or local path |
| `max_num_frames` | Max video frames (default 32) |
| `fps` | Video sampling fps (default 2) |
| `max_new_tokens` | Max new tokens (default 1024) |
| `temperature` | Sampling temperature (0 = greedy) |
| `do_sample` | Enable sampling (default false) |
| `top_p` | Nucleus sampling p (default 1.0) |
| `num_beams` | Beam width (default 1) |
| `use_flash_attention_2` | FlashAttention 2 (default false) |
| `min_pixels` | Min image pixels (default 3126) |
| `max_pixels` | Max image pixels (default 200704) |

### Other flags

| Flag | Description | Default |
|------|-------------|---------|
| `--limit` | Cap number of samples | `None` (all) |
| `--inference_only` | Run inference without metrics | `False` |
| `--save_results` | Persist outputs | `True` |
| `--seed` | RNG seed | `42` |
| `--verbosity` | Log level | `DEBUG` |
| `--debug` | Debug mode | `False` |

---

## Examples

### Example 1: Basic run

```bash
cd embodied-arena

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --main_process_port=29500 \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --output_path ./logs/test
```

### Example 2: Local checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --main_process_port=29500 \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=/data/models/Pelican1.0-VL-7B,max_num_frames=32,fps=2 \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --output_path ./logs/where2place/pelican_local
```

### Example 3: Inference only

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --main_process_port=29500 \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --inference_only \
    --output_path ./logs/inference_only
```

### Example 4: Subset for smoke tests

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --main_process_port=29500 \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --limit 10 \
    --output_path ./logs/quick_test
```

### Example 5: Multi-GPU data parallel

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes=4 \
    --main_process_port=29500 \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --output_path ./logs/multi_gpu
```

### Example 6: Flash Attention 2 (if supported)

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --main_process_port=29500 \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B,use_flash_attention_2=true \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --output_path ./logs/flash_attn
```

---

## Advanced usage

### 1. Batch multiple tasks

`run_all_tasks.sh`:

```bash
#!/bin/bash

TASKS=("where2place-point" "where2place-bbox")
MODEL_PATH="X-Humanoid/Pelican1.0-VL-7B"

for task in "${TASKS[@]}"; do
    echo "Running task: $task"
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --num_processes=1 \
        --main_process_port=29500 \
        -m embodied_eval \
        --model pelican_vl \
        --model_args model_name_or_path=$MODEL_PATH \
        --evaluator eqa \
        --tasks $task \
        --batch_size 1 \
        --output_path ./logs/$task
done
```

### 2. Debug mode

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --main_process_port=29500 \
    -m embodied_eval \
    --model pelican_vl \
    --model_args model_name_or_path=X-Humanoid/Pelican1.0-VL-7B \
    --evaluator eqa \
    --tasks where2place-point \
    --batch_size 1 \
    --debug \
    --verbosity DEBUG \
    --output_path ./logs/debug
```

### 3. List registered tasks

```python
from embodied_eval.common.registry import TASK_REGISTRY
print(TASK_REGISTRY.keys())
```

---

## Troubleshooting

### 1. Model not found

**Error:** `OSError: X-Humanoid/Pelican1.0-VL-7B does not appear to be a model`

**Fix:**

1. Check network access to Hugging Face.
2. Use a local path: `model_name_or_path=/path/to/local/model`
3. Pre-download:

   ```bash
   huggingface-cli download X-Humanoid/Pelican1.0-VL-7B
   ```

### 2. CUDA OOM

**Error:** `RuntimeError: CUDA out of memory`

**Fix:**

1. Lower `max_num_frames`, e.g. `max_num_frames=16`
2. Lower `max_pixels`, e.g. `max_pixels=100352`
3. Use a smaller checkpoint, e.g. `Pelican1.0-VL-3B`

### 3. Port already in use

**Error:** `Address already in use`

**Fix:**

Pick a free port:

```bash
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
accelerate launch --main_process_port=$PORT ...
```

### 4. `qwen_vl_utils` import error

**Error:** `Failed to import qwen_vl_utils`

**Fix:**

```bash
pip install qwen-vl-utils
```

### 5. Permission denied

**Error:** `Permission denied`

**Fix:**

```bash
chmod +x run_pelican_vl.sh
```

---

## Outputs and scoring

Artifacts under `--output_path`:

```text
logs/where2place/pelican_vl_7b_point/
├── results.json           # metrics
├── inference_results.json # raw generations
└── config.json            # run config
```

**Metric:** Where2Place scores spatial alignment between predicted points/boxes and the reference mask (IoU or hit criteria) as **reference_acc** in \[0, 1\].

**Website scale (0–100):** **website score = reference_acc × 100**.

### Inspect results

```python
import json

with open("logs/where2place/pelican_vl_7b_point/results.json", "r") as f:
    results = json.load(f)

print(f"reference_acc: {results['reference_acc']:.2%}")
```

---

## Utils scripts

Run from the **repository root**:

| Script | Purpose |
|--------|---------|
| `quick_test.sh` | Fast check on 5 samples |
| `recompute_accuracy.py` | Recompute accuracy for percentage-style coordinates |
| `reverse_engineer_scaling.py` | Search coordinate scaling that best matches labels |
| `analyze_alternative_strategies.py` | Compare strategies (run `reverse_engineer_scaling.py --all` first) |

```bash
cd embodied-arena
bash embodied_eval/tasks/where2place/utils/quick_test.sh
python embodied_eval/tasks/where2place/utils/reverse_engineer_scaling.py --all
```

---

## Further reading

- [`embodied_eval` overview](../../README.md)
- [Project README](../../../README.md)
- [Developer guide](../../../DEVELOPER_GUIDE.md)
- [PelicanVL on Hugging Face](https://huggingface.co/collections/X-Humanoid/pelican-vl-10)
- [Issues / feedback](https://github.com/lichaozhy/tjucs-airank-2025)
