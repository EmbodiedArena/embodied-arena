# EmbSpatial-Bench evaluation guide

## Contents

1. [Data preparation](#1-data-preparation)
2. [Running evaluation](#2-running-evaluation)
3. [Interpreting results](#3-interpreting-results)

---

## 1. Data preparation

### 1.1 Download the dataset

The EmbSpatial-Bench dataset can be downloaded from Hugging Face.

EmbSpatial-Bench is not released directly in a ready-to-use form; **Phineas476/EmbSpatial-Bench** is a processed copy that serves as a usable EmbSpatial-Bench dataset source.

```bash
# Option 1: download with the Hugging Face datasets library
python -c "from datasets import load_dataset; ds = load_dataset('Phineas476/EmbSpatial-Bench'); ds.save_to_disk('/path/to/embSpatial_Bench')"

# Option 2: if you already have a local copy (Hugging Face layout)
# Expected layout:
# embSpatial_Bench/
#   ├── dataset_info.json
#   ├── README.md
#   └── data/
#       └── test-00000-of-00001.parquet
```

Dataset format:

- Hugging Face dataset with a `test` split
- Each sample has: `question`, `answer_options`, `answer` (index), `relation`, `image` (PIL Image), `image_base64`
- Six spatial relations: `close`, `under`, `right`, `left`, `above`, `behind`
- 3,640 QA pairs in total

### 1.2 Configure data paths

Edit `emspatial-bench.yaml`. Two options:

**Option 1: local dataset on disk** (recommended if already downloaded)

```yaml
dataset_path: /your/path/to/embodied-eval-main/embodied_eval/data/embSpatial_Bench
load_from_disk: false  # the framework auto-detects and loads
eval_split: test
```

**Option 2: Hugging Face online load**

```yaml
dataset_path: Phineas476/EmbSpatial-Bench
load_from_disk: false
eval_split: test
```

**Notes:**

- If the local folder contains `dataset_info.json`, you can set `load_from_disk: true`
- If the local data is only parquet files, use `load_from_disk: false`; the framework handles it
- `dataset_path` should point to the parent directory that contains the `data/` folder

---

## 2. Running evaluation

### 2.1 Using an eval script

Create and run a script, or invoke from the CLI:

```bash
# Create a script (see other tasks’ scripts/ folders)
# or run directly:

python -m embodied_eval \
    --model <model_name> \
    --model_args <model_args> \
    --tasks emspatial-bench \
    --output_path logs/emspatial-bench
```

**Example script** (`run_eval.sh`):

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct \
    --tasks emspatial-bench \
    --output_path logs/emspatial-bench/qwen2_5_vl \
    --batch_size 1 \
    --num_fewshot 0
```

### 2.2 Model arguments

**Remote / HF model:**

```bash
--model_args model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct
```

**Local checkpoint:**

```bash
--model_args model_name_or_path=/path/to/local/model
```

---

## 3. Interpreting results

### 3.1 Output files

After evaluation, outputs are under `<output_path>/YYYYMMDD_HHMMSS/`:

```
├── inference_emspatial-bench.json    # raw predictions
├── samples_emspatial-bench.json     # per-sample evaluation
└── results_emspatial-bench.json     # aggregated metrics (main file)
```

### 3.2 Main metrics

See `results_emspatial-bench.json`:

```json
{
  "close_accuracy": 0.85,
  "under_accuracy": 0.78,
  "right_accuracy": 0.82,
  "left_accuracy": 0.80,
  "above_accuracy": 0.79,
  "behind_accuracy": 0.76,
  "accuracy_average": 0.80,
  "overall": 0.80
}
```

**Metric definitions:**

- `{relation}_accuracy`: accuracy per spatial relation type (0–1)
- `accuracy_average`: mean accuracy over relations
- `overall`: overall accuracy (same as `accuracy_average` here)

**Website score (0–100 scale)**: raw scores are in 0–1; **website score = raw score × 100**

**Relation types:**

- `close`: distance (nearest / farthest)
- `under`: vertical (below)
- `right`: left–right (to the right)
- `left`: left–right (to the left)
- `above`: vertical (above)
- `behind`: front–back (behind)

### 3.3 Detailed results

Inspect `samples_emspatial-bench.json` (JSONL) for per-sample details:

```json
{
  "question_id": "mp3d_57",
  "question": "From your perspective, which object in the image is at the shortest distance?",
  "relation": "close",
  "answer_options": ["garage door", "cabinet", "table", "cart"],
  "answer": 2,
  "target": "table",
  "prediction": "C. table",
  "accuracy": {"accuracy": 1.0}
}
```

---

## Appendix

### A. Troubleshooting

**Wrong dataset path:**

- Check `dataset_path` in `emspatial-bench.yaml`
- Ensure JSON/JSONL layout is valid

**Image decode failures:**

- Verify base64 encoding
- Ensure image payloads are complete

**Prediction matching failures:**

- Model outputs may omit explicit option letters (A/B/C/D)
- The system tries several matching strategies but may still fail
- Prompt the model to answer with the option letter when possible

### B. Re-scoring saved runs

```bash
python -m embodied_eval.tasks.emspatial_bench.process \
    --sample_file logs/emspatial-bench/model/YYYYMMDD_HHMMSS/samples_emspatial-bench.json \
    --results_file logs/emspatial-bench/model/YYYYMMDD_HHMMSS/results_emspatial-bench.json
```

Or in Python:

```python
from embodied_eval.tasks.emspatial_bench.process import post_evaluate_results

post_evaluate_results(
    sample_file_path="logs/emspatial-bench/model/YYYYMMDD_HHMMSS/samples_emspatial-bench.json",
    results_file_path="logs/emspatial-bench/model/YYYYMMDD_HHMMSS/results_emspatial-bench.json"
)
```

### C. Links

- **EmbSpatial-Bench (official)**: https://github.com/mengfeidu/EmbSpatial-Bench
- **Dataset access**: email `mfdu22@m.fudan.edu.cn`
- **Paper**: see the official repository for the paper link

### D. Extra utilities

- Under `utils/`, post-evaluation scripts can re-score saved runs: **samples**-based paths re-evaluate from `samples`; **inference**-based paths build results from inference-only JSON.

- `refill_empty_resps` fills in responses that were missing due to network issues; run it together with the post-eval flow to re-score.
