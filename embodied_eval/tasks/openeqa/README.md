# openEQA evaluation guide

## Contents

1. [Data preparation](#1-data-preparation)
2. [Running evaluation](#2-running-evaluation)
3. [Interpreting results](#3-interpreting-results)

---

## 1. Data preparation

### 1.1 Dataset

- Uses the openEQA **test** split. The dataset is not released directly elsewhere; download from Hugging Face: [ellisbrown/OpenEQA](https://huggingface.co/datasets/ellisbrown/OpenEQA)

- 1,636 items, mostly image-based QA

- Answers are scored with an LLM

### 1.2 Download the dataset

```bash
# Download the dataset
cd ./embodied-eval-main/embodied_eval/data/open-eqa

huggingface-cli download --repo-type dataset --resume-download ellisbrown/OpenEQA --local-dir ./embodied-eval-main/embodied_eval/data/open-eqa
```

Unpack archives so the layout looks like:

./open-eqa

./open-eqa/v0

./open-eqa/hm3d-v0

./open-eqa/scannet-v0

### 1.3 Configure data paths

Edit `openeqa-emeqa.yaml` for dataset and video paths:

```yaml
dataset_path: /path/to/open-eqa
dataset_kwargs:
  video_dir: /path/to/open-eqa
```

---

## 2. Running evaluation

### 2.1 Using the eval scripts

```bash
# 1. API keys (for LLM evaluation)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='your-api-base'

# 2. Edit and run a script
bash embodied_eval/tasks/openeqa/scripts/*.sh
```

**Parameters to set in the scripts:**

- `--model`: model name (e.g. `qwen3_vl`, `internvl3`, `llava_onevision`)
- `--model_args`: model path and kwargs
  - `model_name_or_path`: model path (Hugging Face id or local path)
  - `max_num_frames`: number of video frames to sample (recommended 8–32)
- `--output_path`: where to write results
- `--limit`: cap the number of samples in one run
- `--inference_only`: run inference only (no metric pass)

Before a full run, you can use `--limit 10` to sanity-check the pipeline.

**Post-processing**

- Post-processing utilities live under `utils/`.

- `post_eval.sh` re-runs evaluation for runs that finished inference via `bash embodied_eval/tasks/openeqa/scripts/*.sh` but did not complete scoring (e.g. misconfiguration).

- In `post_eval.sh`, set `BASE_DIR` to the log directory that needs re-scoring; other options mirror the CLI args accepted by `process.py`.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Configure your OpenAI-compatible gateway and key here
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'  # must include /v1 suffix; /v1/ is equivalent

# Log directory to re-score (should contain samples_openeqa-emeqa.json / results_openeqa-emeqa.json)
BASE_DIR="/your/path/to/embodied-eval-main/logs/openeqa/qwen2_5-vl-7b-instruct/20260125_211253/"

python -m embodied_eval.tasks.openeqa.process \
  --base_dir "${BASE_DIR}" \
  --openai_model "gpt-4o" \
  --openai_temperature 0.2 
```

### 2.2 Model loading

**Base model example:**

```bash
--model_args model_name_or_path=Qwen/Qwen3-VL-8B-Instruct,max_num_frames=8
```

---

## 3. Interpreting results

### 3.1 Output files

After evaluation, outputs are under `<output_path>/YYYYMMDD_HHMMSS/`:

```
├── configs_openeqa-emeqa.json    # configuration
├── results_openeqa-emeqa.json      # aggregated metrics
└── samples_openeqa-emeqa.json      # per-sample records and scores
```

With `--inference_only`, only `inference_openeqa-emeqa.json` is produced.

Scoring from the inference-only file is not supported yet, so avoid `--inference_only` if you need metrics.

### 3.2 Main metrics

See `results_openeqa-emeqa.json`.

Example (qwen3-vl):

```json
{
  "attribute recognition_llm_match_score": 3.9875,
  "functional reasoning_llm_match_score": 3.3548387096774195,
  "object localization_llm_match_score": 3.03041825095057,
  "object recognition_llm_match_score": 3.264069264069264,
  "object state recognition_llm_match_score": 3.7142857142857144,
  "spatial understanding_llm_match_score": 2.9863636363636363,
  "world knowledge_llm_match_score": 3.276995305164319,
  "score_per_type_average": 3.3734958400729886,
  "llm_match_score_all_samples_average": 3.378361858190709,
  "overall": 3.3740365087527353
}
```

The first seven entries are scores for different dataset categories. `score_per_type_average` is the mean over all examples. `llm_match_score_all_samples_average` is the mean computed by capability grouping. `overall` is the average of the numeric values present in the results file.

**Scale (1–5):**

- 5: perfect match
- 4: strong match
- 3: partial match
- 2: weak match
- 1: no match

**Website score (0–100 scale)**: raw scores are on 1–5; **website score = (raw − 1) / 4 × 100**

---

## Appendix

### A. Troubleshooting

**API key issues:**

```bash
export OPENAI_API_KEY='your-api-key'
```

**GPU OOM**: lower `max_num_frames` (e.g. to 8)

**Cannot reach the LLM for scoring:**

Predictions are stored under the log directory. After inference finishes, re-run scoring with `embodied_eval/tasks/openeqa/utils/post_eval.sh`.

### B. Links

- **Dataset**: https://huggingface.co/datasets/ellisbrown/OpenEQA
- Note: use the **test** split of this dataset.
