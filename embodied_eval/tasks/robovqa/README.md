# RoboVQA evaluation guide

## Contents

1. [Data preparation](#1-data-preparation)
2. [Running evaluation](#2-running-evaluation)
3. [Interpreting results](#3-interpreting-results)

---

## 1. Data preparation

### 1.1 Download the dataset

Layout:

RoboVQA_TF2HF (dataset root)
├── RoboVQA_TF2HF (Hugging Face copy of `koulx/RoboVQA_TF2HF`)
└── videos (video files)
   └── videos

```bash
# Download the dataset

# Create folders as above, then download koulx/RoboVQA_TF2HF
python -c "from datasets import load_dataset; load_dataset('koulx/RoboVQA_TF2HF', split='val').save_to_disk('/path/to/RoboVQA_TF2HF')"

# Download videos; --video_dir should point to RoboVQA_TF2HF/videos
python -m embodied_eval.tasks.robovqa.utils.download \
    --dataset_path /path/to/RoboVQA_TF2HF \
    --video_dir /path/to/videos \
    --auto_download
```

> **More download options**: see [utils/README.md](utils/README.md) for other download modes and data checks.

### 1.2 Configure data paths

Edit `robovqa.yaml` for dataset and video paths:

```yaml
dataset_path: /path/to/RoboVQA_TF2HF
dataset_kwargs:
  video_dir: /path/to/videos
```

After download, update paths in the YAML. You can also load online by setting `dataset_path` to `koulx/RoboVQA_TF2HF`.

---

## 2. Running evaluation

### 2.1 Using the eval script

```bash
# 1. API keys (for LLM evaluation)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='your-api-base'  # optional

# 2. Edit and run
bash embodied_eval/tasks/robovqa/run_eval.sh
```

**Parameters to set in the script:**

- `--model`: model name (e.g. `qwen3_vl`, `internvl3`, `llava_onevision`)
- `--model_args`: model path and kwargs
  - `model_name_or_path`: model path (Hugging Face id or local path)
  - `max_num_frames`: number of video frames to sample (recommended 8–32)
  - `fps`: sampling frame rate (recommended 2–4)
- `--output_path`: where to write results

### 2.2 Model loading

**Base model:**

```bash
--model_args model_name_or_path=Qwen/Qwen3-VL-4B-Instruct,max_num_frames=32,fps=2
```

**LoRA fine-tuned model:**

```bash
--model_args model_name_or_path=/path/to/lora/sft/,max_num_frames=32,fps=2
```

> Note: the LoRA folder must contain `adapter_config.json`.

---

## 3. Interpreting results

### 3.1 Output files

After evaluation, outputs are under `<output_path>/YYYYMMDD_HHMMSS/`:

```
├── inference_robovqa.json    # raw predictions
├── samples_robovqa.json      # per-sample evaluation
└── results_robovqa.json      # aggregated metrics (main file)
```

### 3.2 Main metrics

See `results_robovqa.json`:

```json
{
  "past_description:freeform_llm_match_score": 4.2,
  "planning:freeform_llm_match_score": 3.8,
  ...
  "llm_match_score_average": 4.0,   // overall score (1–5)
  "overall": 4.0
}
```

**Scale (1–5):**

- 5: perfect match
- 4: strong match
- 3: partial match
- 2: weak match
- 1: no match

**Website score (0–100 scale)**: raw scores are on 1–5; **website score = (raw − 1) / 4 × 100**

The `results_robovqa.json` also includes a convenience field `100score_overall`
that directly reports the website-scale overall score:

```json
{
  "past_description:freeform_llm_match_score": 4.2,
  "planning:freeform_llm_match_score": 3.8,
  "llm_match_score_average": 4.0,
  "overall": 4.0,
  "100score_overall": 75.0
}
```

**Key fields:**

- `llm_match_score_average`: overall mean (aligned with official RoboVQA)
- `{question_type}_llm_match_score`: mean per question type

---

## Appendix

### A. Troubleshooting

**Missing videos:**

```bash
python -m embodied_eval.tasks.robovqa.utils.download --check_only
```

**API key issues:**

```bash
export OPENAI_API_KEY='your-api-key'
```

**GPU OOM**: lower `max_num_frames` (e.g. to 8)

### B. Links

- **RoboVQA (official)**: https://github.com/google-deepmind/robovqa
- **Dataset**: https://huggingface.co/datasets/koulx/RoboVQA_TF2HF
- **Utils documentation**: [utils/README.md](utils/README.md)
