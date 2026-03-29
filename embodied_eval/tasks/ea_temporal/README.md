# EA-Temporal evaluation guide

## Contents

1. [Data preparation](#1-data-preparation)
2. [Running evaluation](#2-running-evaluation)
3. [Interpreting results](#3-interpreting-results)

---

## 1. Data preparation

### 1.1 Download the dataset

The dataset is on Hugging Face and can be downloaded directly: EmbodiedArena/EA-Temporal

Directory layout:

```
output_dir/
├── dataset_config.yaml
├── ea_temporal_merged.json
├── ea_temporal_type1.json
├── ea_temporal_type2.json
├── images
│   ├── task_type_1_high
│   ├── task_type_1_low
│   └── task_type_2_high
├── temporal_1024.zip
├── type1_1021.json
└── type2_1024.json
```

EA-Temporal evaluates VLM temporal perception with **two** task types:

- **Type1 (Action Recognition)**: recognize the robot action from consecutive images
- **Type2 (Chronological Ordering)**: order images in the correct temporal sequence

### 1.2 Configure data paths

Edit `ea-temporal.yaml` for the full dataset and paths. `ea-temporal-type1.yaml` and `ea-temporal-type2.yaml` configure Type1 and Type2 separately. Example for `ea-temporal.yaml`:

```yaml
task: ea-temporal
dataset_path: /path/to/ea_temporal_merged.json
dataset_kwargs:
  image_dir: /path/to/EA-Temporal
```

To configure Type1 and Type2 alone, change `task` (`ea-temporal-type1` / `ea-temporal-type2`) and `dataset_path` (`ea_temporal_type1.json` / `ea_temporal_type2.json`) accordingly.

---

## 2. Running evaluation

### 2.1 Using the eval script

```bash
# 1. API keys (for LLM-based evaluation)
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='your-api-base'  # optional

# 2. Edit and run the script
bash /path/to/run_eval.sh
```

**Parameters to set in the script:**

- `--model`: model name (e.g. `qwen3_vl`, `internvl3`)
- `--model_args`: model path and kwargs
  - `model_name_or_path`: model path (Hugging Face id or local path)
  - `max_num_frames`: number of video frames to sample (recommended 8–32)
  - `fps`: sampling frame rate (recommended 2–4)
- `--tasks`: must match the `task` field in the YAML for the split you are evaluating
- `--output_path`: where to write results

### 2.2 Model loading

```bash
--model_args model_name_or_path=/path/to/model,max_num_frames=32,fps=2
```

---

## 3. Interpreting results

### 3.1 Output files

After evaluation, outputs are under `<output_path>/YYYYMMDD_HHMMSS/`:

```
├── configs_ea-temporal.json    # run / experiment settings
├── samples_ea-temporal.json    # per-sample details
└── results_ea-temporal.json    # aggregated metrics (main file)
```

### 3.2 Main metrics

See `results_ea-temporal.json`:

```json
{
    "type1_llm_match_score": 0.6182336182336182,
    "type2_llm_match_score": 0.3387096774193548,
    "score_average": 0.47847164782648655,
    "overall": 0.47847164782648655
}
```

**Scoring:**

- **Type1**: LLM-based semantic similarity score, range 1–5 (5 = perfect, 1 = no match)
- **Type2**: LLM-based exact sequence-match accuracy, range 0–1
- **overall**: weighted or averaged combination of Type1 and Type2

**Website score (0–100 scale)**: `type1_llm_match_score`, `type2_llm_match_score`, and `overall` in the result file are normalized to 0–1; **website score = raw score × 100**

---

## Appendix

### A. Troubleshooting

**Out of GPU memory**: If you have multiple GPUs, select them in the launch script and set `device_map="auto"` in `--model_args`; otherwise try lowering `max_num_frames` and `batch_size`.

**Slow runs**: With multiple GPUs, increase `--num_processes` (set to the number of GPUs) for multi-process runs (some models may not support this); or raise `batch_size` and lower `max_num_frames`; or split the dataset and run shards separately.

(Note: when comparing models on the same benchmark, keep hyperparameters consistent—do not change only one model’s `max_num_frames` etc. for speed.)
