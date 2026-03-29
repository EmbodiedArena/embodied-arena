## VABench-P point-selection evaluation

### 1. Data

- **Source:** Hugging Face `IffYuan/VABench-P` (wired as `dataset_path` in `vabench.yaml`).
- **Task:** One image + a text instruction; the model must output **8 normalized points** in the range 0–1000.
- **Output format** (must match exactly):

```text
<Answer><point>[[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8]]</point></Answer>
```

The evaluator parses this tag, reads the eight points, maps them back to image space, and computes accuracy.

### 2. Configuration

`vabench.yaml` is ready to use. Key fields:

```yaml
task: vabench
dataset_path: IffYuan/VABench-P
eval_split: test
doc_to_visual: !function process.vabench_doc_to_visual
doc_to_text: !function process.vabench_doc_to_text
process_results: !function process.vabench_process_results
metric_kwargs:
  metric: accuracy
  aggregation: !function process.vabench_aggregate_results
```

To change prompts or decoding, edit `dataset_kwargs` or `generation_kwargs`.

### 3. Running evaluation

#### 3.1 CLI (from repo root)

Example uses `qwen2_5_vl`; swap model and args as needed:

```bash
python -m embodied_eval \
  --model qwen2_5_vl \
  --model_args model_name_or_path=/path/to/your/qwen_vl,max_num_frames=1,use_flash_attention_2=False \
  --evaluator eqa \
  --tasks vabench \
  --batch_size 1 \
  --output_path ./logs/vabench/qwen2_5_vl
```

Notes:

- `--tasks vabench` selects this benchmark.
- `--evaluator eqa` reuses the shared EQA-style evaluator.
- `--model` / `--model_args`: see the matching file under `embodied_eval/models/`.

#### 3.2 Shell scripts (RoboVQA-style)

Provided examples:

- `scripts/iflybot_vlm.sh` — run VABench-P with `iflybot_vlm`.
- `run_eval.sh` — generic template; adjust `--model` and `--model_args`.

From repo root:

```bash
bash embodied_eval/tasks/vabench/run_eval.sh
```

or:

```bash
CUDA_VISIBLE_DEVICES=0 bash embodied_eval/tasks/vabench/scripts/iflybot_vlm.sh
```

### 4. Reading results

- Run folder: `<output_path>/<timestamp>/`
- Main artifacts (same pattern as other EQA tasks):
  - `inference_vabench.json` — raw model outputs
  - `samples_vabench.json` — per-sample scores
  - `results_vabench.json` — aggregated metrics

In `results_vabench.json`, typical fields:

```json
{
  "vabench_accuracy": 0.72,
  "accuracy_average": 0.72,
  "overall": 0.72
}
```

Meaning:

- `*_accuracy` — mean hit rate (0–1) per `question_type`
- `accuracy_average` — mean across types (0–1)
- `overall` — final aggregate (0–1)

**Scoring:** VABench-P parses the eight normalized points from the model output and computes accuracy against the annotated mask.

**Website scale (0–100):** Raw scores are in \[0, 1\]; **website score = raw score × 100**.
