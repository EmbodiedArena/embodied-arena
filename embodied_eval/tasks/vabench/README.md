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

#### 2.1 `metric_mode`

Controls how predicted points are scored against the ground-truth mask.
Must be set under `dataset_kwargs` in the YAML; omitting it or using an
unknown value produces a `ValueError`.  Currently the only supported value
is `"mask"`.

**`"mask"`** — binary point-in-mask scoring.
Each predicted point is checked against the reference mask at its pixel
coordinate.  Points that land on a foreground (non-zero) pixel count as a hit;
points on background count as a miss.

```yaml
dataset_kwargs:
  metric_mode: mask
```

#### 2.2 `expected_points`

How many points the model is expected to output in its response.  Must be set
under `dataset_kwargs` in the YAML; omitting it produces a `ValueError`.

The per-sample accuracy is always `hits / expected_points`, where *hits* counts
how many of the first *N* parsed points fall inside the reference mask.

**For VABench-P, `expected_points` must be `8`.**  The original benchmark
requires the model to describe a target region with eight uniformly distributed
points.  Changing this value alters the task semantics and produces scores that
are not comparable with the published leaderboard.

```yaml
dataset_kwargs:
  metric_mode: mask
  expected_points: 8
```

#### 2.3 `coord_mode`

Controls how the model's output coordinates are interpreted.  Must be set under
`dataset_kwargs` in the YAML; omitting it or using an unknown value produces a
`ValueError`.

| Value | Interpretation | Coordinate conversion |
|---|---|---|
| `"normalized"` | Model outputs coordinates in 0–1000 normalized space | `px = round(x/1000 × width)`, `py = round(y/1000 × height)` |
| `"pixel"` | Model outputs absolute pixel coordinates | `px = round(x)`, `py = round(y)` (no scaling) |

`"normalized"` is the standard choice for most VLMs (Qwen, InternVL, GPT, etc.).
`"pixel"` is intended for models that natively output absolute pixel positions
(e.g. RoboBrain2 — see `vabench_robobrain2.yaml`).

```yaml
dataset_kwargs:
  metric_mode: mask
  expected_points: 8
  coord_mode: normalized
```

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

**Scoring logic:**

1. **Parse** — The evaluator extracts numeric coordinates from the model output
   using `omni_decode_points()`, which handles `[[x1,y1],...]` arrays, JSON
   `point_2d` structures, and other common formats.

2. **Scale** — Coordinates are assumed to be in 0–1000 normalized space and
   mapped to absolute pixels: `px = round(x / 1000 * width)`,
   `py = round(y / 1000 * height)`.

3. **Truncate** — Only the first `expected_points` (= 8) parsed coordinates
   are kept.  Fewer than 8 points in the output means fewer points are checked
   against the mask, directly lowering the score.

4. **Check** — Each pixel coordinate is looked up in the ground-truth binary
   mask.  A pixel with value > 0 counts as a hit; a pixel with value 0 (or
   out of image bounds) counts as a miss.

5. **Score** — The per-sample accuracy is `hits / expected_points`.  With 8
   points, possible scores range from 0.0 (all miss) through intermediate
   steps (0.125, 0.25, …) to 1.0 (all hit).

**Website scale (0–100):** Raw scores are in \[0, 1\]; **website score = raw score × 100**.
