# Embodied Arena Evaluation Framework

## 1. Overview

The Embodied Arena evaluation framework is a comprehensive platform for **embodied AI**, designed to assess vision–language models (VLMs) on embodied scene understanding, robotic manipulation, spatial reasoning, temporal perception, and related tasks.

**Key features:**

- **Multi-benchmark support:** RoboVQA, UniEQA, VABench, VSI-Bench, EmbodiedScene, Beacon3D, Where2Place, Cosmos, EA-Temporal, EmbSpatial-Bench, OpenEQA, ERQA, and more.
- **Broad model coverage:** 28+ integrated models, including closed APIs, open general VLMs, and embodied-specialized models.
- **Unified pipeline:** YAML configs plus `process.py` for loading, inference, and metrics—easy to extend and reproduce.
- **Multiple evaluators:** EQA (visual QA), navigation (Nav), official Beacon3D evaluation, and others.

The framework aligns with the [Embodied Arena](https://www.embodied-arena.com/) platform for benchmarking, leaderboard submissions, and paper reproduction.

---

## 2. Repository layout

### 2.1 Directory tree

```
embodied-eval-main/
├── embodied_eval/              # Core evaluation library
│   ├── models/                 # Model implementations and registry
│   ├── tasks/                  # Benchmark definitions and data handling
│   ├── evaluators/             # Evaluator implementations
│   ├── common/                 # Shared structures and utilities
│   ├── envs/                   # Simulation environments (navigation, manipulation, etc.)
│   ├── utils/                  # General helpers
│   └── data/                   # Models and datasets (create and configure locally; see embodied_eval/README.md)
├── logs/                       # Evaluation outputs
├── utils/                      # Project-level utilities
├── requirements.txt
├── DEVELOPER_GUIDE.md          # Developer guide
└── README.md
```

### 2.2 Top-level folders

| Folder | Purpose |
|--------|---------|
| `embodied_eval/` | Core library: models, tasks, evaluators, environments |
| `logs/` | Outputs organized by task and model |
| `utils/` | Project-level scripts |

### 2.3 `embodied_eval` in detail

See [embodied_eval/README.md](embodied_eval/README.md).

---

## 3. Quick start

### 3.1 Environment setup

```bash
# Clone the repo and enter the project root
cd embodied-eval-main

# Create and activate a conda environment
conda create -n embodied-eval python=3.10
conda activate embodied-eval

# Install dependencies
pip install -r requirements.txt
```

**Model-specific dependencies:** Most models need extra packages (e.g. Qwen2.5-VL needs `qwen-vl-utils`; VILA, LLaVA, etc. need their respective repos). See `embodied_eval/models/README.md`. Some model stacks conflict; if installing many models in one env fails, use separate conda environments per conflicting model.

### 3.2 Basic workflow

The entry point is `python -m embodied_eval` with model, task, and evaluator specified.

**Option A — `run_eval.sh` (one-click batch, recommended)**

Each benchmark under `embodied_eval/tasks/<benchmark>/` provides a **`run_eval.sh`** in the same directory as its **`scripts/`** folder. That layout is the standard “one command to run many models” path (use **bash** on Linux; the driver does not change your current working directory).

| Item | Description |
|------|-------------|
| **Driver script** | `embodied_eval/tasks/<benchmark>/run_eval.sh` |
| **Per-model scripts** | `embodied_eval/tasks/<benchmark>/scripts/*.sh` |
| **What gets run** | Inside `run_eval.sh`, the bash array **`RUN_SCRIPTS`** lists **basenames only** (e.g. `qwen3_vl.sh`). The driver runs `bash` on `scripts/<basename>` in order. Edit that array to add, remove, or reorder models. Some benchmarks omit non-eval helpers (data prep, splitters, `all.sh`-style aggregators); see comments in each `run_eval.sh`. |
| **Output & errors** | Each subscript’s stdout/stderr is printed to your terminal as usual. If one script fails, the driver continues with the next and prints a **summary** at the end; the process exits with code **1** if any run failed or a listed file was missing. |

**If a subscript fails:** Open the corresponding **`embodied_eval/tasks/<benchmark>/scripts/<name>.sh`** and fill in what that template expects—commonly **`OPENAI_API_KEY` / `OPENAI_API_BASE`** (LLM-as-judge or API models), **checkpoint and data paths** (`model_name_or_path`, `--output_path`), **`CUDA_VISIBLE_DEVICES`**, or **conda** `source`/`activate` lines—then re-run.

```bash
cd embodied-eval-main
bash embodied_eval/tasks/robovqa/run_eval.sh
```

**Option A2 — Single model script**

Run one launcher directly (after editing that file):

```bash
bash embodied_eval/tasks/vabench/scripts/qwen2_5_vl.sh
```

To evaluate a new model, add `your_model.sh` under that benchmark’s `scripts/` (copy an existing launcher and adjust `--model`, `--model_args`, `--output_path`), and add `"your_model.sh"` to **`RUN_SCRIPTS`** in `run_eval.sh` if you want it included in the batch.

**Option B — Direct CLI**

```bash
cd embodied-eval-main

# Random port to avoid collisions when launching multiple jobs
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

# Single-GPU run
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=/path/to/Qwen2.5-VL-7B-Instruct,max_num_frames=32,fps=2 \
    --evaluator eqa \
    --tasks robovqa \
    --batch_size 1 \
    --output_path ./logs/robovqa/qwen2_5_vl
```

### 3.3 Common CLI arguments

| Argument | Description | Examples |
|----------|-------------|----------|
| `--model` | Registered model name | `qwen2_5_vl`, `internvl3`, `pelican_vl` |
| `--model_args` | Comma-separated model kwargs | `model_name_or_path=...,max_num_frames=32,fps=2` |
| `--evaluator` | Evaluator type | `eqa` (default), `nav`, `official_beacon3d` |
| `--tasks` | Task / benchmark id | `robovqa`, `unieqa`, `vabench`, `embodied-scene`, … |
| `--batch_size` | Batch size | `1` |
| `--output_path` | Output root | `./logs/robovqa/results` |
| `--limit` | Max samples (debug) | `10` |
| `--inference_only` | Run inference only, skip metrics | — |

Benchmarks that use LLM-as-judge need:

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.openai.com/v1'  # or your endpoint
```

### 3.4 Data and configuration

**Data directory:** This repo does **not** ship `embodied_eval/data/`. Create the folder, download datasets and checkpoints, and set paths in YAML. See [embodied_eval/README.md](embodied_eval/README.md), section on configuring `data/`.

- **Paths:** Set `dataset_path` and `dataset_kwargs` in each benchmark’s `*.yaml`.
- **Preparation:** Each benchmark’s `README.md` describes download and preprocessing.
- **Example:** RoboVQA — `embodied_eval/tasks/robovqa/README.md`.

### 3.5 Outputs

Results live under `--output_path` in timestamped folders `YYYYMMDD_HHMMSS/`:

```
├── inference_<task>.json     # Raw predictions
├── samples_<task>.json       # Per-sample evaluation
└── results_<task>.json       # Aggregated metrics (primary file)
```

---

## 4. Adding a new model or benchmark

### 4.1 Adding a model

**Step 1 — Implement the model class**

Create `your_model.py` under `embodied_eval/models/`, subclassing `BaseAPIModel`:

```python
from embodied_eval.models import BaseAPIModel
from typing import List, Tuple, Union

class YourNewModel(BaseAPIModel):
    def __init__(self, model_name_or_path, max_num_frames=32, fps=2, **kwargs):
        super().__init__()
        # Load model, tokenizer, etc.
        pass

    def respond(self, context: str, visuals: Union[List[str], List[Tuple[str, int]]], **gen_kwargs) -> str:
        """Generate an answer from context and visual inputs."""
        # Implement inference; return a string
        return "model output"
```

**Step 2 — Register the model**

In `embodied_eval/models/__init__.py`, add to `AVAILABLE_MODELS`:

```python
AVAILABLE_MODELS = {
    # ...
    "your_new_model": "YourNewModel",  # or "embodied_eval.models.your_model.YourNewModel"
}
```

**Step 3 — Add a launch script**

Under the target benchmark’s `scripts/`, add `your_model.sh` (copy from e.g. `qwen2_5_vl.sh`) and set `--model`, `--model_args`, `--output_path`. Alternatively extend `run_eval.sh`.

**Step 4 — Document dependencies**

Update `embodied_eval/models/README.md` with install steps and known issues.

### 4.2 Adding a benchmark

**Step 1 — Create a task directory**

`embodied_eval/tasks/your_benchmark/`.

**Step 2 — YAML config**

`your_benchmark.yaml`:

```yaml
task: your_benchmark
dataset_path: /path/to/your/data
dataset_kwargs:
  video_dir: /path/to/videos
  image_dir: /path/to/images
load_from_disk: false
eval_split: test
doc_to_visual: !function process.your_doc_to_visual
doc_to_text: !function process.your_doc_to_text
doc_to_target: !function process.your_doc_to_target
output_type: generate_until
generation_kwargs:
  max_new_tokens: 256
  temperature: 0
process_results: !function process.your_process_results
metric_kwargs:
  metric: your_metric_name
  aggregation: !function process.your_aggregate_results
  higher_is_better: true
```

**Step 3 — `process.py`**

Implement the functions referenced above:

```python
def your_doc_to_visual(doc):
    """Extract visual paths from a sample."""
    return doc.get("video_path") or doc.get("image_path")

def your_doc_to_text(doc):
    """Extract question text."""
    return doc["question"]

def your_doc_to_target(doc):
    """Extract ground-truth answer."""
    return doc["answer"]

def your_process_results(doc, results):
    """Per-sample scoring; return a dict."""
    pass

def your_aggregate_results(results):
    """Aggregate scores into final metrics dict."""
    pass
```

**Step 4 — `run_eval.sh` and README**

- Provide a one-command eval script in `run_eval.sh`.
- Document data, config, and how to read results in `README.md`.

For more detail, see `DEVELOPER_GUIDE.md`.

---

## 5. Markdown index (high level)

**Repository root**

- `README.md` — Main documentation: overview, layout, quick start, extension guide.
- `DEVELOPER_GUIDE.md` — Developer guide: benchmark/model conventions, structure, data practices.

**`embodied_eval/`**

- `README.md` — Deep dive into models, tasks, evaluators.

**`embodied_eval/models/`**

- `README.md` — Per-model install notes (flash-attention, Qwen2.5-VL, VILA, RoboPoint, etc.).

**`embodied_eval/tasks/<benchmark>/`**

- **`run_eval.sh`** — Batch driver next to `scripts/`; runs each basename listed in the embedded **`RUN_SCRIPTS`** array against `scripts/<name>.sh` (see §3.2).
- Most benchmarks include a **`README.md`** for data prep, how to run, and interpreting metrics.

---

## FAQ

**Q1: How do I change parallelism?**

Edit the per-model script under `embodied_eval/tasks/<benchmark>/scripts/` (e.g. `CUDA_VISIBLE_DEVICES`, `accelerate launch --num_processes=N`), or pass the same flags when using **Option B** (direct CLI).

**Q2: How do I add a new metric?**

Implement it in that benchmark’s `process.py` (`process_results` / aggregation), and wire it in YAML `metric_kwargs`.

**Q3: Which visual formats are supported?**

Images (JPG, PNG, …) and videos (MP4, AVI, …) depending on the model—check `embodied_eval/models/README.md`.

**Q4: Out of GPU memory?**

Lower `max_num_frames` and `batch_size`, use a smaller model, or enable `use_flash_attention_2` where supported.

---

## Contributing

Contributions of new models and benchmarks are welcome.

1. Follow existing code style and directory conventions.
2. Ship a complete `README.md` and `run_eval.sh` for new benchmarks.
3. Record model dependencies and pitfalls in `embodied_eval/models/README.md`.
4. Follow `DEVELOPER_GUIDE.md` for development rules.

---

## License and contact

**License:** This project is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). You may obtain a copy of the License at the preceding link.

**Contact:** [embodiedarena@gmail.com](mailto:embodiedarena@gmail.com)
