# Embodied Arena Eval — Developer Guide

> For framework developers: benchmark maintenance and model integration.

---

## Table of contents

- [Part I — Benchmark development](#part-i--benchmark-development)
- [Part II — Model integration](#part-ii--model-integration)

---

# Part I — Benchmark development

## Core expectations

### Authoring principles

**Write for third-party users:** Assume no local datasets, no preconfigured models—only this repository.

### End-to-end user flow

```
Download data → Prepare data → Download model → Configure model → Run evaluation → Interpret results
```

### What most benchmarks still need

Most benchmarks already have `yaml` and `process.py`. **Please add:**

1. **`README.md`** — Full third-party flow (six-step pipeline).
2. **`run_eval.sh`** — Standard one-command evaluation.
3. **`utils/download.py`** — Data download helper (recommended).
4. **Metric documentation** — Aligned with the Embodied Arena website.

### Key rules

- ✅ Keep everything under `embodied_eval/tasks/<your_benchmark>/`.
- ❌ Do not add scripts to the repository root.

---

## 1. Standard directory layout

Each benchmark should follow:

```
embodied_eval/tasks/<your_benchmark>/
├── README.md                    # Required: evaluation guide
├── <benchmark>.yaml             # Required: task config
├── process.py                   # Required: data / metric logic
├── run_eval.sh                  # Recommended: standard eval script
└── utils/                       # Optional
    ├── README.md                # Tooling docs
    ├── download.py              # Data download
    └── preprocess.py            # Preprocessing
```

**Reference:** `embodied_eval/tasks/robovqa/`

---

## 2. Required files

### 2.1 `README.md` (evaluation guide)

**Requirements:** Clear and focused on the happy path, **≤ ~150 lines**.

**Must cover these six steps:**

#### Step 1 — Download data

Provide sources and commands:

```markdown
## 1. Data preparation

### 1.1 Download the dataset

```bash
# Download dataset
python -c "from datasets import load_dataset; load_dataset('xxx/dataset', split='val').save_to_disk('/path/to/dataset')"

# Download videos (if needed)
python -m embodied_eval.tasks.<benchmark>.utils.download --auto_download
```

### 1.2 Configure data paths

Edit `<benchmark>.yaml`:
```yaml
dataset_path: /path/to/dataset
dataset_kwargs:
  video_dir: /path/to/videos
```
```

#### Step 2 — Preprocess (if needed)

Document commands when preprocessing is required.

#### Steps 3–4 — Download and configure the model

```markdown
## 2. Run evaluation

### 2.1 Using the eval script

Edit `run_eval.sh`:
- `model_name_or_path`: Hugging Face ID or local path
- `max_num_frames`: video frames to sample (often 8–32)
```

#### Step 5 — Launch evaluation

```markdown
### 2.2 Execute

```bash
bash embodied_eval/tasks/<benchmark>/run_eval.sh
```
```

#### Step 6 — Interpret results

```markdown
## 3. Understanding outputs

### 3.1 Output files

Under `<output_path>/YYYYMMDD_HHMMSS/`:
- `results_<benchmark>.json` — Primary metrics (start here)

### 3.2 Main metrics

- `metric_name`: Description (aligned with Embodied Arena’s XXX metric)
```

**Avoid:**

- ❌ Lengthy environment setup unless there are special dependencies
- ❌ Exhaustive parameter matrices
- ❌ Many alternative run modes
- ❌ Oversized example dumps

### 2.2 `run_eval.sh`

**Template:**

```bash
#!/bin/bash
# <Benchmark> evaluation script

cd "$(dirname "$0")/../../.." || exit

export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='your-api-base'  # optional

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model qwen3_vl \
    --model_args model_name_or_path=/path/to/model,max_num_frames=32,fps=2 \
    --evaluator eqa \
    --tasks your_benchmark \
    --batch_size 1 \
    --output_path ./logs/your_benchmark/results
```

**Checklist:**

- ✅ Include `cd "$(dirname "$0")/../../.." || exit`
- ✅ Keep arguments minimal; users edit paths and model args
- ✅ Avoid complex branching or multiple modes in one script

### 2.3 `<benchmark>.yaml`

```yaml
task: your_benchmark
dataset_path: /path/to/dataset
dataset_kwargs:
  video_dir: /path/to/videos
load_from_disk: true
eval_split: val
doc_to_visual: !function process.your_doc_to_visual
doc_to_text: !function process.your_doc_to_text
doc_to_target: "answer"
output_type: generate_until
generation_kwargs:
  max_new_tokens: 256
  temperature: 0
process_results: !function process.your_process_results
metric_kwargs:
  metric: your_score
  aggregation: !function process.your_aggregate_results
  higher_is_better: true
```

### 2.4 `process.py`

**Required functions:**

```python
def your_doc_to_visual(doc):
    """Extract visual inputs."""
    pass

def your_doc_to_text(doc):
    """Extract question text."""
    pass

def your_process_results(doc, results):
    """Per-sample scoring."""
    pass

def your_aggregate_results(results):
    """Aggregate to final metrics."""
    pass
```

---

## 3. Optional files

### 3.1 `utils/`

**Recommended:**

- `download.py` — CLI + Python API for fetching data
- `preprocess.py` — Preprocessing
- `README.md` — How to use the tools

Reference: `embodied_eval/tasks/robovqa/utils/`

---

## 4. Development policy

### 4.1 ✅ Allowed changes

- ✅ Anything under your benchmark directory (`embodied_eval/tasks/<your_benchmark>/`)
- ✅ Updating the main README benchmark list

### 4.2 ❌ Disallowed changes

- ❌ No eval scripts at repository root
- ❌ Do not modify other benchmarks’ code without coordination
- ❌ No scratch files, one-off tests, or AI drafts in the root

### 4.3 🗑️ Root cleanliness

**Keep the repository root tidy.**

**Do not place at root:**

- ❌ Ad-hoc test scripts
- ❌ AI-generated scratch code
- ❌ Personal experiment files
- ❌ Obsolete scripts

**Instead:**

- ✅ Delete what is unused
- ✅ Review root layout before each development cycle

**`history_archive/`:**

- For deprecated but potentially useful material
- Organize by owner or feature
- Prune periodically

### 4.4 📊 Data storage

**Allowed in `data/`:**

- ✅ Small JSON annotations (e.g. `data/<benchmark>/*.json`)

**Preferred for large assets:**

- ✅ Host large datasets on a data volume or user-defined path
- ✅ Use absolute paths or environment variables in YAML
- ✅ Document download and layout in README

---

## 5. Benchmark checklist

Before merging:

- [ ] Can a user **download data** from scratch? (commands provided)
- [ ] Can a user **obtain the model** from scratch? (sources + config)
- [ ] Is there a **one-command eval**? (`run_eval.sh`)
- [ ] Are **metrics explained** and aligned with Embodied Arena?
- [ ] Is the README **concise** (≤ ~150 lines, no fluff)?
- [ ] Is the **root directory clean** (no stray scripts/tests)?

---

## 6. Summary (benchmarks)

### Goal: complete third-party flow

The README must let someone go from zero to results:

```
1. Download data → 2. Prepare data → 3. Download model → 4. Configure model → 5. Run eval → 6. Interpret results
```

### Three pillars

1. **Complete** — All six steps with runnable commands
2. **Concise** — README stays short and on the critical path
3. **Aligned** — Metrics match Embodied Arena

**Reference:** `embodied_eval/tasks/robovqa/`

---

# Part II — Model integration

## Core expectations

Add models under `embodied_eval/models/` so every benchmark can call them.

---

## 1. Model integration workflow

### Step 1 — Create the model class

Add `embodied_eval/models/<model_name>.py`:

```python
from embodied_eval.models.base import BaseAPIModel

class YourNewModel(BaseAPIModel):
    def __init__(self, model_name_or_path, **kwargs):
        """
        Initialize the model.
        Args:
            model_name_or_path: Path or HF model id
            **kwargs: Model-specific options (e.g. max_num_frames, fps)
        """
        super().__init__()
        # Initialization
        pass

    def respond(self, query, visual_input=None, **kwargs):
        """
        Generate a response.
        Args:
            query: Question text
            visual_input: Image path, video path, or list of paths
            **kwargs: Extra generation args
        Returns:
            str: Model answer
        """
        # Inference
        pass
```

**Implementation notes:**

- `visual_input` may be a single image path (`str`), a video path (`str`), or a list of image paths.
- Handle the formats your model expects.
- Always return a `str`.

### Step 2 — Register the model

In `embodied_eval/models/__init__.py`:

```python
from .your_new_model import YourNewModel

MODEL_REGISTRY = {
    # ... existing entries
    'your_new_model': YourNewModel,
}
```

*(If your tree uses `AVAILABLE_MODELS` instead, follow the pattern in the current `__init__.py`.)*

### Step 3 — Example launch script

Add `example/vqa/<model_name>.sh`:

```bash
#!/bin/bash
cd "$(dirname "$0")/../.." || exit

export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model your_new_model \
    --model_args model_name_or_path=/path/to/model,max_num_frames=32 \
    --evaluator eqa \
    --tasks robovqa \
    --batch_size 1 \
    --output_path ./logs/robovqa/your_model
```

**Checklist:**

- ✅ `cd "$(dirname "$0")/../.." || exit`
- ✅ Minimal, editable arguments

---

## 2. Common issues

### 2.1 Dependencies

Document installs and troubleshooting in `embodied_eval/models/README.md`:

```markdown
### YourModel

**Install:**
```bash
pip install your-model-package
```

**FAQ:**
1. Symptom
   - Fix
```

Mirror the style of existing model entries.

### 2.2 Visual inputs

**Image-focused models:**

```python
def respond(self, query, visual_input=None, **kwargs):
    if isinstance(visual_input, str):
        image = load_image(visual_input)
    elif isinstance(visual_input, list):
        images = [load_image(img) for img in visual_input]
```

**Video models:**

```python
def respond(self, query, visual_input=None, **kwargs):
    if visual_input and visual_input.endswith('.mp4'):
        frames = load_video(visual_input, max_num_frames=self.max_num_frames)
```

### 2.3 LoRA checkpoints

```python
def __init__(self, model_name_or_path, **kwargs):
    if os.path.exists(os.path.join(model_name_or_path, "adapter_config.json")):
        from peft import PeftModel
        base_model = load_base_model()
        self.model = PeftModel.from_pretrained(base_model, model_name_or_path)
    else:
        self.model = load_model(model_name_or_path)
```

See `embodied_eval/models/qwen3_vl.py` for a concrete pattern.

---

## 3. Model development policy

### 3.1 ✅ Allowed

- ✅ New files under `embodied_eval/models/`
- ✅ Registration in `embodied_eval/models/__init__.py`
- ✅ Updates to `embodied_eval/models/README.md`
- ✅ Scripts under `example/vqa/`

### 3.2 ❌ Disallowed

- ❌ Unrelated edits to other models
- ❌ Changes to `base.py` or core framework code without strong justification
- ❌ Scratch files at repository root

### 3.3 Naming

- File: `<model_name>.py` (snake_case)
- Class: `<ModelName>` (PascalCase)
- Registry key: `'model_name'` (snake_case, matches file stem)

---

## 4. Model integration checklist

- [ ] Model file lives in `embodied_eval/models/`?
- [ ] Registered in `__init__.py`?
- [ ] Example script under `example/vqa/<model>.sh`?
- [ ] Script `cd`s to project root?
- [ ] Dependencies documented in `models/README.md`?
- [ ] Handles images, videos, and frame lists as needed?
- [ ] Smoke-tested on at least one benchmark?

---

## 5. Summary (models)

### Three steps

1. **Implement** — `embodied_eval/models/<model>.py`
2. **Register** — `embodied_eval/models/__init__.py`
3. **Example** — `example/vqa/<model>.sh`

### Essentials

- Subclass `BaseAPIModel`
- Implement `__init__` and `respond`
- Support the visual modalities you claim
- Return strings

### Documentation

Record dependency conflicts and setup issues in `embodied_eval/models/README.md`.

**Further reading:**

- Example implementation: `embodied_eval/models/qwen3_vl.py`
- Dependency notes: `embodied_eval/models/README.md`

---

## Appendix

### References

- Benchmark example: `embodied_eval/tasks/robovqa/`
- Model example: `embodied_eval/models/qwen3_vl.py`
- Main docs: `README.md`

### Housekeeping

**Keep the root clean:**

- 🗑️ No temporary tests, scratch scripts, or AI drafts at top level
- 📦 Move legacy material to `history_archive/` (clear ownership)
- 🧹 Audit the root before starting new work
