# `embodied_eval` — Core package overview

This document describes the layout of the `embodied_eval` evaluation library and the role of each major module.

---

## `models/`

Model implementations and registration. Each model lives in a `*.py` file, subclasses `BaseAPIModel`, and implements `__init__` and `respond`. Models are registered in `AVAILABLE_MODELS` for the evaluation entry point to resolve.

- **Key files:** `__init__.py` (registry, loading helpers, `BaseAPIModel` base class)
- **Examples:** `qwen2_5_vl.py`, `internvl3.py`, `pelican_vl.py`, `embodied_brain.py`, etc.
- **Docs:** `models/README.md` — install steps and troubleshooting per model

---

## `tasks/`

Benchmark definitions. Each benchmark has its own subdirectory with:

- **`<benchmark>.yaml`** — Task config (dataset paths, doc/metric hooks, generation settings, etc.)
- **`process.py`** — Implements `doc_to_visual`, `doc_to_text`, `doc_to_target`, `process_results`, aggregation helpers, etc.
- **`run_eval.sh`** — One-command eval for that benchmark (most benchmarks include one); edit `--model` / `--model_args` and run
- **`scripts/`** — Per-model launch scripts (e.g. `qwen2_5_vl.sh`, `pelican_vl.sh`). To add a new model for this benchmark, copy an existing script and adjust model name and arguments
- **`utils/`** — Download, preprocessing, and other helpers
- **`README.md`** — Data prep, how to run, and how to read results

**Benchmark index:**

| Benchmark | Directory | Task focus |
|-----------|-----------|------------|
| RoboVQA | `robovqa/` | Robotic visual QA |
| UniEQA | `unieqa/` | Unified embodied QA |
| VABench | `vabench/` | Spatial pointing / selection |
| VSI-Bench | `vsibench/` | Spatial reasoning |
| EmbodiedScene | `embodied_scene/` | Embodied scene understanding |
| Beacon3D | `beacon3d/` | 3D scene QA / grounding |
| Where2Place | `where2place/` | Spatial placement |
| Cosmos | `cosmos/` | Embodied reasoning |
| EA-Temporal | `ea_temporal/` | Temporal perception |
| EmbSpatial-Bench | `emspatial_bench/` | Spatial relations |
| OpenEQA | `openeqa/` | Open-ended embodied QA |
| ERQA | `erqa/` | Embodied QA |
| MSQA | `msqa/` | Multi-turn visual QA |
| OST_Bench | `OST_Bench/` | Open-set embodied tasks |

---

## `evaluators/`

Evaluator implementations:

- **`eqa_evaluator.py`** — Generic visual QA (inference + scoring + aggregation)
- **`nav_evaluator.py`** — Navigation-style tasks
- **`official_beacon3d_evaluator.py`** — Official Beacon3D evaluation path

`build_evaluator(args)` selects the implementation from `--evaluator`.

---

## `common/`

Shared data structures (e.g. `Instance`, `TaskConfig`), registries, and metric aggregation utilities.

---

## `envs/`

Simulation and interaction environments for navigation, manipulation, and similar settings:

- **`gym`** — Gym-style wrappers
- **`EBNavEnv`** — Navigation simulation
- **`envs/README.md`** — Setup for gym, Loho-Ravens, AI2-THOR, etc.

---

## `utils/`

Cross-cutting helpers: YAML handling, argument parsing, data loading, etc.

---

## Configuring `data/`

**Important:** The open-source tree does **not** ship a populated `data/` directory. Before running evaluations, create a data root, download datasets and checkpoints, and point your YAML / CLI to those paths.

### 1. Create a data root

Create `embodied_eval/data/` under the repo, or use any directory outside the project. Large assets are often symlinked into `data/` instead of copying.

### 2. Datasets

Requirements differ per benchmark—follow each `tasks/<benchmark>/README.md` for download links and expected layout.

Some benchmarks (e.g. VABench) load directly from Hugging Face and may not need a local mirror.

### 3. Model checkpoints

Host weights under `data/` or anywhere on disk:

| Model / setup | Typical layout |
|---------------|----------------|
| EmbodiedBrain-7B | `data/EmbodiedBrain-7B/` |
| Other open models | Any path, passed via `model_name_or_path` |
| Remote / HF models | Set `model_name_or_path` to the Hugging Face id or URL |

See `models/README.md` for download links and dependencies.

### 4. Wire paths in YAML

After data and models are in place, set `dataset_path` and `dataset_kwargs` in the benchmark YAML:

```yaml
# Example: embodied-scene.yaml
dataset_path: /path/to/your/data/EmbodiedScene/embodied_scene_data.json
dataset_kwargs:
  image_dir: /path/to/your/data/EmbodiedScene
```

### 5. Quick sanity checks

- Read the target benchmark’s `README.md`
- Verify `dataset_path`, `video_dir`, `image_dir`, and similar fields
- Use `--limit 1` for a fast load test before a full run
