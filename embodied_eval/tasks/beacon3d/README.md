# Beacon3D Benchmark

Full evaluation:

# Environment setup
cd /home/arena/embodiedeval/embodied-arena
conda activate embodied-eval
export OPENAI_API_KEY=''
export OPENAI_API_BASE=''

# Run evaluation
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
accelerate launch --num_processes=1 --main_process_port=$PORT -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=/data/arena/models/Qwen2_5-VL-3B-Instruct,max_num_frames=8 \
    --tasks beacon3d-qa \
    --batch_size 1 \
    --output_path ./logs/beacon3d_results \
    --limit 10  # Adjust as needed

**Official site**: https://github.com/beacon-3d/Beacon3D

## Important notice

**Beacon3D originally required 3D scene understanding models, but is now adapted for 2D image/video models.**

- **Grounding task**: Originally required point clouds or 3D bounding box prediction; now uses 2D image sequences (effects may be limited)
- **QA task**: Originally required 3D scene understanding; now uses 2D video/image input
- **Current adaptation**: Supports 2D vision–language models such as Qwen2.5-VL and LLaVA

**Current status**: Adapted for 2D models; supports vision + text input testing.

## Tasks

- `beacon3d-grounding`: 3D object localization task (now adapted to 2D image sequences)
- `beacon3d-qa`: 3D visual question answering task (now adapted to 2D video/images)

## Data setup

### 1. Beacon3D data (already prepared)

```bash
/data/arena/datasets/Beacon3D/
├── data/scannet/scannet_grounding.json (3389 samples)
├── data/scannet/scannet_qa.json (3250 samples)
├── data/scannet/scans/{scan_id}/images/color/*.jpg (ScanNet image sequences)
├── data/3rscan/3rscan_grounding.json (3199 samples)
├── data/3rscan/3rscan_qa.json (2763 samples)
├── data/3rscan/data/{scene_id}/*.jpg (3RScan images)
├── data/multiscan/multiscan_grounding.json (1555 samples)
├── data/multiscan/multiscan_qa.json (1417 samples)
└── data/multiscan/data/{scene_id}/{scene_id}.mp4 (MultiScan videos)
```

### 2. Visual data formats

- **ScanNet**: Image sequences `scans/{scan_id}/images/color/*.jpg`
- **3RScan**: Image files `data/{scene_id}/*.jpg`
- **MultiScan**: Video files `data/{scene_id}/{scene_id}.mp4`

### 3. Optional: download raw 3D scene data

- **ScanNet**: http://www.scan-net.org/
- **3RScan**: https://waldjohannau.github.io/RIO/
- **MultiScan**: https://3dlg-hcvc.github.io/multiscan/

## Configuration

### Per-dataset-domain settings

Edit `dataset_path` and `domain` in the YAML files:

**ScanNet** (default):

```yaml
dataset_path: /data/arena/datasets/Beacon3D/data/scannet/scannet_grounding.json
dataset_kwargs:
  domain: scannet
  point_cloud_dir: /path/to/scannet/scans
```

**3RScan**:

```yaml
dataset_path: /data/arena/datasets/Beacon3D/data/3rscan/3rscan_grounding.json
dataset_kwargs:
  domain: 3rscan
  point_cloud_dir: /path/to/3rscan/scans
```

**MultiScan**:

```yaml
dataset_path: /data/arena/datasets/Beacon3D/data/multiscan/multiscan_grounding.json
dataset_kwargs:
  domain: multiscan
  point_cloud_dir: /path/to/multiscan/scans
```

## Adaptation notes

### Supported 2D models

Models currently adapted for 2D input:

- `qwen2_5_vl`: image/video input ✅
- `llava_onevision`: image/video input
- `internvl3`: image/video input

### Visual data handling

- **QA task**: uses video/image sequences as visual input
- **Grounding task**: uses image sequences (effects may be limited)
- **Auto adaptation**: picks a suitable visual format by dataset domain

### Current behavior

- `doc_to_visual()` returns the corresponding visual paths for the dataset
- The model consumes vision + text input
- QA uses exact match (falls back automatically when LLM evaluation is unavailable)
- Grounding uses IoU computation

## Usage (vision + text mode)

### Grounding task test

```bash
accelerate launch --num_processes=1 -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=/data/arena/models/Qwen2_5-VL-3B-Instruct/ \
    --tasks beacon3d-grounding \
    --batch_size 1 \
    --limit 5 \
    --output_path ./logs/beacon3d_grounding_test
```

### QA task test

```bash
accelerate launch --num_processes=1 -m embodied_eval \
    --model qwen2_5_vl \
    --model_args model_name_or_path=/data/arena/models/Qwen2_5-VL-3B-Instruct/ \
    --tasks beacon3d-qa \
    --batch_size 1 \
    --limit 5 \
    --output_path ./logs/beacon3d_qa_test
```

## Metrics

- **Grounding**: IoU@0.25, IoU@0.5, range 0–1 (simplified for text-only mode where applicable)
- **QA**: exact match or LLM semantic match, range 0–1 (LLM evaluation disabled when no API; falls back to exact match)

**Website score (0–100 scale)**: Raw metrics above are in 0–1; they map linearly to percentage: **website score = raw score × 100**

## Future work

1. **Add 3D models**: integrate RoboPoint or similar 3D models
2. **Enable visual input for 3D**: update `doc_to_visual()` to handle real 3D data
3. **Full evaluation**: re-enable LLM-based QA scoring

## Test verification

Framework integration has been tested and works:

```bash
# Activate environment
conda activate embodied-eval

# Check task registration
python -c "from embodied_eval.tasks import TaskManager; tm=TaskManager(); print([t for t in tm.all_tasks if 'beacon3d' in t])"
# Output: ['beacon3d-grounding', 'beacon3d-qa']
```

## Test results

### Grounding task test results

```
Grounding Evaluation Results:
{'unknown_iou_0.25': 0.0, 'unknown_iou_0.5': 0.0, 'overall_iou_0.25': 0.0, 'overall_iou_0.5': 0.0}
```

### QA task test results

```
QA Evaluation Results:
{'unknown_exact_match': 1.0, 'unknown_llm_match': 1.0, 'overall_exact_match': 1.0, 'overall_llm_match': 1.0, 'match_average': 1.0}
```

**Note**: Because a 2D model is used on a 3D task, IoU grounding scores of 0 are expected. For QA, LLM evaluation is unavailable, so evaluation falls back to exact match.
