# VSI-Bench utils

Helper scripts for preparing the VSI-Bench dataset.

## Files

### `download.py`

Downloads the dataset release.

**Features:**

- Fetch VSI-Bench from Hugging Face (zips plus `test.jsonl`)

**CLI:**

```bash
python -m embodied_eval.tasks.vsibench.utils.download \
    --local_dir /path/to/vsi-bench-raw
```

**Or with the Hugging Face CLI:**

```bash
hf download nyu-visionx/VSI-Bench --repo-type dataset --local-dir /path/to/vsi-bench-raw
```

**Artifacts:**

- `scannet.zip` — ScanNet videos
- `scannetpp.zip` — ScanNet++ videos
- `arkitscenes.zip` — ARKitScenes videos
- `test.jsonl` — test split (questions and answers)

### `preprocess.py`

Preprocessing pipeline.

**Features:**

- Extract video archives
- Convert JSONL rows into a Hugging Face `Dataset`

**CLI:**

```bash
# Full run (extract + convert)
python -m embodied_eval.tasks.vsibench.utils.preprocess \
    --raw_dir /path/to/vsi-bench-raw \
    --output_dir /path/to/vsi-bench

# Extract videos only
python -m embodied_eval.tasks.vsibench.utils.preprocess \
    --raw_dir /path/to/vsi-bench-raw \
    --output_dir /path/to/vsi-bench \
    --extract_only

# Convert JSONL only
python -m embodied_eval.tasks.vsibench.utils.preprocess \
    --raw_dir /path/to/vsi-bench-raw \
    --output_dir /path/to/vsi-bench \
    --convert_only
```

**Steps:**

1. Unpack `scannet.zip`, `scannetpp.zip`, and `arkitscenes.zip` into the video folders
2. Turn `test.jsonl` into Hugging Face `Dataset` files under `test/`

**Layout:**

```text
output_dir/
├── test/                    # Hugging Face Dataset
│   ├── dataset_info.json
│   └── ...
├── scannet/                 # videos
│   ├── scene0015_00.mp4
│   └── ...
├── scannetpp/               # videos
│   └── ...
└── arkitscenes/             # videos
    └── ...
```

## Quick start

### 1. Download

```bash
hf download nyu-visionx/VSI-Bench --repo-type dataset --local-dir /path/to/vsi-bench-raw
```

### 2. Preprocess

```bash
python -m embodied_eval.tasks.vsibench.utils.preprocess \
    --raw_dir /path/to/vsi-bench-raw \
    --output_dir /path/to/vsi-bench
```

### 3. Point `vsibench.yaml` at the output

```yaml
dataset_path: /path/to/vsi-bench
dataset_kwargs:
  video_dir: /path/to/vsi-bench
```

## Notes

- Extracted videos need roughly **5.4 GB** of free disk space
- Leave enough headroom on disk
- Extraction can take a while
