# UniEQA evaluation guide

UniEQA (Unified Embodied Question Answering) is a benchmark for embodied agents’ perception, reasoning, and planning in complex indoor scenes.

## 1. Dataset preparation

Follow these steps to obtain and configure the dataset:

**Step 1:** Download the raw [UniEQA](https://drive.google.com/drive/folders/1az4jSfFvKU2_SMWksUICBeU1P-tAwWpo?usp=drive_link) data.

**Step 2:** Download HM3D data. Historical RGB frames for HM3D scenes can be fetched from this [third-party link](https://www.dropbox.com/scl/fi/t79gsjqlan8dneg7o63sw/open-eqa-hm3d-frames-v0.tgz?rlkey=1iuukwy2g3f5t06q4a3mxqobm) (~12 GB). Example download and extract:

```bash
wget -O open-eqa-hm3d-frames-v0.tgz "https://www.dropbox.com/scl/fi/t79gsjqlan8dneg7o63sw/open-eqa-hm3d-frames-v0.tgz?rlkey=1iuukwy2g3f5t06q4a3mxqobm"
md5sum open-eqa-hm3d-frames-v0.tgz  # Expected: 286aa5d2fda99f4ed1567ae212998370
tar -xzf open-eqa-hm3d-frames-v0.tgz -C Part1/images
rm open-eqa-hm3d-frames-v0.tgz
```

Your layout should look like:

```text
|- Part1
   |- images
      |- hm3d
         |- 000-hm3d-BFRyYbPCCPE
         |- ...
| - ...
```

**Step 3:** Download and extract ScanNet data.

UniEQA uses only a subset of ScanNet scenes. We provide an automated script to download the required `.sens` files and extract image frames.

1. **Automated download and extraction (recommended)**

   Run the script below. It detects ScanNet scenes needed for UniEQA, downloads missing `.sens` files, and extracts RGB frames into the target layout.

   ```bash
   cd embodied_eval/tasks/unieqa/utils
   bash download_sens_and_extract.sh
   ```

2. **Manual workflow**

   If you already have the full ScanNet release, place it under `data/raw/scannet` with this structure:

   ```text
   |- data
      |- raw
         |- scannet
            |- scans
               |- <scanId>
                  |- <scanId>.sens
                  |- ...
   ```

   Then run the extraction script:

   ```bash
   # Script: embodied_eval/tasks/unieqa/utils/extract-frames.py
   python embodied_eval/tasks/unieqa/utils/extract-frames.py \
       --dataset <path_to_unieqa_json> \
       --scannet-root data/raw/scannet \
       --output-directory Part1/images/scannet-v0 \
       --rgb-only
   ```

**ScanNet extraction notes**

You can extract RGB only, or RGB, depth, intrinsics, and poses.

| Mode | Size | Approx. time |
| --- | --- | --- |
| RGB only | 62 GB | ~8 hours |
| RGB-D + intrinsics + poses | 70 GB | ~10 hours |

After extraction, your tree should include:

```text
|- Part1
   |- images
      |- scannet-v0
         |- scene0709_00
         |- ...
      |- hm3d-v0
| - ...
```

---

## 2. Repack (Arrow)

After all images are in place, repack into Arrow for faster I/O and multi-image evaluation.

```bash
# From repo root
cd /your/path/to/embodied-arena

# Preprocess (normalize multi-image paths and write Arrow)
python -m embodied_eval.tasks.unieqa.utils.preprocess \
    --data_root PartPATH \
    --convert_hf \
    --output_path YourOutputPATH
```

**Note:** After repacking, set `dataset_path` in `embodied_eval/tasks/unieqa/unieqa.yaml` to `YourOutputPATH`.

## 3. Pre-built dataset

The steps above describe building the dataset locally. If you prefer not to build it yourself, use the ready-made Hugging Face dataset: **EmbodiedArena/unieqa**.

## 4. Scoring

### 4.1 Rules

UniEQA uses **LLM-as-Judge** (GPT-4o-mini) for semantic match scoring per sample:

| Score | Meaning |
| --- | --- |
| 1 | Fully correct semantically |
| 0.5 | Partially correct (only some question types) |
| 0 | Incorrect |

Aggregates include: `{dimension}_score` (per fine-grained capability), `big_{capability}_score` (five macro capabilities), `score_average`, and `overall`.

### 4.2 Website scale (0–100)

Raw scores are in \[0, 1\], linearly mapped to percent: **website score = raw score × 100**

| Raw | Website |
| --- | --- |
| 0 | 0 |
| 0.5 | 50 |
| 1 | 100 |

---

## 5. Evaluation dimensions

UniEQA has 12 fine-grained skills grouped into five macro capabilities:

| Macro capability | Fine-grained skills |
| :--- | :--- |
| **Object Perception** | Object Type, Object Property, Object State |
| **Spatial Perception** | Spatial Perception |
| **Temporal Perception** | Temporal Perception, Action Perception |
| **Embodied Knowledge** | Affordance, World Knowledge |
| **Embodied Reasoning** | Closed-loop Planning, Open-loop Planning, Task-related Object, Situated Reasoning |

## 6. Running evaluation

### Live evaluation

Use the provided scripts (multi-GPU and Flash Attention supported):

```bash
# Example: Qwen3-VL
CUDA_VISIBLE_DEVICES=0,1 bash embodied_eval/tasks/unieqa/scripts/qwen3_vl.sh
```

### Post-evaluation (offline scoring)

If you ran with `--inference_only`, score asynchronously in parallel with GPT-4o-mini:

```bash
export OPENAI_API_KEY='your_key'
python -m embodied_eval.tasks.unieqa.process --force --base_dir ./logs/unieqa/your_experiment
```

## 7. Model tips

- **Multi-image inputs:** Samples often have 1–10 images; consider `max_num_frames=10` in `model_args`.
- **VRAM:** For high-res models, `max_pixels=50176` helps keep speed reasonable.
- **Qwen3-VL:** Use `max_num_frames=1` and `use_flash_attention_2=False`.
- **InternVL3.5-8B:** Use `max_num_frames=1` and `fps=2`.

## 8. FAQ

**Dataset loading:** `git clone` of the UniEQA dataset may leave Git LFS pointer files.

```bash
# Fix: use the download helper to fetch images
python -m embodied_eval.tasks.unieqa.utils.download --auto_download --output_path ./embodied_eval/data/unieqa/UniEQA_Dataset
# Or build the standardized on-disk layout
python -m embodied_eval.tasks.unieqa.utils.build_disk_dataset --raw_data_path ./embodied_eval/data/unieqa/UniEQA_Dataset --output_path ./embodied_eval/data/unieqa/UniEQA_Dataset_Disk
```

**LLM-as-Judge errors:** For “Your request was blocked”, verify `OPENAI_API_KEY` and `OPENAI_API_BASE`.

```bash
export OPENAI_API_KEY='your-api-key'
export OPENAI_API_BASE='https://api.gpt.ge/v1'  # Must end with /v1
```

**Only one sample loads:** Often Hugging Face dataset cache.

```bash
## 9. Path configuration

### 9.0 Dataset directory layout

The pre-built dataset (`EmbodiedArena/unieqa`) contains both the raw images and
the Arrow-repacked data in one tree.  The `unieqa_full_multi_v3/` directory is
the compiled Arrow dataset that `dataset_path` should point to; the `111/UniEQA/`
tree holds the raw JSON task definitions and image files referenced at runtime.

```text
{unieqa_root}/
├── 111/
│   └── UniEQA/
│       ├── data/
│       │   ├── Part1/
│       │   │   ├── images/
│       │   │   │   ├── hm3d-v0/
│       │   │   │   │   ├── 000-hm3d-BFRyYbPCCPE/  (~100–600 PNG frames)
│       │   │   │   │   ├── 001-hm3d-TPhiubUHKcP/
│       │   │   │   │   └── ...
│       │   │   │   └── scannet-v0/
│       │   │   │       ├── scene0709_00/           (~100–600 JPG frames)
│       │   │   │       ├── scene0785_00/
│       │   │   │       └── ...
│       │   │   └── ...
│       │   ├── Part2/
│       │   ├── Part3/
│       │   ├── Part4/
│       │   ├── Part5/
│       │   └── Part6/
│       │       └── images/...
│       ├── affordance/core/data.json
│       ├── spatial_perception/core/data.json
│       ├── object_type/core/data.json
│       └── ... (12 skill dimensions, each with core/data.json)
│       └── README.md
├── unieqa_full_multi_v3/          ← set dataset_path to this directory
│   ├── dataset_dict.json
│   └── train/
│       ├── data-00000-of-00001.arrow   (2 242 samples)
│       ├── dataset_info.json
│       └── state.json
├── download_unieqa_core_json.sh
├── scannet-v0.tar.gz
└── scannet_needed_scenes.txt
```

- The `images` column of every Arrow sample stores absolute paths into the
  `111/UniEQA/data/Part{N}/images/...` tree (as described in 9.1).
- `Part1/images/` holds HM3D and ScanNet video frames (~63 000 PNG/JPG files).
- `Part2` through `Part6` hold additional task images (small RGB snapshots).

### 9.1 Why paths break after moving the dataset

When the dataset is built locally via `preprocess.py`, image paths are resolved to
**absolute paths** on the builder's machine at line 118 of `utils/preprocess.py`:

```python
processed_sample['images'].extend([str(f) for f in img_files])
```

These absolute paths are stored in the `images` column of the Arrow dataset
(`unieqa_full_multi_v3/train/data-*.arrow`).  When the dataset directory is
copied to another machine, those stored paths point to locations that do not
exist, and `Image.open()` fails with a `FileNotFoundError`.

**The Hugging Face pre-built dataset (`EmbodiedArena/unieqa`) is affected by the
same issue.**  It was built on a specific machine and contains that machine's
absolute paths.  You must configure `image_root` (see 9.2) to use it on any
other system.

The only scenario that does **not** require `image_root` is building the
dataset from scratch on the same machine where it will be evaluated, in which
case the `--data_root` path passed to `preprocess.py` matches the runtime path
exactly.

### 9.2 Runtime path override

You can redirect image paths at runtime without rebuilding the dataset.  In
`unieqa.yaml`, set two entries under `dataset_kwargs`:

| Key | Meaning | Example |
|-----|---------|---------|
| `dataset_path` | Path to the Arrow dataset directory (`unieqa_full_multi_v3`) | `/home/arena/.../unieqa_full_multi_v3` |
| `image_root` | Parent directory that contains `Part1/`, `Part2/`, … — i.e. the `{data_root}` used when running `preprocess.py` | `/home/arena/.../data/unieqa` |

**How it works:** When the framework tries to load an image and finds that the
stored path does not exist on disk, it looks for the first `Part{N}/` anchor in
the path, discards everything before it, and prepends `image_root`.  For example:

```
Stored:    /home/tanghyyy/.../data/unieqa/Part1/images/hm3d-v0/004-hm3d/00099-rgb.png
Anchored:  Part1/images/hm3d-v0/004-hm3d/00099-rgb.png
Resolved:  {image_root}/Part1/images/hm3d-v0/004-hm3d/00099-rgb.png
```

If `image_root` is not set, the original stored path is used unchanged.

```yaml
# Example unieqa.yaml excerpt
dataset_path: /home/arena/embodiedeval/.../embodied_eval/data/unieqa/unieqa_full_multi_v3
dataset_kwargs:
  image_root: /home/arena/embodiedeval/.../embodied_eval/data/unieqa
load_from_disk: true
eval_split: train
```
```
