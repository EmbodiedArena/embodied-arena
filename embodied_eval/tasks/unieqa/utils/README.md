# UniEQA utils

This directory contains utility scripts for managing the UniEQA dataset and related workflows.

## Files

### `download.py`

Dataset download and integrity checks.

**CLI:**

```bash
# Hugging Face variant (images only; quick tests)
python -m embodied_eval.tasks.unieqa.utils.download --auto_download --download_type huggingface --output_path /path/to/UniEQA

# HM3D pack (~12 GB)
python -m embodied_eval.tasks.unieqa.utils.download --auto_download --download_type hm3d --output_path /path/to/UniEQA

# Verify dataset integrity only
python -m embodied_eval.tasks.unieqa.utils.download --check_only --output_path /path/to/UniEQA
```

**Download types:**

- `huggingface` — image data from Hugging Face (fast; good for smoke tests)
- `full` — full release (Google Drive; manual download required)
- `hm3d` — HM3D assets (~12 GB)
- `scannet` — ScanNet (manual download)

**Features:**

- Multiple data sources
- Automatic integrity checks
- Dependency handling
- Resumable downloads

## Data formats

### Hugging Face variant

- `image`: PIL `Image` objects

### Full on-disk format

`data.json` shape:

```
{
  "data": [{
    "sample_id": int,
    "task_instruction_id": int,
    "task_instance": {
      "context": str,           # question text
      "images_path": [str]      # list of image paths
    },
    "response": str             # answer text
  }],
  "metadata": {
    "task_instruction": [str]   # task instruction strings
  }
}
```

## Notes

- The full release includes HM3D (~12 GB) and ScanNet (~62–70 GB).
- Google Drive downloads may require manual authentication.
- ScanNet frame extraction typically takes about 8–10 hours.
- The Hugging Face image-only path is supported for quick testing.
