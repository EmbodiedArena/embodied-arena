# Cosmos Reasoning Benchmark

## Overview

Cosmos Reasoning Benchmark is an embodied reasoning evaluation benchmark released by NVIDIA, used to assess vision–language models’ reasoning ability in the physical world. The benchmark covers multiple tasks in robotics and embodied AI.

**Official resources:**

- Paper: [Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning](https://arxiv.org/abs/2503.15558)
- Code: [nvidia-cosmos/cosmos-reason1](https://github.com/nvidia-cosmos/cosmos-reason1)
- Models: [HuggingFace Collection](https://huggingface.co/collections/nvidia/cosmos-reason1-benchmark)

## Dataset description

The Cosmos benchmark has 5 subtasks covering different embodied reasoning scenarios:

| Subtask | Description | # samples | Domain |
|---------|-------------|-----------|--------|
| **BridgeV2** | Robot manipulation; predict the next action | 100 | Robot manipulation |
| **RoboVQA** | Robotic visual QA | 110 | Robot perception |
| **AgibotWorld** | Industrial robot tasks | 100 | Industrial robots |
| **HoloAssist** | First-person human demonstrations | 100 | Human demonstration |
| **RoboFail** | Robot failure detection | 100 | Fault diagnosis |

**Total**: 510 benchmark samples

## Data format

Each sample contains:

- **video**: path to the video file (MP4)
- **qa_pairs**: question–answer pair
  - **question**: question text
  - **index2ans**: option dictionary (A, B, C, D, …)
  - **answer**: correct option letter

Example:

```json
{
    "video": "clips/example.mp4",
    "qa_pairs": {
        "question": "Given what the robot has done, what is the next action?",
        "index2ans": {
            "A": "move forward",
            "B": "close gripper",
            "C": "open gripper",
            "D": "move left"
        },
        "answer": "B"
    }
}
```

## Data preparation

### 1. Download the dataset

The original release does not include some clip files and videos; you need to download and process them yourself.

Processed data is available at: https://drive.google.com/drive/folders/10PwR2RQLPkXiFJqZLpjLT4blIudSQGX6?usp=sharing

The dataset layout should be:

```
embodied_eval/data/cosmos/
├── bridgev2/
│   ├── bridgev2_benchmark_qa_pairs.json
│   └── clips/*.mp4
├── robovqa/
│   ├── robovqa_benchmark_qa_pairs.json
│   └── clips/*.mp4
├── agibot/
│   ├── agibot_benchmark_qa_pairs.json
│   └── clips/*.mp4
├── holoassist/
│   ├── holoassist_benchmark_qa_pairs.json
│   └── clips/*.mp4
└── robofail/
    ├── robofail_benchmark_qa_pairs.json
    └── clips/*.mp4
```

### 2. Verify data

```bash
# Check dataset completeness
ls embodied_eval/data/cosmos/*/clips/*.mp4 | wc -l
# Should print 510 (or a close number)
```

## Evaluation

### Metrics

- **Accuracy**: multiple-choice accuracy (0–1)
- **Per-subtask Accuracy**: accuracy per subtask (0–1)
- **Overall Accuracy**: mean accuracy across subtasks (0–1)

**Website score (0–100 scale)**: Raw scores are in 0–1; they map linearly to percent: **website score = raw score × 100**

### Running evaluation

#### 1. Evaluate a single subtask

```bash
# BridgeV2
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    -m embodied_eval \
    --model qwen2_vl \
    --model_args model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct \
    --evaluator eqa \
    --tasks cosmos-bridgev2 \
    --batch_size 1 \
    --output_path ./logs/cosmos/qwen2_5_vl_7b/bridgev2

# RoboVQA
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    -m embodied_eval \
    --model qwen2_vl \
    --model_args model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct \
    --evaluator eqa \
    --tasks cosmos-robovqa \
    --batch_size 1 \
    --output_path ./logs/cosmos/qwen2_5_vl_7b/robovqa
```

#### 2. Evaluate all subtasks

**Recommended: use the provided script**

```bash
bash embodied_eval/tasks/cosmos/run_eval.sh
```

This script runs all 5 subtasks in order and aggregates results.

**Note:** This framework does **not** support `--tasks cosmos` to run all subtasks in one shot; you must run them one by one (or use the script above).

## Configuration files

### YAML

Each subtask has its own YAML:

- `cosmos-bridgev2.yaml` — BridgeV2 manipulation
- `cosmos-robovqa.yaml` — RoboVQA visual QA
- `cosmos-agibot.yaml` — AgibotWorld industrial robots
- `cosmos-holoassist.yaml` — HoloAssist human demonstration
- `cosmos-robofail.yaml` — RoboFail failure detection

### Processing functions

`process.py` defines:

- `cosmos_doc_to_visual()`: extract video path
- `cosmos_doc_to_text()`: format question and options
- `cosmos_doc_to_target()`: extract target answer
- `cosmos_process_results()`: handle model output
- `cosmos_aggregate_results()`: aggregate metrics
- `parse_letter_response()`: parse letter answers

## Example results

```json
{
  "bridgev2_accuracy": 0.75,
  "robovqa_accuracy": 0.68,
  "agibot_accuracy": 0.72,
  "holoassist_accuracy": 0.70,
  "robofail_accuracy": 0.65,
  "accuracy_average": 0.70,
  "overall_accuracy": 0.70,
  "overall": 0.70
}
```

## Post-processing

To recompute results (e.g. after changing the scoring logic):

```bash
python embodied_eval/tasks/cosmos/process.py \
    --sample_file ./logs/cosmos/model_name/run_id/samples_cosmos-bridgev2.json \
    --results_file ./logs/cosmos/model_name/run_id/results_cosmos-bridgev2.json
```

## Supported models

In principle any video-capable VLM, including:

- Qwen2-VL / Qwen2.5-VL
- InternVL
- LLaVA-Video
- Cosmos-Reason1 (official)
- Other VLMs with video understanding

## FAQ

### Q1: Video file not found

**A**: Check that paths are correct; in YAML, `video_dir` must point to a directory that contains a `clips/` subdirectory.

### Q2: How to change generation settings?

**A**: Edit `generation_kwargs` in the YAML:

```yaml
generation_kwargs:
  max_new_tokens: 512  # increase max tokens
  temperature: 0.1     # adjust temperature
```

### Q3: Batch size

**A**: Video models use a lot of GPU memory; `batch_size=1` is recommended. Increase only if you have enough memory.

## Citation

If you use the Cosmos benchmark, please cite:

```bibtex
@article{cosmos-reason1,
  title={Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning},
  author={NVIDIA Research},
  journal={arXiv preprint arXiv:2503.15558},
  year={2025}
}
```

## License

- Code: Apache 2.0 License
- Data: CC-BY-4.0 License
- Models: NVIDIA Open Model License

## Contact

Questions or suggestions:

- GitHub Issues: [embodied-arena](https://github.com/lichaozhy/tjucs-airank-2025)
- NVIDIA Cosmos: cosmos-license@nvidia.com
