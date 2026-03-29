# EmbodiedScene Visual QA Benchmark

EmbodiedScene evaluates vision-language models on embodied scene understanding across five question types:

- **object_perception**: Object property and attribute recognition
- **spatial_perception**: Spatial relationship understanding
- **temporal_perception**: Temporal sequence understanding
- **embodied_knowledge**: Embodied common sense knowledge
- **embodied_reasoning**: Embodied reasoning and inference

## Dataset

- **Total samples**: 9675
- **Source**: `embodied_eval/data/EmbodiedScene/visual_qa_dataset.json`
- **Question formats**: Open-ended (`open`) and Multiple-choice (`mcq`)

## Data Preparation

The original dataset file has a nested structure (`{"metadata": ..., "data": [...]}`).
A preprocessed flat JSON file is required for the evaluation framework.
Run the following command to generate it:

```bash
python3 -c "
import json
with open('embodied_eval/data/EmbodiedScene/visual_qa_dataset.json', 'r') as f:
    data = json.load(f)
with open('embodied_eval/data/EmbodiedScene/embodied_scene_data.json', 'w') as f:
    json.dump(data['data'], f, ensure_ascii=False)
"
```

## Evaluation

Use the following task name:
- `embodied-scene`: Evaluate on all question types

Requires `OPENAI_API_KEY` environment variable to be set for LLM-as-judge scoring.

## Metrics

- **LLM-as-judge**: GPT-based scoring (0, 0.5, or 1) for both open-ended and MCQ questions
- Results are reported per question type and as an overall average
