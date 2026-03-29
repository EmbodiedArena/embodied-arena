'''
EmbodiedScene Visual QA Benchmark
Evaluates VLMs on embodied scene understanding tasks:
- object_perception: Object property and attribute recognition
- spatial_perception: Spatial relationship understanding
- temporal_perception: Temporal sequence understanding
- embodied_knowledge: Embodied common sense knowledge
- embodied_reasoning: Embodied reasoning and inference

Implementation Note:
- Open-ended questions use LLM-as-judge for evaluation
- MCQ questions use LLM-as-judge for evaluation
- Both formats follow the same GPT scoring logic as EA-Temporal
'''
import os
import re
import json
import argparse
import numpy as np
import pandas as pd

from collections import defaultdict
from openai import OpenAI
from typing import Optional, List
from loguru import logger as eval_logger
from tqdm import tqdm
from PIL import Image

# Question types for EmbodiedScene
EMBODIED_SCENE_QUESTION_TYPES = [
    "object_perception",
    "spatial_perception",
    "temporal_perception",
    "embodied_knowledge",
    "embodied_reasoning",
]

# Metrics for all question types
METRICS_FOR_EMBODIED_SCENE = {
    "llm_match_score": "llm_match_embodied_scene"
}

# Lazy OpenAI client initialization
_client = None


def get_openai_client():
    """Get OpenAI client with lazy initialization"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before running evaluation: "
                "export OPENAI_API_KEY='your-api-key'"
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=api_base if api_base else None
        )
    return _client


# For backward compatibility
class _ClientProxy:
    def __getattr__(self, name):
        return getattr(get_openai_client(), name)

client = _ClientProxy()


def embodied_scene_doc_to_visual(doc, dataset_kwargs=None):
    """
    Extract and load images from EmbodiedScene document.

    Args:
        doc: Document containing 'images_path'
        dataset_kwargs: Dataset configuration with 'image_dir' key

    Returns:
        List of PIL Image objects
    """
    images_path = doc.get("images_path", [])

    if isinstance(images_path, str):
        images_path = [images_path]

    if not images_path:
        eval_logger.warning(f"No images found for sample {doc.get('question_id', 'unknown')}")
        return []

    image_dir = dataset_kwargs.get("image_dir", "") if dataset_kwargs else ""

    pil_images = []
    for img_path in images_path:
        # Handle relative paths starting with "./"
        img_path = img_path.lstrip("./")
        if image_dir:
            full_path = os.path.join(image_dir, img_path)
        else:
            full_path = img_path

        try:
            if os.path.exists(full_path):
                img = Image.open(full_path).convert('RGB')
                pil_images.append(img)
            else:
                eval_logger.warning(f"Image not found: {full_path}")
        except Exception as e:
            eval_logger.error(f"Error loading image {full_path}: {e}")

    return pil_images


def embodied_scene_doc_to_text(doc, dataset_kwargs=None):
    """
    Extract question text from EmbodiedScene document.
    For MCQ questions, appends the choices to the question.

    Args:
        doc: Document containing 'question' and optionally 'choices'

    Returns:
        Question text string
    """
    question = doc.get("question", "")
    choices = doc.get("choices", [])
    question_format = doc.get("question_format", "open")

    if question_format == "mcq" and choices:
        choices_text = "\n".join(choices)
        return f"{question}\n{choices_text}"
    return question


def embodied_scene_doc_to_target(doc, dataset_kwargs=None):
    """
    Extract target answer from EmbodiedScene document.

    Args:
        doc: Document containing 'answer'

    Returns:
        Target answer as string
    """
    return str(doc.get("answer", ""))


def del_think(text):
    """Remove <think>...</think> tags from model output"""
    if not text:
        return ""
    start_token = "<think>"
    end_token = "</think>"
    pattern = re.escape(start_token) + r'.*?' + re.escape(end_token)
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()


def load_prompt(question_format: str) -> str:
    """Load evaluation prompt for different question formats"""
    if question_format == "open":
        return """Now, you will be presented with a correct response, and a student's answer to an open-ended question about embodied scene understanding. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: 1. If the student's answer correctly captures the key information in the correct answer (e.g., same object, property, or relationship), give it a score of '1'. 2. If the answer is correct but contains additional relevant explanation, assign it a '1'. 3. If the student's answer is partially correct (e.g., correct category but wrong specific attribute), assign a score of '0.5'. 4. If the student provides a related but not fully correct answer, assign a score of '0.5'. 5. If the answer is completely incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(0.5 or 0)', which is your final judgement. Output in JSON format. For instance: '{{"Assessment": "xxxxx", "Final Score": "1(0.5 or 0)"}}'. The correct response and student's answer is provided below.

Correct Answer: {answer}
Student's Answer: {prediction}"""
    elif question_format == "mcq":
        return """Now, you will be presented with a correct response, and a student's answer to a multiple-choice question about embodied scene understanding. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: 1. If the student's answer matches the correct option (e.g., 'A', 'B', 'C', or the full option text), give it a score of '1'. 2. If the answer contains the correct option letter or text along with additional explanation, assign it a '1'. 3. If the student's answer is ambiguous but leans toward the correct option, assign a score of '0.5'. 4. If the answer is completely incorrect or selects a wrong option, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(0.5 or 0)', which is your final judgement. Output in JSON format. For instance: '{{"Assessment": "xxxxx", "Final Score": "1(0.5 or 0)"}}'. The correct response and student's answer is provided below.

Correct Answer: {answer}
Student's Answer: {prediction}"""
    else:
        raise ValueError(f"Unknown question format: {question_format}")


def llm_match_embodied_scene(
    question: str,
    answer: str,
    prediction: str,
    question_format: str = "open",
    openai_model: str = "gpt-4o-mini",
    openai_seed: int = 1234,
    openai_max_tokens: int = 256,
    openai_temperature: float = 0.2,
    max_tries: int = 3,
):
    """LLM evaluation for EmbodiedScene questions"""
    import time

    if prediction is None or not prediction.strip():
        return 0

    prompt = load_prompt(question_format)
    messages = prepare_openai_messages(
        prompt.format(answer=answer, prediction=prediction)
    )

    for attempt in range(max_tries):
        try:
            output = call_openai_api(
                messages=messages,
                model=openai_model,
                seed=openai_seed,
                max_tokens=openai_max_tokens,
                temperature=openai_temperature,
            )
            return parse_score_json(output)
        except Exception as e:
            if attempt < max_tries - 1:
                eval_logger.warning(f"LLM evaluation failed (attempt {attempt + 1}/{max_tries}): {e}")
                time.sleep(1)
            else:
                eval_logger.error(f"LLM evaluation failed after {max_tries} attempts: {e}")
                return 0


def prepare_openai_messages(content: str):
    """Prepare messages for OpenAI API"""
    return [{"role": "user", "content": content}]


def call_openai_api(
    messages: list,
    model: str = "gpt-4o-mini",
    seed: int = None,
    max_tokens: int = 256,
    temperature: float = 0.2,
    verbose: bool = False,
):
    """Call OpenAI API with given parameters"""
    client = get_openai_client()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print("OpenAI API response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content


def parse_score_json(output: str) -> float:
    """Parse score from JSON output format"""
    if not output:
        return 0.0

    try:
        data = json.loads(output)
        score_str = data.get("Final Score", "0")

        if isinstance(score_str, str):
            match = re.search(r'(1|0\.5|0)', score_str)
            if match:
                return float(match.group(1))
        elif isinstance(score_str, (int, float)):
            return float(score_str)
    except json.JSONDecodeError:
        match = re.search(r'"Final Score":\s*"?(1|0\.5|0)"?', output)
        if match:
            return float(match.group(1))

    return 0.0


def embodied_scene_process_results(doc, results, dataset_kwargs=None):
    """
    Process model prediction results and compute evaluation metrics.

    Args:
        doc: Original document
        results: Model prediction results (list of strings)
        dataset_kwargs: Dataset configuration

    Returns:
        Dictionary containing evaluation results
    """
    raw_prediction = del_think(results[0]) if results else ""
    doc["prediction"] = raw_prediction

    target = str(doc.get("answer", ""))
    question = embodied_scene_doc_to_text(doc)
    question_type = doc.get("question_type", "object_perception")
    question_format = doc.get("question_format", "open")

    result_dict = {
        "target": target,
        "question_type": question_type,
        "question_format": question_format,
    }

    score = llm_match_embodied_scene(
        question=question,
        answer=target,
        prediction=raw_prediction,
        question_format=question_format,
    )
    doc["llm_match_score"] = score
    result_dict["llm_match_score"] = score

    print(doc)
    return result_dict


def embodied_scene_aggregate_results(results):
    """
    Aggregate evaluation results by question type.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary containing aggregated metrics
    """
    if not results:
        return {}

    for r in results:
        assert "question_type" in r, r

    results_df = pd.DataFrame(results)
    output = {}

    for question_type, indices in results_df.groupby("question_type").groups.items():
        per_type = results_df.iloc[indices]
        if "llm_match_score" in per_type.columns:
            metric_data = per_type["llm_match_score"].tolist()
            if metric_data and isinstance(metric_data[0], dict):
                avg_score = np.mean([x["llm_match_score"] for x in metric_data])
            else:
                avg_score = np.mean(metric_data)
            output[f"{question_type}_llm_match_score"] = avg_score

    # Compute per-metric averages across question types
    # Key format: "{question_type}_llm_match_score" -> metric suffix is "llm_match_score"
    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if isinstance(val, (float, int)):
            # Match keys like "object_perception_llm_match_score"
            for qtype in EMBODIED_SCENE_QUESTION_TYPES:
                if key.startswith(qtype + "_"):
                    metric_name = key[len(qtype) + 1:]
                    metric_to_values[metric_name].append(val)
                    break

    for metric_name, vals in metric_to_values.items():
        if vals:
            output[f"{metric_name}_average"] = sum(vals) / len(vals)

    # overall = average of per-question-type scores only (exclude summary keys)
    per_type_vals = [v for k, v in output.items()
                     if any(k.startswith(qt + "_") for qt in EMBODIED_SCENE_QUESTION_TYPES)]
    if per_type_vals:
        output["overall"] = sum(per_type_vals) / len(per_type_vals)

    eval_logger.info(f"EmbodiedScene Evaluation Results: {output}")
    return output


def _load_dataset_index(dataset_path: Optional[str] = None) -> dict:
    """
    Load visual_qa_dataset.json and build a doc_id -> metadata mapping.
    Returns empty dict if dataset not found.
    """
    if dataset_path is None:
        # Try to find the dataset relative to this file or common locations
        this_dir = os.path.dirname(os.path.abspath(__file__))
        home_dir = os.path.expanduser("~")
        candidates = [
            os.path.join(this_dir, "../../data/EmbodiedScene/visual_qa_dataset.json"),
            os.path.join(this_dir, "../../../data/EmbodiedScene/visual_qa_dataset.json"),
            os.path.join(this_dir, "../../../../data/EmbodiedScene/visual_qa_dataset.json"),
            os.path.join(home_dir, "data/EmbodiedScene/visual_qa_dataset.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                dataset_path = os.path.normpath(c)
                break

    if not dataset_path or not os.path.exists(dataset_path):
        eval_logger.warning("visual_qa_dataset.json not found; question_type will default to 'object_perception'")
        return {}

    eval_logger.info(f"Loading dataset index from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    index = {}
    for i, item in enumerate(raw.get("data", [])):
        index[i] = {
            "question_type": item.get("question_type", "object_perception"),
            "question_format": item.get("question_format", "open"),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
        }
    eval_logger.info(f"Dataset index built: {len(index)} entries")
    return index


def post_evaluate_results(
    sample_file_path: str,
    results_file_path: str,
    openai_model: str = "gpt-4o-mini",
    dataset_path: Optional[str] = None,
):
    """
    Post-process evaluation results from saved samples.

    Args:
        sample_file_path: Path to samples JSON file (jsonl format)
        results_file_path: Path to save results JSON file
        openai_model: OpenAI model for LLM evaluation
        dataset_path: Path to visual_qa_dataset.json for question_type lookup
    """
    if not os.path.exists(sample_file_path):
        eval_logger.error(f"Sample file not found: {sample_file_path}")
        return

    data = []
    with open(sample_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    eval_logger.info(f"Loaded {len(data)} samples from {sample_file_path}")

    # Build doc_id -> metadata index from the original dataset
    dataset_index = _load_dataset_index(dataset_path)

    results = []
    for doc in tqdm(data, desc="Re-evaluating EmbodiedScene samples"):
        pred_raw = ""
        if isinstance(doc.get("resps"), list) and doc["resps"] and doc["resps"][0]:
            pred_raw = doc["resps"][0][0]
        elif doc.get("prediction"):
            pred_raw = doc["prediction"]

        pred_raw = del_think(pred_raw)

        # Look up metadata from dataset index using doc_id
        doc_id = doc.get("doc_id")
        meta = dataset_index.get(doc_id, {}) if doc_id is not None else {}

        target = str(doc.get("target", doc.get("answer", meta.get("answer", ""))))
        question = doc.get("question", meta.get("question", str(doc.get("doc", ""))))
        question_type = doc.get("question_type", meta.get("question_type", "object_perception"))
        question_format = doc.get("question_format", meta.get("question_format", "open"))

        score = llm_match_embodied_scene(
            question=question,
            answer=target,
            prediction=pred_raw,
            question_format=question_format,
            openai_model=openai_model,
        )
        doc["llm_match_score"] = score

        result_dict = {
            "target": target,
            "question_type": question_type,
            "question_format": question_format,
            "llm_match_score": score,
        }
        results.append(result_dict)

    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    eval_logger.info(f"Updated samples saved to {sample_file_path}")

    output = embodied_scene_aggregate_results(results)
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    eval_logger.info(f"Aggregated results saved to {results_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-evaluate EmbodiedScene results from saved samples."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Log directory, e.g., .../logs/embodied_scene/<model>/<run_id>/",
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        default=None,
        help="Path to samples JSON file",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for LLM evaluation",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to visual_qa_dataset.json for question_type lookup (auto-detected if not set)",
    )
    args = parser.parse_args()

    sample_path = args.sample_file
    results_path = args.results_file

    if args.base_dir and (not sample_path or not results_path):
        base_dir = args.base_dir.rstrip("/")
        if not sample_path:
            # 优先读 inference_only 模式保存的文件，其次读完整评估保存的 samples 文件
            inference_path = f"{base_dir}/inference_embodied-scene.json"
            samples_path = f"{base_dir}/samples_embodied-scene.json"
            if os.path.exists(inference_path):
                sample_path = inference_path
            else:
                sample_path = samples_path
        if not results_path:
            results_path = f"{base_dir}/results_embodied-scene.json"

    if not sample_path or not results_path:
        raise ValueError(
            "You must provide --base_dir or both --sample_file and --results_file."
        )

    post_evaluate_results(
        sample_file_path=sample_path,
        results_file_path=results_path,
        openai_model=args.openai_model,
        dataset_path=args.dataset_path,
    )
