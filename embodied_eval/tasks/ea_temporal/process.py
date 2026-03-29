'''
EA-Temporal Benchmark for temporal perception evaluation
Evaluates VLMs on two types of temporal perception tasks:
- Type1: Action Recognition
- Type2: Temporal Ordering

Implementation Note:
- Both Type1 and Type2 use LLM-as-judge for evaluation
- Type1: Evaluates semantic similarity for action recognition
- Type2: Evaluates correctness of temporal sequence ordering
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

# Question types for EA-Temporal
EA_TEMPORAL_QUESTION_TYPES = [
    "type1",  # Action recognition
    "type2",  # Chronological ordering
]

# Metrics for different task types
METRICS_FOR_TYPE1 = {
    "llm_match_score": "llm_match_type1"
}

METRICS_FOR_TYPE2 = {
    "llm_match_score": "llm_match_type2"
}

# Lazy OpenAI client initialization
_client = None


def ea_temporal_preprocess_doc(doc):
    """
    Preprocess document before Dataset.from_list() to ensure consistent types.
    Converts list responses to JSON strings to avoid PyArrow type errors.
    
    Args:
        doc: Document dictionary
    
    Returns:
        Preprocessed document
    """
    if isinstance(doc.get("response"), list):
        # Convert list to JSON string, store original type indicator
        doc["response"] = json.dumps(doc["response"])
        doc["_response_is_list"] = True
    else:
        doc["_response_is_list"] = False
    return doc


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


def ea_temporal_doc_to_visual(doc, dataset_kwargs=None):
    """
    Extract and load images from EA-Temporal document.
    
    Args:
        doc: Document containing 'task_instance' with 'images_path'
        dataset_kwargs: Dataset configuration with 'image_dir' key
    
    Returns:
        List of PIL Image objects
    """
    task_instance = doc.get("task_instance", {})
    images_path = task_instance.get("images_path", [])
    
    if isinstance(images_path, str):
        images_path = [images_path]
    
    if not images_path:
        eval_logger.warning(f"No images found for sample {doc.get('sample_id', 'unknown')}")
        return []
    
    # Construct full paths if image_dir is provided
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
                # Load image as PIL Image object
                img = Image.open(full_path).convert('RGB')
                pil_images.append(img)
            else:
                eval_logger.warning(f"Image not found: {full_path}")
        except Exception as e:
            eval_logger.error(f"Error loading image {full_path}: {e}")
    
    return pil_images


def ea_temporal_doc_to_text(doc, dataset_kwargs=None):
    """
    Extract question text from EA-Temporal document.
    
    Args:
        doc: Document containing 'task_instance' with 'context'
    
    Returns:
        Question text string
    """
    task_instance = doc.get("task_instance", {})
    context = task_instance.get("context", "")
    return context


def ea_temporal_doc_to_target(doc, dataset_kwargs=None):
    """
    Extract target answer from EA-Temporal document.
    
    Args:
        doc: Document containing 'response'
    
    Returns:
        Target response as string
    """
    response = doc.get("response", "")
    
    # Handle JSON string from preprocessing (originally a list)
    if doc.get("_response_is_list", False) and isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            pass
    
    if isinstance(response, list):
        # Type2: Convert list to comma-separated string
        return ", ".join(map(str, response))
    return str(response)


def del_think(text):
    """Remove <think>...</think> tags from model output"""
    if not text:
        return ""
    start_token = "<think>"
    end_token = "</think>"
    pattern = re.escape(start_token) + r'.*?' + re.escape(end_token)
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()


def load_prompt(prompt_type: str) -> str:
    """Load evaluation prompt for different task types"""
    if prompt_type == "type1":
        return """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to action recognition from sequential images. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: 1. If the student's answer correctly identifies the action semantically (e.g., 'picking up' vs 'pick up'), give it a score of '1'. 2. If the answer is correct but contains additional relevant explanation about the action, assign it a '1'. 3. If the student identifies the correct action type but with a different object (e.g., 'pick up the spoon' vs 'pick up the ladle'), and the objects are similar in appearance or function, assign a score of '0.5'. 4. If the student identifies a related but different action (e.g., 'reaching for' vs 'pick up'), assign a score of '0.5'. 5. If the answer is completely incorrect, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(0.5 or 0)', which is your final judgement. Output in JSON format. For instance: '{{"Assessment": "xxxxx", "Final Score": "1(0.5 or 0)"}}'. The correct response and student's answer is provided below.

Correct Answer: {answer}
Student's Answer: {prediction}"""
    elif prompt_type == "type2":
        return """Now, you will be presented with a correct response, and a student's answer to an open-ended question related to temporal ordering of sequential images. The task is to identify the correct chronological order of shuffled images. Your job is to compare the student's answer to the correct one and assign a score based on the following rules: 1. If the student's answer matches the correct sequence exactly (e.g., '2, 1, 3' or 'B, A, C'), give it a score of '1'. 2. If the answer is correct but uses different notation (e.g., 'second, first, third' vs '2, 1, 3'), assign it a '1'. 3. If the student gets the majority of the sequence correct but makes minor errors (e.g., correct first and last but wrong middle), assign a score of '0.5'. 4. If the student provides a partially correct sequence or describes the temporal relationship correctly but not in the exact format, assign a score of '0.5'. 5. If the answer is completely incorrect or reversed, give it a '0'. Begin your evaluation with an 'Assessment:' paragraph, where you elaborate on your thought process. Conclude with 'Final Score: 1(0.5 or 0)', which is your final judgement. Output in JSON format. For instance: '{{"Assessment": "xxxxx", "Final Score": "1(0.5 or 0)"}}'. The correct response and student's answer is provided below.

Correct Answer: {answer}
Student's Answer: {prediction}"""
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def llm_match_type1(
    question: str,
    answer: str,
    prediction: str,
    openai_model: str = "gpt-4o-mini",
    openai_seed: int = 1234,
    openai_max_tokens: int = 256,
    openai_temperature: float = 0.2,
    max_tries: int = 3,
):
    """LLM evaluation for type1 (action recognition)"""
    import time
    
    if prediction is None or not prediction.strip():
        return 0
    
    prompt = load_prompt("type1")
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


def llm_match_type2(
    question: str,
    answer: str,
    prediction: str,
    openai_model: str = "gpt-4o-mini",
    openai_seed: int = 1234,
    openai_max_tokens: int = 256,
    openai_temperature: float = 0.2,
    max_tries: int = 3,
):
    """LLM evaluation for type2 (temporal ordering)"""
    import time
    
    if prediction is None or not prediction.strip():
        return 0
    
    prompt = load_prompt("type2")
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
    max_tokens: int = 128,
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
        # Try to parse as JSON
        data = json.loads(output)
        score_str = data.get("Final Score", "0")
        
        # Extract numeric score from "1(0.5 or 0)" format
        if isinstance(score_str, str):
            # Match patterns like "1", "0.5", "0"
            match = re.search(r'(1|0\.5|0)', score_str)
            if match:
                return float(match.group(1))
        elif isinstance(score_str, (int, float)):
            return float(score_str)
    except json.JSONDecodeError:
        # Fallback: try to extract score directly from text
        match = re.search(r'"Final Score":\s*"?(1|0\.5|0)"?', output)
        if match:
            return float(match.group(1))
    
    return 0.0


def ea_temporal_process_results(doc, results, dataset_kwargs=None):
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
    
    target = doc.get("response", "")
    question = ea_temporal_doc_to_text(doc)
    
    # Parse JSON string back to list if it was converted during preprocessing
    if doc.get("_response_is_list", False) and isinstance(target, str):
        try:
            target = json.loads(target)
        except json.JSONDecodeError:
            pass
    
    # Determine task type based on response format
    if isinstance(target, list):
        question_type = "type2"
    else:
        question_type = "type1"
    
    result_dict = {"target": target if isinstance(target, str) else str(target)}
    result_dict["question_type"] = question_type
    
    if question_type == "type1":
        # Action recognition: use LLM match
        for key, value in METRICS_FOR_TYPE1.items():
            score = eval(value)(question, str(target), raw_prediction)
            doc[key] = score
            result_dict[key] = score
    else:
        # Chronological ordering: use LLM match
        target_str = ", ".join(map(str, target)) if isinstance(target, list) else str(target)
        for key, value in METRICS_FOR_TYPE2.items():
            score = eval(value)(question, target_str, raw_prediction)
            doc[key] = score
            result_dict[key] = score
    print(doc)
    return result_dict


def ea_temporal_aggregate_results(results):
    """
    Aggregate evaluation results by task type.
    
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
    
    # Aggregate by question type
    for question_type, indices in results_df.groupby("question_type").groups.items():
        per_type = results_df.iloc[indices]
        
        if question_type == "type1":
            metrics = METRICS_FOR_TYPE1
        else:
            metrics = METRICS_FOR_TYPE2
        
        for metric in metrics.keys():
            if metric in per_type.columns:
                metric_data = per_type[metric].tolist()
                # Handle both nested dict format (old) and direct value format (new)
                if metric_data and isinstance(metric_data[0], dict):
                    avg_score = np.mean([x[metric] for x in metric_data])
                else:
                    avg_score = np.mean(metric_data)
                output[f"{question_type}_{metric}"] = avg_score
    
    # Compute overall average across all metrics
    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if "_" in key and isinstance(val, (float, int)):
            qtype, metric_name = key.rsplit("_", 1)
            metric_to_values[metric_name].append(val)
    
    for metric_name, vals in metric_to_values.items():
        if vals:
            output[f"{metric_name}_average"] = sum(vals) / len(vals)
    
    # Overall score
    if output:
        output["overall"] = sum(output.values()) / len(output)
    
    eval_logger.info(f"EA-Temporal Evaluation Results: {output}")
    return output


def post_evaluate_results(
    sample_file_path: str,
    results_file_path: str,
    openai_model: str = "gpt-4o-mini",
):
    """
    Post-process evaluation results from saved samples.
    
    Args:
        sample_file_path: Path to samples JSON file (jsonl format)
        results_file_path: Path to save results JSON file
        openai_model: OpenAI model for LLM evaluation
    """
    if not os.path.exists(sample_file_path):
        eval_logger.error(f"Sample file not found: {sample_file_path}")
        return
    
    # Read samples
    data = []
    with open(sample_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    eval_logger.info(f"Loaded {len(data)} samples from {sample_file_path}")
    
    results = []
    for doc in tqdm(data, desc="Re-evaluating EA-Temporal samples"):
        # Extract prediction
        pred_raw = ""
        if isinstance(doc.get("resps"), list) and doc["resps"] and doc["resps"][0]:
            pred_raw = doc["resps"][0][0]
        elif doc.get("prediction"):
            pred_raw = doc["prediction"]
        
        pred_raw = del_think(pred_raw)
        
        target = doc.get("target", doc.get("response", ""))
        question = doc.get("question", doc.get("task_instance", {}).get("context", ""))
        question_type = doc.get("question_type", "type1")
        
        result_dict = {"target": str(target), "question_type": question_type}
        
        if question_type == "type1":
            for key, value in METRICS_FOR_TYPE1.items():
                score = eval(value)(question, str(target), pred_raw)
                doc[key] = score
                result_dict[key] = score
        else:
            target_str = ", ".join(map(str, target)) if isinstance(target, list) else str(target)
            for key, value in METRICS_FOR_TYPE2.items():
                score = eval(value)(question, target_str, pred_raw)
                doc[key] = score
                result_dict[key] = score
        
        results.append(result_dict)
    
    # Save updated samples
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    eval_logger.info(f"Updated samples saved to {sample_file_path}")
    
    # Save aggregated results
    output = ea_temporal_aggregate_results(results)
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    eval_logger.info(f"Aggregated results saved to {results_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-evaluate EA-Temporal results from saved samples."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Log directory, e.g., .../logs/ea_temporal/<model>/<run_id>/",
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
    args = parser.parse_args()

    sample_path = args.sample_file
    results_path = args.results_file

    if args.base_dir and (not sample_path or not results_path):
        base_dir = args.base_dir.rstrip("/")
        if not sample_path:
            sample_path = f"{base_dir}/samples_ea-temporal.json"
        if not results_path:
            results_path = f"{base_dir}/results_ea-temporal.json"

    if not sample_path or not results_path:
        raise ValueError(
            "You must provide --base_dir or both --sample_file and --results_file."
        )

    post_evaluate_results(
        sample_file_path=sample_path,
        results_file_path=results_path,
        openai_model=args.openai_model,
    )
