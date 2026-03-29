'''
OST-Bench benchmark adaptation for embodied-eval framework
Official Repo: https://github.com/OpenRobotLab/OST-Bench
Paper: https://arxiv.org/pdf/2507.07984

Implementation Note:
This implementation maintains dialogue history by grouping samples by scan_id
and processing them sequentially to ensure multi-turn context is preserved.
'''

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, List, Dict
from loguru import logger as eval_logger
from tqdm import tqdm

# Global cache for dialogue history management
# Pre-computed mapping: {f"{scan_id}_turn_{turn_id}": [all_historical_images]}
_HISTORY_MAP = None

# Question types for OST-Bench
OST_BENCH_QUESTION_TYPES = [
    "Agent_visible_info-existence(Judgement)",
    "Agent_visible_info-location(Description)",
    "Agent_visible_info-attribute(Description)",
    "Agent_visible_info-count(Description)",
    "Agent_visible_info-temporal(Description)",
    "Agent_visible_info-navigation(Description)",
    "Agent_visible_info-existence(Recall)",
    "Agent_visible_info-location(Recall)",
    "Agent_visible_info-attribute(Recall)",
    "Agent_visible_info-count(Recall)",
    "Agent_visible_info-temporal(Recall)",
    "Agent_visible_info-navigation(Recall)"
]

# Metrics for OST-Bench evaluation
METRICS_FOR_OST_BENCH = {
    "ost_bench_score": "compute_ost_bench_score"
}

# Number word mapping for enumeration questions
NUM_MAPPING = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
}


# Note: History map not needed - original OST-Bench passes only current turn images
# Multi-turn context is maintained by the model's conversation history, not by accumulating images


def ost_bench_doc_to_visual(doc, dataset_kwargs=None):
    """
    Extract visual data paths from OST-Bench document.
    
    Returns ONLY the current turn's images (typically 5 images per turn).
    This matches the original OST-Bench implementation where:
    - Each turn passes only its own images to the model
    - Historical context is maintained through the model's conversation history
    - NOT by accumulating images across turns
    
    Args:
        doc: Document containing 'new_observations' field
        dataset_kwargs: Dataset configuration containing 'image_dir'

    Returns:
        List of image file paths for the CURRENT turn only
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    image_dir = dataset_kwargs.get("image_dir", "/data/arena/datasets/OST-Bench/imgs")
    new_observations = doc.get("new_observations", [])
    
    # Return only current turn's images (matching original implementation)
    full_paths = []
    for obs_path in new_observations:
        full_path = os.path.join(image_dir, obs_path)
        if os.path.exists(full_path):
            full_paths.append(full_path)
        else:
            eval_logger.warning(f"Image file not found: {full_path}")
    
    if not full_paths:
        eval_logger.warning(f"No valid images for {doc.get('scan_id', 'unknown')} turn {doc.get('turn_id', '?')}")
    
    return full_paths


def ost_bench_doc_to_text(doc, dataset_kwargs=None):
    """
    Extract question text from OST-Bench document.

    Args:
        doc: Document containing question information

    Returns:
        Formatted question string
    """
    # Use user_message if available (formatted prompt), otherwise use origin_question
    if "user_message" in doc:
        return doc["user_message"]
    else:
        return doc.get("origin_question", "")


def ost_bench_doc_to_target(doc, dataset_kwargs=None):
    """
    Extract target answer from OST-Bench document.

    Args:
        doc: Document containing answer information

    Returns:
        Target answer string
    """
    return doc.get("answer", "")


def parse_ost_bench_prediction(prediction):
    """
    Parse the model's prediction to extract the answer from JSON format.

    Args:
        prediction: Raw model prediction string

    Returns:
        Extracted answer string, or original prediction if parsing fails
    """
    if not prediction:
        return ""

    # Clean the prediction string
    pred_str = prediction.strip()

    # Remove markdown code blocks if present
    import re
    pred_str = re.sub(r'```\w*\n?', '', pred_str).strip()

    try:
        # Try to parse as JSON
        parsed = json.loads(pred_str)
        if isinstance(parsed, dict) and "answer" in parsed:
            return str(parsed["answer"])
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract answer from various formats

    # Look for "answer": "value" pattern
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', pred_str, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1)

    # Look for "answer": value pattern (without quotes)
    answer_match = re.search(r'"answer"\s*:\s*([^,\}\n]+)', pred_str, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip().strip('"')

    # If all parsing fails, return the original prediction
    return prediction


def ost_bench_process_results(doc, results, dataset_kwargs=None):
    """
    Process model prediction results and compute evaluation metrics.

    Args:
        doc: Original document
        results: Model prediction results
        dataset_kwargs: Dataset configuration

    Returns:
        Dictionary containing evaluation results
    """
    raw_prediction = results[0] if results else ""
    doc["prediction"] = raw_prediction

    # Parse the prediction to extract the answer
    prediction = parse_ost_bench_prediction(raw_prediction)

    # Get ground truth answer
    target = ost_bench_doc_to_target(doc, dataset_kwargs)

    # Get question type for categorization
    question_type = doc.get("type", "unknown")

    result_dict = {"target": target}
    result_dict["question_type"] = question_type

    # Compute metrics
    for key, value in METRICS_FOR_OST_BENCH.items():
        score = eval(value)(doc, target, prediction)
        doc[key] = {key: score}
        result_dict[key] = doc[key]

    return result_dict


def ost_bench_aggregate_results(results):
    """
    Aggregate evaluation results following the original OST-Bench methodology.
    Computes composite metrics across different question categories and formats.

    Args:
        results: List of result dictionaries containing question_type and scores

    Returns:
        Dictionary containing:
        - Per-question-type scores
        - Composite category scores (A_state, A_info, AO)
        - Format-based scores (Judgement, Estimation, Temporal-loc, Counting)
        - Overall score
    """
    if not results:
        return {}

    results_df = pd.DataFrame(results)
    output = {}
    static_results = {}

    # Compute per-question-type metrics
    if "question_type" in results_df.columns:
        for question_type, indices in results_df.groupby("question_type").groups.items():
            if question_type == "None" or question_type is None:
                continue
            
            per_type = results_df.iloc[indices]
            metric_name = "ost_bench_score"
            
            if metric_name in per_type.columns:
                metric_data = per_type[metric_name].tolist()
                avg_score = np.mean([x[metric_name] for x in metric_data])
                static_results[question_type] = avg_score
                output[f"{question_type}"] = avg_score

    # Compute composite scores following original OST-Bench logic
    # Handle multi-part judgement questions by averaging
    if 'Agent_object_spatial-direction(Judgement1)' in static_results:
        output['Agent_object_spatial-direction(Judgement)'] = np.mean([
            static_results.get(f'Agent_object_spatial-direction(Judgement{i})', 0) 
            for i in range(1, 4)
        ])
    
    if 'Agent_object_spatial-distance(Judgement1)' in static_results:
        output['Agent_object_spatial-distance(Judgement)'] = np.mean([
            static_results.get(f'Agent_object_spatial-distance(Judgement{i})', 0) 
            for i in range(1, 4)
        ])
    
    if 'Agent_visible_info-existence(Temporal-loc1)' in static_results:
        output['Agent_visible_info-existence(Temporal-loc)'] = np.mean([
            static_results.get('Agent_visible_info-existence(Temporal-loc1)', 0),
            static_results.get('Agent_visible_info-existence(Temporal-loc2)', 0)
        ])

    # Agent State scores
    a_state_judge_keys = ['Agent_state-orientation(Judgement)', 'Agent_state-position(Judgement)']
    a_state_esti_keys = ['Agent_state-orientation(Estimation)', 'Agent_state-position(Estimation)']
    
    if all(k in static_results for k in a_state_judge_keys):
        output['A_state(Judge)'] = np.mean([static_results[k] for k in a_state_judge_keys])
    if all(k in static_results for k in a_state_esti_keys):
        output['A_state(Esti)'] = np.mean([static_results[k] for k in a_state_esti_keys])
    if all(k in static_results for k in a_state_judge_keys + a_state_esti_keys):
        output['A_state'] = np.mean([static_results[k] for k in a_state_judge_keys + a_state_esti_keys])

    # Agent Visible Info scores
    a_info_judge_keys = [
        'Agent_visible_info-existence(Judgement)',
        'Agent_visible_info-order(Judgement)',
        'Agent_visible_info-diversity(Judgement)'
    ]
    if all(k in static_results for k in a_info_judge_keys):
        output['A_info(Judge)'] = np.mean([static_results[k] for k in a_info_judge_keys])
    
    if 'Agent_visible_info-existence(Temporal-loc)' in output:
        output['A_info(Temp)'] = output['Agent_visible_info-existence(Temporal-loc)']
    elif 'Agent_visible_info-existence(Temporal-loc1)' in static_results:
        output['A_info(Temp)'] = np.mean([
            static_results.get('Agent_visible_info-existence(Temporal-loc1)', 0),
            static_results.get('Agent_visible_info-existence(Temporal-loc2)', 0)
        ])
    
    if 'Agent_visible_info-quantity(Counting)' in static_results:
        output['A_info(Count)'] = static_results['Agent_visible_info-quantity(Counting)']
    
    a_info_all_keys = a_info_judge_keys + ['Agent_visible_info-quantity(Counting)']
    if 'A_info(Temp)' in output:
        if all(k in static_results for k in a_info_judge_keys + ['Agent_visible_info-quantity(Counting)']):
            output['A_info'] = (output['A_info(Judge)'] + output['A_info(Temp)'] + output['A_info(Count)']) / 3.0

    # Agent-Object Spatial scores
    ao_judge_keys = ['Agent_object_spatial-direction(Judgement)', 'Agent_object_spatial-distance(Judgement)']
    ao_esti_keys = ['Agent_object_spatial-direction(Estimation)', 'Agent_object_spatial-distance(Estimation)']
    ao_temp_keys = ['Agent_object_spatial-direction(Temporal-loc)', 'Agent_object_spatial-distance(Temporal-loc)']
    
    if all(k in output or k in static_results for k in ao_judge_keys):
        vals = [output.get(k, static_results.get(k, 0)) for k in ao_judge_keys]
        output['AO(Judge)'] = np.mean(vals)
    
    if all(k in static_results for k in ao_esti_keys):
        output['AO(Esti)'] = np.mean([static_results[k] for k in ao_esti_keys])
    
    if all(k in static_results for k in ao_temp_keys):
        output['AO(Temp)'] = np.mean([static_results[k] for k in ao_temp_keys])
    
    if 'AO(Judge)' in output and 'AO(Esti)' in output and 'AO(Temp)' in output:
        output['AO'] = np.mean([output['AO(Judge)'], output['AO(Esti)'], output['AO(Temp)']])

    # Format-based scores
    judgement_keys = [
        'Agent_object_spatial-direction(Judgement)', 'Agent_object_spatial-distance(Judgement)',
        'Agent_visible_info-existence(Judgement)', 'Agent_visible_info-order(Judgement)',
        'Agent_visible_info-diversity(Judgement)', 'Agent_state-orientation(Judgement)',
        'Agent_state-position(Judgement)'
    ]
    valid_judgement = [output.get(k, static_results.get(k)) for k in judgement_keys 
                       if k in output or k in static_results]
    if valid_judgement:
        output['Judgement'] = np.mean(valid_judgement)

    estimation_keys = [
        'Agent_object_spatial-direction(Estimation)', 'Agent_object_spatial-distance(Estimation)',
        'Agent_state-orientation(Estimation)', 'Agent_state-position(Estimation)'
    ]
    valid_estimation = [static_results[k] for k in estimation_keys if k in static_results]
    if valid_estimation:
        output['Estimation'] = np.mean(valid_estimation)

    temporal_keys = [
        'Agent_object_spatial-direction(Temporal-loc)', 'Agent_object_spatial-distance(Temporal-loc)'
    ]
    if 'Agent_visible_info-existence(Temporal-loc)' in output:
        temporal_keys.append('Agent_visible_info-existence(Temporal-loc)')
    valid_temporal = [output.get(k, static_results.get(k)) for k in temporal_keys 
                     if k in output or k in static_results]
    if valid_temporal:
        output['Temporal-loc'] = np.mean(valid_temporal)

    if 'Agent_visible_info-quantity(Counting)' in static_results:
        output['Counting'] = static_results['Agent_visible_info-quantity(Counting)']

    # Overall score (average of 8 composite metrics)
    composite_keys = [
        'A_state(Judge)', 'A_state(Esti)', 'A_info(Judge)', 'A_info(Temp)',
        'A_info(Count)', 'AO(Judge)', 'AO(Esti)', 'AO(Temp)'
    ]
    valid_composites = [output[k] for k in composite_keys if k in output]
    if valid_composites:
        output['Overall'] = np.mean(valid_composites)

    eval_logger.info(f"OST-Bench Evaluation Results: {output}")
    return output


def longest_common_subsequence(str1, str2):
    """
    Compute the length of longest common subsequence between two strings.
    
    Args:
        str1: First string
        str2: Second string
    
    Returns:
        int: Length of LCS
    """
    m, n = len(str1), len(str2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_judgement_score(doc, target, prediction):
    """
    Evaluate judgement questions using longest common subsequence matching.
    This matches the original OST-Bench evaluation method.

    Args:
        doc: Document containing 'option' field with answer choices
        target: Ground truth answer
        prediction: Model prediction

    Returns:
        float: 1.0 if best-matching option equals ground truth, 0.0 otherwise
    """
    if prediction is None or target is None:
        return 0.0
    
    options = doc.get("option", [])
    if not options:
        # Fallback to exact match if no options provided
        pred_normalized = str(prediction).strip().lower()
        target_normalized = str(target).strip().lower()
        return 1.0 if pred_normalized == target_normalized else 0.0
    
    # Remove quotes from target and ensure it's in options
    target = str(target).replace('"', '').replace("'", '')
    prediction = str(prediction)
    
    # Find option with highest LCS score
    option_scores = [
        longest_common_subsequence(prediction.strip().lower(), option.strip().lower()) 
        for option in options
    ]
    option_idx = np.argmax(option_scores)
    
    return 1.0 if target == options[option_idx] else 0.0


def compute_estimation_score(doc, target, prediction):
    """
    Evaluate numerical estimation answers using graduated accuracy scoring.
    This follows the VSI-bench methodology used in original OST-Bench.
    
    Scoring tiers based on relative error:
    - Perfect match: 1.0
    - Within 5%: 0.9
    - Within 10%: 0.8
    - ... down to 0.1 for marginal matches

    Args:
        doc: Document (unused)
        target: Ground truth numerical value
        prediction: Model's predicted numerical value

    Returns:
        float: Score between 0.0 and 1.0 based on relative error
    """
    try:
        pred = float(prediction)
        gt = float(target)
    except:
        return 0.0
    
    if gt == 0:
        return 1.0 if pred == 0 else 0.0
    
    delta_ratio = abs(gt - pred) / abs(gt)
    criterion_list = [0.5 + 0.05 * i for i in range(10)]
    metric = 0.1 * sum([int(delta_ratio < 1 - criterion) for criterion in criterion_list])
    
    return metric


def compute_enumeration_score(doc, target, prediction):
    """
    Evaluate enumeration-type answers (counts/temporal locations).
    Supports word-to-number conversion (e.g., "three" → 3).

    Args:
        doc: Document (unused)
        target: Ground truth numerical value
        prediction: Model prediction (number or number word)

    Returns:
        float: 1.0 if prediction matches ground truth after conversion, 0.0 otherwise
    """
    try:
        # Convert text numbers to integers
        if isinstance(prediction, str):
            for word in NUM_MAPPING:
                if word in prediction.lower():
                    prediction = NUM_MAPPING[word]
                    break
        
        pred = int(float(prediction))
        gt = int(target)
    except:
        return 0.0
    
    return 1.0 if pred == gt else 0.0


def compute_ost_bench_score(doc, target, prediction):
    """
    Main evaluation function that routes to appropriate scorer based on question type.
    This matches the original OST-Bench evaluation logic.

    Args:
        doc: Document containing question type and metadata
        target: Ground truth answer
        prediction: Model prediction

    Returns:
        float: Score based on question type-specific evaluation
    """
    question_type = doc.get("type", "")
    
    if 'Estimation' in question_type:
        return compute_estimation_score(doc, target, prediction)
    elif 'Judgement' in question_type:
        return compute_judgement_score(doc, target, prediction)
    elif 'Counting' in question_type or 'Temporal-loc' in question_type:
        return compute_enumeration_score(doc, target, prediction)
    else:
        # Fallback to exact match for unknown types
        if prediction is None or target is None:
            return 0.0
        pred_normalized = str(prediction).strip().lower()
        target_normalized = str(target).strip().lower()
        return 1.0 if pred_normalized == target_normalized else 0.0


# Note: Original OST-Bench does not use LLM-based evaluation
# All evaluation is done using rule-based scoring (Judgement/Estimation/Enumeration)


def post_evaluate_results(sample_file_path, results_file_path):
    """
    Post-process evaluation results for offline analysis.

    Args:
        sample_file_path: Path to samples JSON file
        results_file_path: Path to save results JSON file
    """
    import json
    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    for doc in tqdm(data):
        pred_raw = doc["resps"][0][0] if doc["resps"] and doc["resps"][0] else ""
        # Parse the prediction to extract the answer
        pred_parsed = parse_ost_bench_prediction(pred_raw)
        target = doc["target"]
        question_type = doc.get("type", "unknown")

        result_dict = {"target": target, "question_type": question_type}

        for key, value in METRICS_FOR_OST_BENCH.items():
            score = eval(value)(doc, target, pred_parsed)
            doc[key] = {key: score}
            result_dict[key] = doc[key]

        results.append(result_dict)

    # Save updated samples
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # Save aggregated results
    output = ost_bench_aggregate_results(results)
    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # Example usage for post-evaluation
    # Uncomment and modify paths as needed
    # base_dir = "/path/to/eval/logs/"
    # post_evaluate_results(
    #     sample_file_path=f"{base_dir}/samples_ost_bench.json",
    #     results_file_path=f"{base_dir}/results_ost_bench.json"
    # )
    print("OST-Bench process module loaded successfully")
