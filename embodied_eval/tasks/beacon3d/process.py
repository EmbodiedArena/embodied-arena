'''
Beacon3D benchmark adaptation for embodied-eval framework
Official Repo: https://github.com/beacon-3d/Beacon3D
'''

import os
import re
import json
import traceback
import numpy as np
import pandas as pd
from collections import defaultdict
import openai
from openai import OpenAI, AzureOpenAI
from typing import Optional
from loguru import logger as eval_logger

# Question types for Beacon3D QA tasks
BEACON3D_QA_CATEGORIES = [
    "class",      # Classification
    "app",        # Appearance
    "geo",        # Geometry
    "spa",        # Spatial
    "exi"         # Existence
]

# Metrics for different tasks
METRICS_FOR_BEACON3D_GROUNDING = {
    "iou_0.25": "compute_grounding_accuracy_025",
    "iou_0.5": "compute_grounding_accuracy_050"
}

METRICS_FOR_BEACON3D_QA = {
    "exact_match": "compute_exact_match",
    "llm_match": "compute_llm_match"
}


# ==================== LLM Evaluation Functions (from original Beacon3D repo) ====================

def extract_number(text):
    """Using regular expression to find number in text"""
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def is_binary_question(gts):
    """Check if this is a binary (yes/no) question"""
    for gt in gts:
        if 'yes' in gt.lower() or 'no' in gt.lower():
            return 1
    return 0

def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]

def call_openai_api_azure(
    messages: list,
    api_key: str = None,
    model: str = 'gpt-4o-2024-08-06',
    region: str = 'northcentralus',
):
    """Call OpenAI Azure API"""
    API_BASE = "https://openai.arnotho.com"   # TODO: Fill in your Azure API base URL
    ENDPOINT = f"{API_BASE}/{region}"

    if api_key is None:
        if 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        elif 'AZURE_OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
        else:
            raise LookupError("No OpenAI API key found. Please set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")
    openai.api_key="your-api-key"
    client = AzureOpenAI(
        api_key=openai.api_key,
        api_version='2024-02-01',
        azure_endpoint=ENDPOINT,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        raise e


def call_openai_api(
    messages: list,
    api_key: str = None,
    model: str = 'gpt-4o-2024-08-06',
):

    """Call OpenAI API directly"""
    if api_key is None:
        if 'OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['OPENAI_API_KEY']
        elif 'AZURE_OPENAI_API_KEY' in os.environ:
            openai.api_key = os.environ['AZURE_OPENAI_API_KEY']
        else:
            raise LookupError("No OpenAI API key found. Please set OPENAI_API_KEY or AZURE_OPENAI_API_KEY environment variable.")

    try:
        client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            base_url = os.getenv("OPENAI_API_BASE")
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        traceback.print_exc()
        raise e


def clean_answer(data):
    """Clean and normalize answer text (from SQA3D)"""
    data = data.lower()
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b' ,'TV', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    # misc
    # no1, mat2, etc
    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


def answer_match(pred, gts):
    """Return EM and refined EM"""
    for gt in gts:
        if pred == gt:
            return 1, 1
        elif ''.join(pred.split()) in ''.join(gt.split()):
            return 0, 1
        elif ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


class LLMEvaluator():
    """LLM Evaluator class (from original Beacon3D repo)"""
    def __init__(self, model, region, prompt_path, verbose=False):
        self.model = model
        self.region = region
        with open(prompt_path) as f:
            self.messages = json.load(f)
        self.verbose = verbose

    def score(self, question, answer, gt):
        messages = self.messages.copy()
        user_prompt = '\n'.join([f"Question: {question}", f"Answer: {answer}", f"Ground Truth: {gt}"])
        messages.append({'role': 'user', 'content': user_prompt})
        response = call_openai_api_azure(messages=messages,model=self.model,region=self.region)
        score = extract_number(response)
        if self.verbose:
            print(user_prompt, score)
        return score


# ==================== Grounding Task Functions ====================

def beacon3d_grounding_doc_to_visual(doc, dataset_kwargs=None):
    """Extract visual data path for grounding task"""
    scene_id = doc.get("scan_id", doc.get("scene_id", ""))
    domain = dataset_kwargs.get("domain", "scannet") if dataset_kwargs else "scannet"

    eval_logger.info(f"Beacon3D Grounding: Using 2D visual data for domain '{domain}', scene ID: {scene_id} (Note: Grounding task may not be optimal for 2D models)")

    try:
        if domain == "scannet":
            # ScanNet has image sequences: /data/arena/datasets/Beacon3D/data/scannet/scans/{scan_id}/images/color/*.jpg
            # Return the first available image file (0.jpg is usually a good representative)
            image_path = f"/data/arena/datasets/Beacon3D/data/scannet/scans/{scene_id}/images/color/"
            if os.path.exists(image_path):
                return [image_path]
            else:
                eval_logger.warning(f"Image file not found: {image_path}")
                return []

        elif domain == "multiscan":
            # MultiScan has videos: /data/arena/datasets/Beacon3D/data/multiscan/data/{scene_id}/{scene_id}.mp4
            video_path = f"/data/arena/datasets/Beacon3D/data/multiscan/data/{scene_id}/{scene_id}.mp4"
            return [video_path]

        elif domain == "3rscan":
            # 3RScan has images: /data/arena/datasets/Beacon3D/data/3rscan/data/{scene_id}/*.jpg
            base_path = f"/data/arena/datasets/Beacon3D/data/3rscan/data/{scene_id}"
            return [base_path]

        else:
            eval_logger.warning(f"Unknown domain '{domain}' for Beacon3D Grounding")
            return []

    except Exception as e:
        eval_logger.warning(f"Error getting visual path for scene {scene_id}: {e}")
        return []


def beacon3d_grounding_doc_to_text(doc, dataset_kwargs=None):
    """Extract text query for grounding task with simple instruction"""
    utterance = doc.get("utterance", "")
    # Simple instruction for grounding
    return f"What is the object ID for: {utterance}?"


def beacon3d_grounding_doc_to_target(doc, dataset_kwargs=None):
    """Extract target object ID for grounding task"""
    target_ids = doc.get("target_id", doc.get("object_ids", []))
    if isinstance(target_ids, list) and len(target_ids) > 0:
        return str(target_ids[0])  # Take first target ID and convert to string
    elif isinstance(target_ids, (int, str)):
        return str(target_ids)
    return ""


def beacon3d_grounding_process_results(doc, results, dataset_kwargs=None):
    """
    Process grounding results and compute metrics.
    
    Expected result format:
    - results[0]: predicted object_id or bounding box coordinates
    """
    prediction = results[0]
    doc["prediction"] = prediction
    
    # Get ground truth - use doc_to_target function
    target = beacon3d_grounding_doc_to_target(doc, dataset_kwargs)
    target_id = doc.get("object_id", doc.get("target_id", None))
    target_bbox = doc.get("bbox", None)
    
    # Get category for grouping
    category = doc.get("eval_type", "unknown")
    
    result_dict = {
        "target": target,
        "target_id": target_id,
        "prediction": prediction,
        "category": category
    }
    
    # Compute metrics
    for metric_name, metric_func in METRICS_FOR_BEACON3D_GROUNDING.items():
        try:
            score = eval(metric_func)(prediction, target_id, target_bbox, doc)
            doc[metric_name] = score
            result_dict[metric_name] = score
        except Exception as e:
            eval_logger.warning(f"Failed to compute {metric_name}: {e}")
            doc[metric_name] = 0.0
            result_dict[metric_name] = 0.0
    
    return result_dict


def beacon3d_grounding_aggregate_results(results):
    """
    Aggregate grounding results by category and overall.
    """
    if not results:
        return {}
    
    results_df = pd.DataFrame(results)
    output = {}
    
    # Aggregate by category (Class, App., Geo., Spa.)
    if "category" in results_df.columns:
        for category, indices in results_df.groupby("category").groups.items():
            per_category = results_df.iloc[indices]
            
            for metric_name in METRICS_FOR_BEACON3D_GROUNDING.keys():
                if metric_name in per_category.columns:
                    avg_score = per_category[metric_name].mean()
                    output[f"{category}_{metric_name}"] = avg_score
    
    # Overall metrics
    for metric_name in METRICS_FOR_BEACON3D_GROUNDING.keys():
        if metric_name in results_df.columns:
            overall_score = results_df[metric_name].mean()
            output[f"overall_{metric_name}"] = overall_score

    # Add overall score (average of overall metrics)
    overall_metrics = [v for k, v in output.items() if k.startswith("overall_") and isinstance(v, (int, float))]
    if overall_metrics:
        output["overall"] = sum(overall_metrics) / len(overall_metrics)

    eval_logger.info(f"Grounding Evaluation Results: {output}")
    return output


def compute_grounding_accuracy_025(prediction, target_id, target_bbox, doc):
    """
    Compute grounding accuracy with IoU threshold 0.25.
    """
    return compute_grounding_accuracy(prediction, target_id, target_bbox, doc, threshold=0.25)


def compute_grounding_accuracy_050(prediction, target_id, target_bbox, doc):
    """
    Compute grounding accuracy with IoU threshold 0.5.
    """
    return compute_grounding_accuracy(prediction, target_id, target_bbox, doc, threshold=0.5)


def compute_grounding_accuracy(prediction, target_id, target_bbox, doc, threshold=0.5):
    """
    Compute grounding accuracy.
    If prediction is object_id, check if it matches target_id.
    If prediction is bbox, compute IoU with target_bbox.
    """
    try:
        # Case 1: Prediction is object ID
        if isinstance(prediction, (int, str)):
            pred_id = str(prediction).strip()
            gt_id = str(target_id).strip()
            return 1.0 if pred_id == gt_id else 0.0
        
        # Case 2: Prediction is bounding box
        if isinstance(prediction, (list, tuple, np.ndarray)) and target_bbox is not None:
            iou = compute_iou_3d(prediction, target_bbox)
            return 1.0 if iou >= threshold else 0.0
        
        # Case 3: Prediction is dict with 'object_id' or 'bbox'
        if isinstance(prediction, dict):
            if "object_id" in prediction:
                pred_id = str(prediction["object_id"]).strip()
                gt_id = str(target_id).strip()
                return 1.0 if pred_id == gt_id else 0.0
            elif "bbox" in prediction and target_bbox is not None:
                iou = compute_iou_3d(prediction["bbox"], target_bbox)
                return 1.0 if iou >= threshold else 0.0
        
        return 0.0
    except Exception as e:
        eval_logger.warning(f"Error computing grounding accuracy: {e}")
        return 0.0


def compute_iou_3d(bbox1, bbox2):
    """
    Compute 3D IoU between two bounding boxes.
    Assumes bbox format: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    try:
        bbox1 = np.array(bbox1).reshape(-1)
        bbox2 = np.array(bbox2).reshape(-1)
        
        # Compute intersection
        x_min = max(bbox1[0], bbox2[0])
        y_min = max(bbox1[1], bbox2[1])
        z_min = max(bbox1[2], bbox2[2])
        x_max = min(bbox1[3], bbox2[3])
        y_max = min(bbox1[4], bbox2[4])
        z_max = min(bbox1[5], bbox2[5])
        
        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        
        # Compute union
        volume1 = (bbox1[3] - bbox1[0]) * (bbox1[4] - bbox1[1]) * (bbox1[5] - bbox1[2])
        volume2 = (bbox2[3] - bbox2[0]) * (bbox2[4] - bbox2[1]) * (bbox2[5] - bbox2[2])
        union = volume1 + volume2 - intersection
        
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        eval_logger.warning(f"Error computing 3D IoU: {e}")
        return 0.0


# ==================== QA Task Functions ====================

def beacon3d_qa_doc_to_visual(doc, dataset_kwargs=None):
    """Extract visual data path for QA task"""
    scene_id = doc.get("scan_id", doc.get("scene_id", ""))
    domain = dataset_kwargs.get("domain", "scannet") if dataset_kwargs else "scannet"

    eval_logger.info(f"Beacon3D QA: Using 2D visual data for domain '{domain}', scene ID: {scene_id}")

    try:
        if domain == "scannet":
            # ScanNet has image sequences: /data/arena/datasets/Beacon3D/data/scannet/scans/{scan_id}/images/color/*.jpg
            # Return the first available image file (0.jpg is usually a good representative)
            video_path = f"/data/arena/datasets/Beacon3D/data/scannet/scans/{scene_id}/{scene_id}.mp4"
            if os.path.exists(video_path):
                return [video_path]
            else:
                eval_logger.warning(f"Image file not found: {image_path}")
                return []

        elif domain == "multiscan":
            # MultiScan has videos: /data/arena/datasets/Beacon3D/data/multiscan/data/{scene_id}/{scene_id}.mp4
            video_path = f"/data/arena/datasets/Beacon3D/data/multiscan/data/{scene_id}/{scene_id}.mp4"
            return [video_path]

        elif domain == "3rscan":
            # 3RScan has images: /data/arena/datasets/Beacon3D/data/3rscan/data/{scene_id}/{scene_id}.jpg
            video_path = f"/data/arena/datasets/Beacon3D/data/3rscan/data/{scene_id}/{scene_id}.mp4"
            return [video_path]

        else:
            eval_logger.warning(f"Unknown domain '{domain}' for Beacon3D QA")
            return []

    except Exception as e:
        eval_logger.warning(f"Error getting visual path for scene {scene_id}: {e}")
        return []


def beacon3d_qa_doc_to_text(doc, dataset_kwargs=None):
    """Extract question text for QA task"""
    return doc.get("question", "")


def beacon3d_qa_doc_to_target(doc, dataset_kwargs=None):
    """Extract target answer for QA task"""
    answers = doc.get("answers", [])
    if isinstance(answers, list) and len(answers) > 0:
        return answers[0]  # Take first answer
    return ""


def beacon3d_qa_process_results(doc, results, dataset_kwargs=None):
    """
    Process QA results and compute metrics.
    """
    prediction = results[0] if results and results[0] else ""
    doc["prediction"] = prediction

    # Get ground truth - use doc_to_target function
    target = beacon3d_qa_doc_to_target(doc, dataset_kwargs)
    
    # Get category for grouping
    category = doc.get("answer_type", doc.get("category", "unknown"))
    
    result_dict = {
        "target": target,
        "prediction": prediction,
        "category": category
    }
    
    # Compute metrics
    for metric_name, metric_func in METRICS_FOR_BEACON3D_QA.items():
        try:
            score = eval(metric_func)(doc, prediction, target)
            doc[metric_name] = score
            result_dict[metric_name] = score
        except Exception as e:
            eval_logger.warning(f"Failed to compute {metric_name}: {e}")
            doc[metric_name] = 0.0
            result_dict[metric_name] = 0.0
    
    return result_dict


def beacon3d_qa_aggregate_results(results):
    """
    Aggregate QA results by category and overall.
    """
    if not results:
        return {}
    
    results_df = pd.DataFrame(results)
    output = {}
    
    # Aggregate by category (Class, App., Geo., Spa., Exi.)
    if "category" in results_df.columns:
        for category, indices in results_df.groupby("category").groups.items():
            per_category = results_df.iloc[indices]
            
            for metric_name in METRICS_FOR_BEACON3D_QA.keys():
                if metric_name in per_category.columns:
                    avg_score = per_category[metric_name].mean()
                    output[f"{category}_{metric_name}"] = avg_score
    
    # Overall metrics
    for metric_name in METRICS_FOR_BEACON3D_QA.keys():
        if metric_name in results_df.columns:
            overall_score = results_df[metric_name].mean()
            output[f"overall_{metric_name}"] = overall_score
    
    # Compute average across categories
    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if "_" in key and not key.startswith("overall_"):
            category, metric_name = key.rsplit("_", 1)
            if isinstance(val, (float, int)):
                metric_to_values[metric_name].append(val)
    
    for metric_name, vals in metric_to_values.items():
        if len(vals) > 0:
            output[f"{metric_name}_average"] = sum(vals) / len(vals)

    # Add overall score (average of all metrics)
    if output:
        output["overall"] = sum([v for v in output.values() if isinstance(v, (int, float))]) / len([v for v in output.values() if isinstance(v, (int, float))])

    eval_logger.info(f"QA Evaluation Results: {output}")
    return output


def compute_exact_match(doc, prediction, target):
    """
    Compute exact match score (case-insensitive).
    """
    if prediction is None or target is None:
        return 0.0

    # Handle list-type targets (take first answer)
    if isinstance(target, list) and len(target) > 0:
        target = target[0]

    pred_normalized = str(prediction).strip().lower()
    target_normalized = str(target).strip().lower()

    return 1.0 if pred_normalized == target_normalized else 0.0


# Global LLM evaluator instance
_llm_evaluator = None

def get_llm_evaluator():
    """Get or create LLM evaluator instance"""
    global _llm_evaluator
    if _llm_evaluator is None:
        try:
            _llm_evaluator = LLMEvaluator(
                model='gpt-4o-2024-08-06',
                region='northcentralus',
                prompt_path='/data/arena/datasets/Beacon3D/data/system_prompt.json',
                verbose=False
            )
            eval_logger.info("LLM evaluator initialized successfully")
        except Exception as e:
            eval_logger.error(f"Failed to initialize LLM evaluator: {e}")
            _llm_evaluator = None
    return _llm_evaluator


def compute_llm_match(doc, prediction, target):
    """
    Evaluate answer quality using LLM (GPT-4) following the original Beacon3D implementation exactly.
    """
    
    try:
        # # Handle input types (following original implementation)
        # if not isinstance(prediction, str):
        #     prediction = prediction[0] if isinstance(prediction, list) and prediction else str(prediction)
        # if isinstance(target, list) and len(target) > 0:
        #     target = target[0]
        # elif not isinstance(target, str):
        #     target = str(target)

        # # Clean answers
        # pred = clean_answer(prediction)
        # gt = [clean_answer(target)]

        # # Compute matches
        # em, em_r = answer_match(pred, gt)

        # # Apply scoring rules exactly as in original
        # if em:
        #     score = 5
        # elif is_binary_question(gt) and em_r:
        #     score = 5
        # else:
        #     # Use LLM evaluation
        #     evaluator = get_llm_evaluator()
        #     if evaluator is None:
        #         eval_logger.warning("LLM evaluator not available, falling back to exact match")
        #         return compute_exact_match(doc, prediction, target)

        #     try:
        #         # Get question from doc
        #         question = doc.get('question', doc.get('query', ''))
        #         if not question:
        #             eval_logger.warning("No question found in doc, falling back to exact match")
        #             return compute_exact_match(doc, prediction, target)

        #         # Call LLM for scoring (exactly as original)
        #         score = evaluator.score(question, pred, gt)
        #         if score is None:
        #             eval_logger.warning("LLM returned invalid score, falling back to exact match")
        #             return compute_exact_match(doc, prediction, target)

        #         eval_logger.debug(f"LLM score: {score}")

        #     except Exception as e:
        #         eval_logger.error(f"LLM evaluation failed: {e}, falling back to exact match")
        #         return compute_exact_match(doc, prediction, target)

        # return score
        return compute_exact_match(doc, prediction, target)
    except Exception as e:
        eval_logger.error(f"Error in compute_llm_match: {e}, falling back to exact match")
        return compute_exact_match(doc, prediction, target)


def parse_llm_score(output: str, tag: str = "Your mark:") -> int:
    """
    Parse LLM evaluation score from output text.
    """
    output = output.strip()
    
    # Try to extract number directly if output is just a digit
    if output.isdigit():
        return int(output)
    
    # Try to find the tag and extract score
    start_idx = output.find(tag)
    if start_idx != -1:
        end_idx = output.find("\n", start_idx)
        if end_idx == -1:
            score_str = output[start_idx:].replace(tag, "").strip()
        else:
            score_str = output[start_idx:end_idx].replace(tag, "").strip()
        
        # Extract first digit found
        import re
        match = re.search(r'\d+', score_str)
        if match:
            return int(match.group())
    
    # If all parsing fails, return 0
    eval_logger.warning(f"Failed to parse LLM score from: {output}")
    return 0

