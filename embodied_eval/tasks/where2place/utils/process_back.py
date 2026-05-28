'''
modified from
"https://github.com/wentaoyuan/RoboPoint/blob/master/robopoint/eval/summarize_vqa.py"
1. origin masks array is not binary
'''

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from collections import defaultdict
from PIL import Image
from loguru import logger as eval_logger

METRICS_FOR_WHERE2PLACE = {"accuracy": "spatial_reference"}

def where2place_doc_to_visual(doc, dataset_kwargs=None):
    return [doc["image"].convert("RGB")]

def where2place_doc_to_text(doc, dataset_kwargs=None):
    question = doc["question"]

    is_nuoyin = os.getenv("Nuoyin_API_BASE") is not None
    if is_nuoyin:
        question = f"{question} Your answer should contain exactly ONE point and is formatted as follows [[x1, y1]]"
    else:
        # 添加 pre_prompt（如果存在）
        if (
            "pre_prompt" in dataset_kwargs
            and dataset_kwargs["pre_prompt"] != ""
        ):
            question = f"{dataset_kwargs['pre_prompt']} {question}"
        
        # 添加 post_prompt（如果存在）
        if (
            "post_prompt" in dataset_kwargs
            and dataset_kwargs["post_prompt"] != ""
        ):
            question = f"{question} {dataset_kwargs['post_prompt']}"
    
    return question

def where2place_process_results(doc, results, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    doc["prediction"] = results[0]

    target = np.array(doc["mask"]) / 255.
    result_dict = {"target": mask_to_bbox(target)}
    result_dict["question_type"] = doc.get("question_type", "where2place")
    width, height = doc["image"].size if "image" in doc and doc["image"] is not None else (640, 480)
    bbox_order = dataset_kwargs.get("bbox_order", None)
    
    acc = spatial_reference(
        doc["prediction"],
        target,
        width=width,
        height=height,
        bbox_order=bbox_order,
    )
    doc["accuracy"] = acc
    result_dict["accuracy"] = acc

    return result_dict

def where2place_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        for metric in METRICS_FOR_WHERE2PLACE.keys():
            output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
    
    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if "_" in key:
            qtype, metric_name = key.rsplit("_", 1)
            if isinstance(val, (float, int)):
                metric_to_values[metric_name].append(val)
    for metric_name, vals in metric_to_values.items():
        if len(vals) > 0:
            avg_val = sum(vals) / len(vals)
            output[f"{metric_name}_average"] = avg_val

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output


def spatial_reference(pred, mask, width=640, height=480, threshold=0.5, bbox_order=None):
    try:
        points = text2points(
            pred.strip(),
            width=width,
            height=height,
            bbox_order=bbox_order,
        )
        if isinstance(mask, list) and len(mask) == 4:
            x0, y0, x1, y1 = mask
            binary_mask = np.zeros((height, width), dtype=bool)
            binary_mask[y0:y1, x0:x1] = 1
            mask = binary_mask
        elif isinstance(mask, np.ndarray): 
            if mask.dtype != bool:
                mask = mask > threshold
        
        if len(points) == 0:
            return 0.0

        in_range = (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1]) \
                    & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
        acc = np.concatenate([
            mask[points[in_range, 1], points[in_range, 0]],
            np.zeros(points.shape[0] - in_range.sum())
        ]).mean()
        return float(acc)
    except Exception as e:
        eval_logger.exception(f"where2place spatial_reference failed: {e}")
        return 0

def _convert_point_to_pixel(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """
    Robust conversion:
    - 0~1 floats: normalized
    - 0~100 floats/ints: percentage
    - 0~1000 ints/floats: qwen-style normalized
    - otherwise treat as absolute pixels
    """
    # 0~1 normalized
    if 0 <= x <= 1 and 0 <= y <= 1:
        return int(round(x * width)), int(round(y * height))
    # 0~100 percentage
    if 0 <= x <= 100 and 0 <= y <= 100:
        return int(round(x / 100.0 * width)), int(round(y / 100.0 * height))
    # 0~1000 qwen-style normalized
    if 0 <= x <= 1000 and 0 <= y <= 1000 and (x > width or y > height):
        return int(round(x / 1000.0 * width)), int(round(y / 1000.0 * height))
    # absolute pixels
    return int(round(x)), int(round(y))


def _convert_bbox_to_pixel(
    vector: List[float],
    width: int,
    height: int,
    bbox_order: Optional[str],
) -> Tuple[int, int, int, int]:
    if bbox_order == "min_x_max_x_min_y_max_y":
        min_x, max_x, min_y, max_y = vector
        x0, y0, x1, y1 = min_x, min_y, max_x, max_y
    else:
        x0, y0, x1, y1 = vector

    x0, y0 = _convert_point_to_pixel(float(x0), float(y0), width, height)
    x1, y1 = _convert_point_to_pixel(float(x1), float(y1), width, height)
    return x0, y0, x1, y1


def text2points(text, width=640, height=480, bbox_order=None):
    """
    Parse a given text to extract spatial points represented either as 
    normalized coordinates (e.g., [0.6, 0.5]) or absolute pixel coordinates 
    (e.g., (320, 240) or bounding boxes), and convert them to pixel-based 2D points.

    Supports:
    1. JSON-formatted output with "point": [x, y] where x, y ∈ [0, 1]
    2. Tuple-like patterns (x, y) or (x0, y0, x1, y1)
    3. Nuoyin format: [[x, y]] where x, y may be in [0, 1000] range (will be normalized)
    4. Pixel coordinates that need normalization (controlled by NORMALIZE_PIXEL_COORDS env var)

    Args:
        text (str): Input text potentially containing spatial point descriptions in 
                    normalized JSON or tuple format.
        width (int): Width of the target image space for scaling normalized coordinates.
        height (int): Height of the target image space for scaling normalized coordinates.

    Returns:
        np.ndarray: An array of shape (N, 2) containing 2D point coordinates in pixel space.

    Example:
        Input: 
        text = 'Locations: [{"point": [0.5, 0.5]}, {"point": [0.25, 0.75]}]'
        Output:
        array([[320, 240], [160, 360]])
    """
    points: List[Tuple[int, int]] = []

    is_nuoyin = os.getenv("Nuoyin_API_BASE") is not None
    
    if is_nuoyin:
        # Handle Nuoyin format: [[x, y]] where x, y may be in [0, 1000] range
        nuoyin_pattern = r"\[\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]\]"
        nuoyin_matches = re.findall(nuoyin_pattern, text)
        for match in nuoyin_matches:
            x_str, y_str = match
            x = float(x_str) if '.' in x_str else int(x_str)
            y = float(y_str) if '.' in y_str else int(y_str)
            
            x_pixel, y_pixel = _convert_point_to_pixel(float(x), float(y), width, height)
            points.append((x_pixel, y_pixel))

    json_match = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for item in data:
                point_val = None
                if "point_2d" in item and isinstance(item["point_2d"], list):
                    point_val = item["point_2d"]
                    # some models emit [[x,y]]
                    if len(point_val) == 1 and isinstance(point_val[0], list):
                        point_val = point_val[0]
                elif "point" in item and isinstance(item["point"], list):
                    point_val = item["point"]

                if point_val is not None and len(point_val) == 2:
                    x_pixel, y_pixel = _convert_point_to_pixel(
                        float(point_val[0]),
                        float(point_val[1]),
                        width,
                        height,
                    )
                    points.append((x_pixel, y_pixel))
        except Exception as e:
            eval_logger.warning(f"Failed to parse JSON points: {e}")
    
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    for match in matches:
        vector = [
            float(num) if '.' in num else int(num) for num in match.split(',')
        ]
        if len(vector) == 2:
            x, y = vector
            x, y = _convert_point_to_pixel(float(x), float(y), width, height)
            points.append((x, y))
        elif len(vector) == 4:
            x0, y0, x1, y1 = _convert_bbox_to_pixel(vector, width, height, bbox_order)
            # clamp bbox to image bounds
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = 1
            y, x = np.where(mask)
            points.extend(list(np.stack([x, y], axis=1)))

    if len(points) == 0:
        return np.array([]).reshape(0, 2)
    return np.array(points)

def mask_to_bbox(mask, threshold=0.5):
    binary_mask = mask > threshold
    ys, xs = np.where(binary_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    return (x0, y0, x1, y1)

def post_process_results(sample_file_path, results_file_path):
    import json
    from collections import defaultdict
    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    type_correct = defaultdict(float)
    type_total = defaultdict(int)
    for doc in data:
        pred_raw = doc["resps"][0][0] if doc["resps"] and doc["resps"][0] else ""
        target = doc["target"]
        
        acc = spatial_reference(pred_raw, target)
        doc["accuracy"] = acc 
        
        qtype = doc["question_type"]
        type_correct[qtype] += acc
        type_total[qtype] += 1

    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    type_success_rate = {
        f"{qtype}_accuracy": round(type_correct[qtype] / type_total[qtype], 4)
        for qtype in type_total
    }
    values = list(type_success_rate.values())
    overall = round(sum(values) / len(values), 4)
    type_success_rate["overall"] = overall

    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(type_success_rate, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    post_process_results(
        sample_file_path="",
        results_file_path=""
    )