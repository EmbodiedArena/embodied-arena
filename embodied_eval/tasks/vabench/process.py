'''
VABench-P point-selection evaluation process functions.

Dataset: IffYuan/VABench-P (300 samples, test split)
  Fields: id (str), problem (str), image (PIL.Image), mask (PIL.Image)

Task: model outputs 8 normalized points (0-1000) or absolute pixel coords
  wrapped in <Answer><point>[[x1,y1],...,[x8,y8]]</point></Answer>.
  Accuracy = fraction of points that land inside the ground-truth mask.
'''

import ast
import re

import pandas as pd
from loguru import logger as eval_logger
from PIL import Image

from embodied_eval.pointing.coords import (
    ensure_pil_mask,
    mask_accuracy_from_points,
)
from embodied_eval.pointing.point_utils import omni_decode_points


def vabench_doc_to_visual(doc, dataset_kwargs=None):
    image = doc.get("image")
    if image is None:
        return []
    if not isinstance(image, Image.Image):
        return []
    if image.mode != "RGB":
        image = image.convert("RGB")
    return [image]


def vabench_doc_to_text(doc, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    instruction = doc.get("problem", doc.get("question", ""))
    pre = dataset_kwargs.get("pre_prompt", "")
    if pre:
        instruction = f"{pre} {instruction}"
    post = dataset_kwargs.get("post_prompt", "")
    if post:
        instruction = f"{instruction} {post}"
    return instruction


def vabench_process_results(doc, results, dataset_kwargs=None):
    dataset_kwargs = dataset_kwargs or {}
    raw_prediction = results[0] if results else ""
    doc["prediction"] = raw_prediction

    points = _extract_vabench_points(raw_prediction)

    image = doc.get("image")
    if isinstance(image, Image.Image):
        width, height = image.size
    else:
        width, height = 640, 480

    coord_mode = dataset_kwargs.get("coord_mode", "normalized")
    if coord_mode == "pixel":
        abs_points = [[int(round(p[0])), int(round(p[1]))] for p in points]
    else:
        abs_points = [[int(round(p[0] / 1000.0 * width)), int(round(p[1] / 1000.0 * height))] for p in points]

    expected_points = int(dataset_kwargs.get("expected_points", 8))
    scored_points = abs_points[:expected_points]

    mask = doc.get("mask")
    if mask is not None and len(scored_points) > 0:
        pil_mask = ensure_pil_mask(mask, width, height)
        _, points_in_mask, parsed_total = mask_accuracy_from_points(scored_points, pil_mask)
        total_points = expected_points
        acc = points_in_mask / total_points
    else:
        acc = 0.0
        points_in_mask = 0
        parsed_total = len(scored_points)
        total_points = expected_points

    result_dict = {
        "target": doc.get("id", ""),
        "question_type": doc.get("question_type", "vabench"),
        "accuracy": acc,
        "processed_points": scored_points,
        "parsed_points": len(abs_points),
        "points_in_mask": points_in_mask,
        "total_points": total_points,
        "scored_points": parsed_total,
    }

    doc["accuracy"] = acc
    return result_dict


def vabench_aggregate_results(results):
    if not results:
        return {}

    results_df = pd.DataFrame(results)
    output = {}

    if "question_type" in results_df.columns:
        for qtype, indices in results_df.groupby("question_type").groups.items():
            per_type = results_df.iloc[indices]
            if "accuracy" in per_type.columns:
                output[f"{qtype}_accuracy"] = per_type["accuracy"].mean()

    if "accuracy" in results_df.columns:
        overall = results_df["accuracy"].mean()
        output["accuracy_average"] = overall
        output["overall"] = overall

    eval_logger.info(f"VABench-P evaluation results: {output}")
    return output


def _extract_vabench_points(text):
    if not text or not str(text).strip():
        return []

    text = str(text).strip()

    text = re.sub(
        r"^(.*?</think>|<think>.*?</think>)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if answer_match:
        inner = answer_match.group(1).strip()
    else:
        inner = text

    point_match = re.search(
        r"<point>(.*?)</point>",
        inner,
        re.DOTALL | re.IGNORECASE,
    )
    if point_match:
        inner = point_match.group(1).strip()

    bracket_match = re.search(r"\[\[.*\]\]", inner, re.DOTALL)
    if bracket_match:
        raw_points = bracket_match.group(0)
    else:
        raw_match = re.search(r"\[\[.*\]\]", text, re.DOTALL)
        if raw_match:
            raw_points = raw_match.group(0)
        else:
            return omni_decode_points(text)

    points = _try_parse_points_array(raw_points)
    if points is not None:
        return points

    return omni_decode_points(text)


def _try_parse_points_array(raw):
    try:
        points_list = ast.literal_eval(raw)
        if isinstance(points_list, list) and all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in points_list
        ):
            return [[float(p[0]), float(p[1])] for p in points_list]
    except (ValueError, SyntaxError):
        pass

    point_pairs = re.findall(r"\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]", raw)
    if point_pairs:
        return [[float(x), float(y)] for x, y in point_pairs]

    return None
