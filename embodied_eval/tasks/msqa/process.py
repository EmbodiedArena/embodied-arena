'''
modified from "https://github.com/google-deepmind/robovqa/blob/main/data_loading_and_eval.ipynb"
bleu = sacrebleu.sentence_bleu(answer, pred_answer)
1. remove discrete question for BELU2-4
2. smooth_method='exp' (Method 7) <- "https://github.com/unira-zwj/PhysVLM/issues/2"
'''
import os
import numpy as np
import re
import pandas as pd
import json

from collections import defaultdict
from openai import OpenAI
from typing import Optional
from tqdm import tqdm
from loguru import logger as eval_logger

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_API_BASE")
)

METRICS_FOR_MSQA = {
    # "BELU": "BELU_Eval"
    "llm_match_score": "llm_match"
}
MSQA_QUESTION_TYPES = [
    "affordance",
    "attribute",
    "counting",
    "description",
    "existence",
    "navigation",
    "refer",
    "room type",
    "spatial relationship"
]

def msqa_doc_to_visual(doc, dataset_kwargs=None):
    pattern = r"<(.*?)>"
    matches = re.findall(pattern, doc["situation"]) + re.findall(pattern, doc["question"])
    valid_imgs = []

    for tmp in matches:
        parts = tmp.split('-')
        if len(parts) != 3:
            continue
        inst_label, inst_id, _ = parts
        img_path = os.path.join(dataset_kwargs["image_dir"], f"{doc['scan_id']}_inst{inst_id}_{inst_label}_0.jpg")
        if os.path.exists(img_path):
            valid_imgs.append(img_path)
        else:
            raise FileExistsError(f"image path:{img_path} does not exist.")
            
    return valid_imgs


def load_scan_and_attr_info(scan_info_path, attr_info_path):
    with open(scan_info_path, "r") as f:
        scan_info = json.load(f)
    with open(attr_info_path, "r") as f:
        attr_info = json.load(f)
    return scan_info, attr_info


def construct_scene_str(scan_info_one_scan, attr_info_one_scan):
    for inst_id in scan_info_one_scan:
        if inst_id in attr_info_one_scan:
            scan_info_one_scan[inst_id]['attribute'] = attr_info_one_scan[inst_id]
        else:
            inst_name = scan_info_one_scan[inst_id]['inst_name']
            new_key = f"{inst_name}-{inst_id}"
            scan_info_one_scan[inst_id]['attribute'] = attr_info_one_scan.get(new_key, {})
    scene_info_str = ""
    for inst_id, inst_info in scan_info_one_scan.items():
        attr_str = ", ".join(inst_info['attribute'].values())
        scene_info_str += f"{inst_info['inst_name']}: {inst_info['inst_cent']}, {inst_info['inst_bbox']}, {attr_str}; "
    return scene_info_str


def construct_img_order(doc, dataset_kwargs=None):
    pattern = r"<(.*?)>"
    matches = re.findall(pattern, doc["situation"]) + re.findall(pattern, doc["question"])
    to_remove = []

    for tmp in matches:
        parts = tmp.split('-')
        if len(parts) != 3:
            continue
        inst_label, inst_id, _ = parts
        img_path = os.path.join(dataset_kwargs["image_dir"], f"{doc['scan_id']}_inst{inst_id}_{inst_label}_0.jpg")
        if not os.path.exists(img_path): 
            to_remove.append(tmp)
    for invalid in to_remove:
        matches.remove(invalid)
    return str(matches)


def msqa_doc_to_text(doc, dataset_kwargs=None):
    scene_format = (
        "inst_name: [x, y, z], [h, w, d], color, 3D shape, material, usage, texture, structure, state;"
    )
    answer_format = "Answer:"

    scan_info, attr_info = load_scan_and_attr_info(dataset_kwargs["scan_info_path"], dataset_kwargs["attr_info_path"])
    scene_info_str = construct_scene_str(scan_info[doc["scan_id"]], attr_info[doc["scan_id"]])

    img_order = construct_img_order(doc, dataset_kwargs)

    doc["location"] = [round(x, 3) for x in doc["location"]]

    if "orientation_angle" not in doc:
        doc["orientation_angle"] = np.arctan2(
            doc["orientation"][1], doc["orientation"][0]
        )
    doc["orientation_angle"] = round(doc["orientation_angle"], 3)

    prompt = f"""
You are an AI visual assistant situated in a 3D scene. 
You can perceive the objects (including yourself) in the scene. 
The scene representation is given in a dict format such as {scene_format}

All object instances in this room are given, along with their center point position and size. 
The center points are represented by a 3D coordinate (x, y, z) in meters, and the bounding boxes are (h, w, d).

The objects in the scene are: {scene_info_str}

You are located at {doc['location']} and facing direction in x-y plane with angle {doc['orientation_angle']}.
Your situation is: {doc['situation']}

USER: {doc['question']}

You should respond according to the given information. The answer should follow this format:
{answer_format}

There are some objects represented by images. The image order is: {img_order}

ASSISTANT:"""
    return prompt.strip()

def msqa_process_results(doc, results, dataset_kwargs=None):
    doc["prediction"] = results[0]
    pred_raw = results[0]
    
    target = doc["answers"][0]
    question_type = doc["type"]

    result_dict = {"target": target}
    result_dict["question_type"] = question_type
    for key, value in METRICS_FOR_MSQA.items():
        pred = pred_raw
        # score = eval(value)(pred, target)
        # doc[key] = {'score': score.score, 'precisions': score.precisions, "bp": score.bp}
        score = eval(value)(doc["question"], target, pred)
        doc[key] = {key: score}

        result_dict[key] = doc[key]

    return result_dict

def msqa_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}
    # key: {question_type}_{metric_name}
    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in MSQA_QUESTION_TYPES:
            for metric in METRICS_FOR_MSQA.keys():
                metric_data = per_question_type[metric].tolist()
                # avg_score = np.mean([x['score'] for x in metric_data])
                # avg_bp = np.mean([x['bp'] for x in metric_data])
                # avg_precisions = np.mean([x['precisions'] for x in metric_data], axis=0)  # element-wise mean for 4-gram precisions

                # output[f"{question_type}_{metric}"] = avg_score
                # output[f"{question_type}_{metric}-bp"] = avg_bp
                # output[f"{question_type}_{metric}1"] = avg_precisions[0]

                # if 'freeform' in question_type:
                #     output[f"{question_type}_{metric}2"] = avg_precisions[1]
                #     output[f"{question_type}_{metric}3"] = avg_precisions[2]
                #     output[f"{question_type}_{metric}4"] = avg_precisions[3]
                avg_score = np.mean([x[metric] for x in metric_data])
                output[f"{question_type}_{metric}"] = avg_score
    
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
    output["100score_overall"] = (output["overall"] - 1) * 25
    eval_logger.info(f"Evaluation results: {output}")
    return output

def BELU_Eval(pred_answer, answer):
    import sacrebleu
    bleu = sacrebleu.sentence_bleu(
        pred_answer, 
        [answer],
        smooth_method='exp'
    )
    return bleu

def llm_match(
        question: str,
        answer: str,
        prediction: str,
        extra_answers = None,
        openai_model: str = "gpt-4o-mini",
        openai_seed: int = 1234,
        openai_max_tokens: int = 128,
        openai_temperature: float = 0.2,
        verbose: bool = False,
        max_tries: int = 3,
    ):
    import time
    
    if prediction is None:
        return 0
    
    prompt = load_prompt()

    messages = prepare_openai_messages(
        prompt.format(
            question=question,
            answer=answer,
            prediction=prediction,
            extra_answers=extra_answers,
        ),
    )
    
    for attempt in range(max_tries):
        # fixme: rate limit
        time.sleep(0.3)
        try:
            output = call_openai_api(
                messages=messages,
                model=openai_model,
                seed=openai_seed,
                max_tokens=openai_max_tokens,
                temperature=openai_temperature,
                verbose=verbose,
            )
            return parse_score(output)
        except Exception as e:
            if attempt < max_tries - 1:
                eval_logger.warning(f"LLM evaluation failed (attempt {attempt + 1}/{max_tries}): {e}")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                eval_logger.error(f"LLM evaluation failed after {max_tries} attempts: {e}")
                return 0  # Return 0 score if all attempts fail


def load_prompt():
    prompt = """
    You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
    To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
    5 means that the response perfectly matches the answer.
    1 means that the response is completely different from the answer.

    Example 1:
    Question: Is it overcast?
    Answer: no
    Response: yes
    Your mark: 1

    Example 2:
    Question: Who is standing at the table?
    Answer: woman
    Response: Jessica
    Your mark: 3

    Example 3:
    Question: Are there drapes to the right of the bed?
    Answer: yes
    Response: yes
    Your mark: 5

    Your Turn:
    Question: {question}
    Answer: {answer}
    Response: {prediction}    
    """
    return prompt

def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]

def call_openai_api(
    messages: list,
    model: str = "gpt-4o",
    seed = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content


def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())

def post_evaluate_results(sample_file_path, results_file_path):
    import json
    with open(sample_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    for doc in tqdm(data):
        pred_raw = doc["resps"][0][0] if doc["resps"] and doc["resps"][0] else ""
        target = doc["target"]
        question_type = doc["question_type"]
        question = doc.get("question", doc.get("doc", ""))

        result_dict = {"target": target}
        result_dict["question_type"] = question_type
        
        for key, value in METRICS_FOR_MSQA.items():
            pred = pred_raw
            # Call the evaluation function directly with the new signature
            score = eval(value)(question, target, pred)
            doc[key] = {key: score}
            result_dict[key] = doc[key]

        results.append(result_dict)

    # samples_robovqa.json
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    output = msqa_aggregate_results(results)

    with open(results_file_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    base_dir = "/data/arena/datasets/msr3d/msqa"
    post_evaluate_results(
        sample_file_path=f"{base_dir}/msqa.json",
        results_file_path=f"{base_dir}/results_msqa.json"
    )