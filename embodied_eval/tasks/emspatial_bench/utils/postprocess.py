import argparse
import json
import os
import re
from loguru import logger as eval_logger

# Reuse aggregation from process module
from embodied_eval.tasks.emspatial_bench.process import emspatial_aggregate_results






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSONL file containing model predictions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory to save post-processed results.",
    )
    args = parser.parse_args()
    input_file = args.input_file
    output_dir = args.output_dir

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    os.makedirs(output_dir, exist_ok=True)

    eval_logger.info(f"Loading inference file: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    eval_logger.info(f"Loaded {len(data)} records from inference file")

    results = []

    for doc in data:
        # original target is stored as numeric option index (0->A,1->B,...)
        # Try common keys where target might be stored
        target_idx = None
        for k in ("target", "answer", "ans", "label"):
            if k in doc and doc.get(k) is not None:
                try:
                    target_idx = int(doc.get(k))
                except Exception:
                    # sometimes stored as string of index
                    try:
                        target_idx = int(str(doc.get(k)).strip())
                    except Exception:
                        target_idx = None
                break

        # Preprocess model responses: remove any <think>...</think> blocks
        if isinstance(doc.get("resps"), list) and doc["resps"]:
            cleaned_resps = []
            for entry in doc["resps"]:
                # entry may be a list or string
                text = entry[0] if isinstance(entry, (list, tuple)) and entry else (entry if isinstance(entry, str) else "")
                # remove <think>...</think> blocks
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
                # also collapse multiple newlines/spaces
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    cleaned_resps.append(text)
            doc["resps_cleaned"] = cleaned_resps

        # Extract model response raw text (prefer cleaned resps, then prediction)
        pred_raw = ""
        if doc.get("resps_cleaned"):
            pred_raw = doc["resps_cleaned"][0]
        elif isinstance(doc.get("resps"), list) and doc["resps"] and doc["resps"][0]:
            try:
                pred_raw = doc["resps"][0][0]
            except Exception:
                pred_raw = str(doc["resps"][0])
        elif doc.get("prediction"):
            pred_raw = doc.get("prediction")

        pred_idx = None
        if pred_raw:
            # Strategy: letter A-D first
            m = re.search(r"\b([A-Da-d])\b", pred_raw)
            if m:
                pred_idx = ord(m.group(1).upper()) - ord('A')
            else:
                # number forms 1-4 -> index 0-3
                m2 = re.search(r"\b([1-4])\b", pred_raw)
                if m2:
                    pred_idx = int(m2.group(1)) - 1

        # Compute simple score
        score = 0.0
        if target_idx is None:
            eval_logger.warning(f"Record missing numeric target; doc_id={doc.get('doc_id')} - skipping scoring")
        else:
            if pred_idx is None:
                score = 0.0
            else:
                score = 1.0 if pred_idx == target_idx else 0.0

        # Attach prediction and score back to doc in same shape as process expects
        doc["prediction"] = pred_raw
        doc["accuracy"] = {"accuracy": score}

        # Prepare result dict for aggregation
        relation = doc.get("relation", "unknown")
        results.append({"target": doc.get("target"), "relation": relation, "accuracy": {"accuracy": score}})

    # Save updated samples file
    samples_out = os.path.join(output_dir, "samples_emspatial-bench.json")
    with open(samples_out, 'w', encoding='utf-8') as f:
        for doc in data:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    eval_logger.info(f"Wrote updated samples to {samples_out}")

    # Aggregate results using existing function
    agg = emspatial_aggregate_results(results)
    results_out = os.path.join(output_dir, "results_emspatial-bench.json")
    with open(results_out, 'w', encoding='utf-8') as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    eval_logger.info(f"Wrote aggregated results to {results_out}")

    