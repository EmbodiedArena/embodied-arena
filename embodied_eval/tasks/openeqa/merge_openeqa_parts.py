"""
Merge OpenEQA (openeqa-emeqa) parallel part outputs and aggregate results.

Usage (run from repo root):
  python embodied_eval/tasks/openeqa/merge_openeqa_parts.py <output_base> <run_id> <num_parts>

Example:
  python embodied_eval/tasks/openeqa/merge_openeqa_parts.py $(pwd)/logs/openeqa/iflybot_vlm 20260311_020000 4

Notes:
  - This script does NOT call any LLM judge API.
  - It aggregates using the scores already stored in samples_openeqa-emeqa.json (jsonl).
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict


OPENEQA_EMEQA_QUESTION_TYPES = [
    "attribute recognition",
    "functional reasoning",
    "object localization",
    "object recognition",
    "object state recognition",
    "spatial understanding",
    "world knowledge",
]


def load_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def dump_jsonl(path: str, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _extract_llm_match_score(sample: dict) -> int:
    """
    Expected shapes seen in this repo:
      - sample["llm_match_score"] = {"llm_match_score": 1..5}
    """
    val = sample.get("llm_match_score")
    if isinstance(val, dict):
        inner = val.get("llm_match_score")
        if isinstance(inner, (int, float)):
            return int(inner)
    if isinstance(val, (int, float)):
        return int(val)
    return 0


def aggregate_openeqa_emeqa(samples: list[dict]) -> dict:
    # Group scores by question_type (category)
    scores_by_type: dict[str, list[int]] = defaultdict(list)
    all_scores: list[int] = []

    for s in samples:
        qtype = s.get("question_type") or s.get("category") or ""
        score = _extract_llm_match_score(s)
        all_scores.append(score)
        if qtype:
            scores_by_type[qtype].append(score)

    output: dict[str, float] = {}

    # Per-type averages (only for the official known types to stay consistent with process.py)
    per_type_avgs: list[float] = []
    for qtype in OPENEQA_EMEQA_QUESTION_TYPES:
        scores = scores_by_type.get(qtype, [])
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        output[f"{qtype}_llm_match_score"] = avg
        per_type_avgs.append(avg)

    # Averages
    if per_type_avgs:
        output["llm_match_score_per_type_average"] = sum(per_type_avgs) / len(per_type_avgs)
    if all_scores:
        output["llm_match_score_all_samples_average"] = sum(all_scores) / len(all_scores)

    # Match the existing (slightly unusual) behavior in openeqa/process.py:
    # overall = mean of all numeric values currently in output.
    numeric_vals = [v for v in output.values() if isinstance(v, (int, float))]
    output["overall"] = (sum(numeric_vals) / len(numeric_vals)) if numeric_vals else 0.0

    return output


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: python embodied_eval/tasks/openeqa/merge_openeqa_parts.py <output_base> <run_id> <num_parts>")
        sys.exit(1)

    output_base = os.path.abspath(sys.argv[1])
    run_id = sys.argv[2]
    num_parts = int(sys.argv[3])

    task_name = "openeqa-emeqa"
    run_dir = os.path.join(output_base, run_id)
    samples_file = f"samples_{task_name}.json"

    all_samples: list[dict] = []
    for part in range(num_parts):
        part_dir = os.path.join(run_dir, f"part_{part}")
        path = os.path.join(part_dir, samples_file)
        if not os.path.isfile(path):
            print(f"Warning: {path} not found, skip part {part}")
            continue
        part_samples = load_jsonl(path)
        all_samples.extend(part_samples)
        print(f"  part_{part}: {len(part_samples)} samples")

    if not all_samples:
        print(f"Error: no part samples found under: {run_dir}/part_*/{samples_file}")
        sys.exit(1)

    all_samples.sort(key=lambda x: x.get("doc_id", 0))

    os.makedirs(run_dir, exist_ok=True)
    merged_path = os.path.join(run_dir, samples_file)
    dump_jsonl(merged_path, all_samples)
    print(f"Merged {len(all_samples)} samples -> {merged_path}")

    aggregated = aggregate_openeqa_emeqa(all_samples)
    results_path = os.path.join(run_dir, f"results_{task_name}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=4)
    print(f"Aggregated results -> {results_path} (overall: {aggregated.get('overall', 'N/A')})")

    config_src = os.path.join(run_dir, "part_0", f"configs_{task_name}.json")
    if os.path.isfile(config_src):
        import shutil

        config_dst = os.path.join(run_dir, f"configs_{task_name}.json")
        shutil.copy2(config_src, config_dst)
        print(f"Config copied -> {config_dst}")

    print("Done.")


if __name__ == "__main__":
    main()

