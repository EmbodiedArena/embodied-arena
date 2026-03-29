#!/usr/bin/env python3
"""Remove invalid samples from EmbodiedScene dataset JSON.

A sample is considered invalid when:
1) it has no image path in `images_path`, or
2) any image in `images_path` does not exist under the given images root.

Usage examples:
    python3 clean_invalid_dataset.py \
      --json-path embodied_eval/data/EmbodiedScene/embodied_scene_data.json \
      --images-root embodied_eval/data/EmbodiedScene/images \
      --dry-run

    python3 clean_invalid_dataset.py \
      --json-path embodied_eval/data/EmbodiedScene/embodied_scene_data.json \
      --images-root embodied_eval/data/EmbodiedScene/images \
      --in-place
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter out samples whose images are missing."
    )
    parser.add_argument(
        "--json-path",
        required=True,
        type=Path,
        help="Path to dataset JSON file.",
    )
    parser.add_argument(
        "--images-root",
        required=True,
        type=Path,
        help="Root directory where images are stored.",
    )
    parser.add_argument(
        "--images-field",
        default="images_path",
        help="Field name that stores image path(s). Default: images_path",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output JSON path. If omitted, writes to "
            "<json_path>.cleaned.json unless --in-place is used."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input JSON directly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print statistics, do not write output JSON.",
    )
    parser.add_argument(
        "--show-missing-limit",
        type=int,
        default=20,
        help="How many missing image paths to print. Default: 20",
    )
    return parser.parse_args()


def normalize_image_refs(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        refs: List[str] = []
        for item in value:
            if isinstance(item, str):
                refs.append(item)
        return refs
    return []


def to_candidate_paths(ref: str, images_root: Path, json_dir: Path) -> Sequence[Path]:
    ref = ref.strip()
    if not ref:
        return []

    p = Path(ref)
    candidates: List[Path] = []

    if p.is_absolute():
        candidates.append(p)

    # Keep original relative interpretation (relative to JSON location).
    candidates.append(json_dir / ref)

    # Map common prefixes to images_root.
    normalized = ref.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]

    if normalized.startswith("images/"):
        normalized = normalized[len("images/") :]
    elif "/images/" in normalized:
        normalized = normalized.split("/images/", 1)[1]

    if normalized:
        candidates.append(images_root / normalized)

    # Deduplicate while preserving order.
    unique: List[Path] = []
    seen = set()
    for c in candidates:
        try:
            key = str(c.resolve(strict=False))
        except RuntimeError:
            key = str(c)
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def find_missing_images(
    image_refs: Iterable[str], images_root: Path, json_dir: Path
) -> List[str]:
    missing: List[str] = []
    for ref in image_refs:
        candidates = to_candidate_paths(ref, images_root=images_root, json_dir=json_dir)
        if not candidates or not any(path.exists() for path in candidates):
            missing.append(ref)
    return missing


def default_output_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}.cleaned{json_path.suffix}")


def main() -> None:
    args = parse_args()

    json_path = args.json_path.expanduser().resolve()
    images_root = args.images_root.expanduser().resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected top-level JSON to be a list, got: {type(data).__name__}"
        )

    json_dir = json_path.parent
    valid_samples = []
    invalid_samples = []
    all_missing_refs: List[Tuple[str, str]] = []

    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            invalid_samples.append((idx, sample, ["<non-dict sample>"]))
            all_missing_refs.append((f"index={idx}", "<non-dict sample>"))
            continue

        refs = normalize_image_refs(sample.get(args.images_field))
        if not refs:
            invalid_samples.append((idx, sample, ["<empty images_path>"]))
            qid = str(sample.get("question_id", f"index={idx}"))
            all_missing_refs.append((qid, "<empty images_path>"))
            continue

        missing = find_missing_images(refs, images_root=images_root, json_dir=json_dir)
        if missing:
            invalid_samples.append((idx, sample, missing))
            qid = str(sample.get("question_id", f"index={idx}"))
            for m in missing:
                all_missing_refs.append((qid, m))
        else:
            valid_samples.append(sample)

    total = len(data)
    invalid_count = len(invalid_samples)
    valid_count = len(valid_samples)

    print(f"Input JSON: {json_path}")
    print(f"Images root: {images_root}")
    print(f"Total samples: {total}")
    print(f"Valid samples: {valid_count}")
    print(f"Invalid samples removed: {invalid_count}")

    if all_missing_refs:
        print()
        print("Examples of missing image references:")
        for qid, missing_ref in all_missing_refs[: max(0, args.show_missing_limit)]:
            print(f"  question_id={qid} -> {missing_ref}")

    if args.dry_run:
        print()
        print("Dry run enabled: no file written.")
        return

    if args.in_place and args.output_path is not None:
        raise ValueError("Use either --in-place or --output-path, not both.")

    output_path = (
        json_path
        if args.in_place
        else (
            args.output_path.expanduser().resolve()
            if args.output_path
            else default_output_path(json_path)
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(valid_samples, f, ensure_ascii=False, indent=2)

    print()
    print(f"Filtered JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
