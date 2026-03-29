#!/usr/bin/env python3
"""Fill empty model responses in EmbSpatial-Bench samples JSONL.

This is designed for runs where network/API issues caused blank responses
(e.g., ``resps`` equals ``[[""]]``).

Given a log directory that contains ``samples_emspatial-bench.json``, this script:

- Finds samples whose response text is missing/empty/whitespace
- Reloads the corresponding dataset item (by ``doc_id``) to obtain the image
- Re-sends the stored ``doc`` prompt to a model via ``OpenAIAsyncCompatible``
- Writes the model response back into ``resps`` with the same nested-list format
- Leaves all non-empty samples untouched

By default it writes the updated samples file in-place and creates a timestamped
backup next to it.

Environment:
- ``OPENAI_API_KEY`` / ``OPENAI_API_BASE``: required by ``OpenAIAsyncCompatible``
- ``EMSPATIAL_DATASET_PATH``: optional override for dataset root directory
- ``EMSPATIAL_REFILL_MODEL``: optional default model name
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger


# Allow running this file directly via path (python path/to/refill_empty_resps.py)
# without requiring callers to set PYTHONPATH or run as a module.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))


def coerce_item_image_to_pil(item: Dict[str, Any]):
	"""Best-effort conversion of a dataset item's image into a PIL.Image.

	The parquet loader may yield:
	- PIL.Image.Image
	- dict with 'bytes' or 'path'
	- base64 string in 'image_base64'
	"""
	try:
		from PIL import Image
	except Exception:
		return None

	img = item.get("image")
	if isinstance(img, Image.Image):
		return img.convert("RGB") if img.mode != "RGB" else img

	# datasets Image feature often serializes as dict
	if isinstance(img, dict):
		if "bytes" in img and img["bytes"]:
			try:
				from io import BytesIO

				pil = Image.open(BytesIO(img["bytes"])).convert("RGB")
				return pil
			except Exception:
				pass
		if "path" in img and img["path"]:
			try:
				pil = Image.open(img["path"]).convert("RGB")
				return pil
			except Exception:
				pass

	# Fallback to base64
	b64 = item.get("image_base64")
	if isinstance(b64, str) and b64.strip():
		try:
			import base64
			from io import BytesIO

			raw = b64.strip()
			if raw.startswith("data:image") and "," in raw:
				raw = raw.split(",", 1)[1]
			image_bytes = base64.b64decode(raw)
			pil = Image.open(BytesIO(image_bytes)).convert("RGB")
			return pil
		except Exception:
			pass

	return None


def _read_yaml_dataset_path() -> Optional[Path]:
	"""Read dataset_path from emspatial-bench.yaml (best-effort).

	Note: the task yaml contains custom tags like `!function` which are not
	supported by `yaml.safe_load` by default. We therefore parse the file as
	plain text and extract the `dataset_path:` line.
	"""
	yaml_path = Path(__file__).resolve().parents[1] / "emspatial-bench.yaml"
	if not yaml_path.exists():
		return None
	try:
		for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
			line = raw_line.strip()
			if not line or line.startswith("#"):
				continue
			if not line.lower().startswith("dataset_path"):
				continue
			# Split at the first ':' and strip inline comments
			parts = raw_line.split(":", 1)
			if len(parts) != 2:
				continue
			value = parts[1].strip()
			if "#" in value:
				value = value.split("#", 1)[0].strip()
			# Strip quotes if present
			if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
				value = value[1:-1].strip()
			if not value:
				continue
			value = os.path.expanduser(os.path.expandvars(value))
			return Path(value)
	except Exception as e:
		logger.warning(f"Failed to read dataset_path from {yaml_path}: {e}")
		return None

	return None


def resolve_dataset_root() -> Optional[Path]:
	env_path = os.getenv("EMSPATIAL_DATASET_PATH")
	if env_path:
		return Path(env_path)
	return _read_yaml_dataset_path()


def resolve_parquet_file(dataset_root: Path, split: str = "test") -> Path:
	"""Resolve the parquet file path for the dataset split."""
	data_dir = dataset_root / "data"
	if not data_dir.exists():
		raise FileNotFoundError(f"Dataset data dir not found: {data_dir}")

	matches = sorted(data_dir.glob(f"{split}-*.parquet"))
	if not matches:
		matches = sorted(data_dir.glob("*.parquet"))
	if not matches:
		raise FileNotFoundError(f"No parquet file found under: {data_dir}")

	return matches[0]


def load_emspatial_dataset(dataset_root: Path, split: str = "test"):
	from datasets import load_dataset

	parquet_path = resolve_parquet_file(dataset_root, split=split)
	logger.info(f"Loading dataset split='{split}' from parquet: {parquet_path}")
	ds = load_dataset("parquet", data_files={split: str(parquet_path)})
	return ds[split]


def get_response_text(sample: Dict[str, Any]) -> Optional[str]:
	"""Return the first response text if structure matches, else None."""
	resps = sample.get("resps")
	if resps is None:
		return None
	if not isinstance(resps, list) or len(resps) == 0:
		return None
	first = resps[0]
	if not isinstance(first, list) or len(first) == 0:
		return None
	text = first[0]
	if text is None:
		return None
	return str(text)


def is_empty_response(sample: Dict[str, Any]) -> bool:
	text = get_response_text(sample)
	return text is None or text.strip() == ""


def set_response_text(sample: Dict[str, Any], new_text: str) -> None:
	"""Set sample['resps'] in-place while preserving existing structure."""
	if "resps" in sample and isinstance(sample.get("resps"), list) and sample["resps"]:
		if isinstance(sample["resps"][0], list) and sample["resps"][0]:
			sample["resps"][0][0] = new_text
			return
		if isinstance(sample["resps"][0], list) and len(sample["resps"][0]) == 0:
			sample["resps"][0].append(new_text)
			return
	sample["resps"] = [[new_text]]


@dataclass
class RefillConfig:
	log_dir: Path
	sample_file: Path
	model_name: str
	max_new_tokens: int
	temperature: float
	timeout: int
	max_retries: int
	batch_size: int
	dataset_split: str
	dry_run: bool
	backup: bool
	retry_on_blank: int


def build_model(cfg: RefillConfig):
	from embodied_eval.models.openai_async_compatible import OpenAIAsyncCompatible

	return OpenAIAsyncCompatible(
		model_name_or_path=cfg.model_name,
		batch_size=cfg.batch_size,
		max_new_tokens=cfg.max_new_tokens,
		temperature=cfg.temperature,
		timeout=cfg.timeout,
		max_retries=cfg.max_retries,
		max_frames_num=32,
	)


def _batched(seq: Sequence[int], batch_size: int) -> List[List[int]]:
	out: List[List[int]] = []
	cur: List[int] = []
	for x in seq:
		cur.append(x)
		if len(cur) >= batch_size:
			out.append(cur)
			cur = []
	if cur:
		out.append(cur)
	return out


def _backup_file(path: Path) -> Path:
	ts = time.strftime("%Y%m%d_%H%M%S")
	bak = path.with_suffix(path.suffix + f".bak_{ts}")
	bak.write_bytes(path.read_bytes())
	return bak


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Refill empty responses in EmbSpatial-Bench samples JSONL",
	)
	parser.add_argument("--log_dir", type=str, required=True, help="log 目录，包含 samples_emspatial-bench.json")
	parser.add_argument("--sample_file", type=str, default=None, help="samples 文件路径（优先级高于 log_dir）")
	parser.add_argument(
		"--model",
		type=str,
		default=os.getenv("EMSPATIAL_REFILL_MODEL", "gemini-2.5-pro"),
		help="用于回填的模型名（OpenAI 兼容 chat.completions 模型名）",
	)
	parser.add_argument("--max_new_tokens", type=int, default=int(os.getenv("EMSPATIAL_REFILL_MAX_NEW_TOKENS", "512")))
	parser.add_argument("--temperature", type=float, default=float(os.getenv("EMSPATIAL_REFILL_TEMPERATURE", "0")))
	parser.add_argument("--timeout", type=int, default=int(os.getenv("EMSPATIAL_REFILL_TIMEOUT", "60")))
	parser.add_argument("--max_retries", type=int, default=int(os.getenv("EMSPATIAL_REFILL_MAX_RETRIES", "3")))
	parser.add_argument("--batch_size", type=int, default=int(os.getenv("EMSPATIAL_REFILL_BATCH_SIZE", "8")))
	parser.add_argument("--dataset_split", type=str, default=os.getenv("EMSPATIAL_DATASET_SPLIT", "test"))
	parser.add_argument("--dry_run", action="store_true", help="只统计空回答，不调用模型，不写回")
	parser.add_argument("--no_backup", action="store_true", help="不备份原 samples 文件")
	parser.add_argument(
		"--retry_on_blank",
		type=int,
		default=int(os.getenv("EMSPATIAL_REFILL_RETRY_ON_BLANK", "1")),
		help="如果模型仍返回空字符串，最多额外重试次数（逐条重试）",
	)

	args = parser.parse_args()

	log_dir = Path(args.log_dir)
	sample_file = Path(args.sample_file) if args.sample_file else (log_dir / "samples_emspatial-bench.json")
	if not sample_file.exists():
		logger.error(f"samples file not found: {sample_file}")
		return 2

	cfg = RefillConfig(
		log_dir=log_dir,
		sample_file=sample_file,
		model_name=args.model,
		max_new_tokens=args.max_new_tokens,
		temperature=args.temperature,
		timeout=args.timeout,
		max_retries=args.max_retries,
		batch_size=max(1, int(args.batch_size)),
		dataset_split=str(args.dataset_split),
		dry_run=bool(args.dry_run),
		backup=not bool(args.no_backup),
		retry_on_blank=max(0, int(args.retry_on_blank)),
	)

	logger.info(f"Reading samples: {cfg.sample_file}")
	samples: List[Dict[str, Any]] = []
	with cfg.sample_file.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			samples.append(json.loads(line))

	empty_indices = [i for i, s in enumerate(samples) if is_empty_response(s)]
	empty_doc_ids = [int(samples[i].get("doc_id")) for i in empty_indices if samples[i].get("doc_id") is not None]

	logger.info(f"Total samples: {len(samples)}")
	logger.info(f"Empty responses: {len(empty_indices)}")
	if empty_indices:
		logger.info(f"First 10 empty doc_ids: {empty_doc_ids[:10]}")

	if cfg.dry_run or not empty_indices:
		logger.info("Dry-run mode or no empties; exiting without changes.")
		return 0

	dataset_root = resolve_dataset_root()
	if dataset_root is None:
		logger.error("Cannot resolve dataset root. Set EMSPATIAL_DATASET_PATH or ensure emspatial-bench.yaml exists.")
		return 3

	try:
		ds = load_emspatial_dataset(dataset_root, split=cfg.dataset_split)
	except Exception as e:
		logger.exception(f"Failed to load dataset: {e}")
		return 4

	model = build_model(cfg)

	from tqdm import tqdm

	updated = 0
	failed = 0

	for batch in tqdm(_batched(empty_indices, cfg.batch_size), desc="Refilling empty responses"):
		contexts: List[str] = []
		visuals_list: List[List[Any]] = []
		mapping: List[int] = []

		for idx in batch:
			s = samples[idx]
			doc_id = s.get("doc_id")
			if doc_id is None:
				failed += 1
				continue

			context = str(s.get("doc", ""))
			if not context.strip():
				failed += 1
				continue

			img = None
			try:
				item = ds[int(doc_id)]
				img = coerce_item_image_to_pil(item)
			except Exception as e:
				logger.warning(f"Failed to fetch dataset item for doc_id={doc_id}: {e}")

			contexts.append(context)
			visuals_list.append([img] if img is not None else [])
			mapping.append(idx)

		if not mapping:
			continue

		try:
			responses = model.batch_respond(contexts, visuals_list)
		except Exception as e:
			logger.exception(f"Batch API call failed: {e}")
			failed += len(mapping)
			continue

		for sample_idx, resp in zip(mapping, responses):
			resp_text = (resp or "")
			if resp_text.strip() == "" and cfg.retry_on_blank > 0:
				doc_id = samples[sample_idx].get("doc_id")
				context = str(samples[sample_idx].get("doc", ""))
				img = None
				try:
					item = ds[int(doc_id)]
					img = coerce_item_image_to_pil(item)
				except Exception:
					pass

				for _ in range(cfg.retry_on_blank):
					try:
						resp_text = model.respond(context, [img] if img is not None else []) or ""
					except Exception:
						resp_text = ""
					if resp_text.strip() != "":
						break

			if resp_text.strip() == "":
				failed += 1
				continue

			set_response_text(samples[sample_idx], resp_text)
			updated += 1

	logger.info(f"Updated samples: {updated}")
	logger.info(f"Failed refills: {failed}")

	if cfg.backup:
		bak = _backup_file(cfg.sample_file)
		logger.info(f"Backup created: {bak}")

	tmp_path = cfg.sample_file.with_suffix(cfg.sample_file.suffix + ".tmp")
	with tmp_path.open("w", encoding="utf-8") as f:
		for s in samples:
			f.write(json.dumps(s, ensure_ascii=False) + "\n")

	tmp_path.replace(cfg.sample_file)
	logger.info(f"Wrote updated samples: {cfg.sample_file}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())