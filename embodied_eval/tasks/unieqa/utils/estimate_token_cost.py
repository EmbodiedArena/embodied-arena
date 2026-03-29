#!/usr/bin/env python3
"""
Estimate expected token usage for UniEQA-style samples.

Notes:
- Text tokens are estimated via tiktoken if available; otherwise a rough
  4 chars/token heuristic is used.
- Image tokens depend on the provider/model. Pass --image_token_per_image
  using your provider's guidance (default is 0, i.e., not counted).
"""
import argparse
import random
from typing import List


def _estimate_text_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback heuristic: ~4 chars per token
        return max(1, len(text) // 4)


def _cap_images(images: List[str], cap: int | None) -> int:
    if cap is None:
        return len(images)
    return min(len(images), cap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate expected token usage/cost.")
    parser.add_argument("--dataset_path", required=True, help="Path to HF dataset (load_from_disk).")
    parser.add_argument("--split", default="train", help="Dataset split name.")
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of samples to estimate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--cap_images", type=int, default=None, help="Max images per sample (e.g., 10/15).")
    parser.add_argument("--image_token_per_image", type=float, default=0.0,
                        help="Estimated tokens per image from provider docs.")
    parser.add_argument("--avg_output_tokens", type=float, default=0.0,
                        help="Estimated average output tokens per sample.")
    parser.add_argument("--input_rate", type=float, default=0.002, help="Input $/1K tokens.")
    parser.add_argument("--output_rate", type=float, default=0.012, help="Output $/1K tokens.")
    args = parser.parse_args()

    from datasets import load_from_disk  # lazy import

    ds = load_from_disk(args.dataset_path)
    if hasattr(ds, "keys") and args.split in ds:
        ds = ds[args.split]

    total = len(ds)
    sample_size = min(args.sample_size, total)
    random.seed(args.seed)
    indices = random.sample(range(total), sample_size)

    text_tokens_sum = 0
    image_tokens_sum = 0
    image_count_sum = 0

    for idx in indices:
        ex = ds[idx]
        question = ex.get("question", "") or ""
        images = ex.get("images", []) or []
        capped_count = _cap_images(images, args.cap_images)

        text_tokens_sum += _estimate_text_tokens(question)
        image_tokens_sum += capped_count * args.image_token_per_image
        image_count_sum += capped_count

    avg_text_tokens = text_tokens_sum / sample_size
    avg_image_tokens = image_tokens_sum / sample_size
    avg_input_tokens = avg_text_tokens + avg_image_tokens

    avg_output_tokens = args.avg_output_tokens
    input_cost = (avg_input_tokens / 1000.0) * args.input_rate
    output_cost = (avg_output_tokens / 1000.0) * args.output_rate
    total_cost = input_cost + output_cost

    print("=== Token/Cost Estimate ===")
    print(f"samples_used: {sample_size} / {total}")
    print(f"cap_images: {args.cap_images}")
    print(f"avg_images_per_sample: {image_count_sum / sample_size:.3f}")
    print(f"avg_text_tokens: {avg_text_tokens:.2f}")
    print(f"avg_image_tokens: {avg_image_tokens:.2f} (per_image={args.image_token_per_image})")
    print(f"avg_input_tokens: {avg_input_tokens:.2f}")
    print(f"avg_output_tokens: {avg_output_tokens:.2f}")
    print(f"cost_per_sample: ${total_cost:.6f}")
    print(f"  - input_cost:  ${input_cost:.6f}")
    print(f"  - output_cost: ${output_cost:.6f}")


if __name__ == "__main__":
    main()
