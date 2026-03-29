from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import time
import os
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test Step3-VL-10B model")
parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate (default: 1024)")
parser.add_argument("--quick-test", action="store_true", help="Quick test with only 50 tokens")
parser.add_argument("--use-flash-attention-2", action="store_true", help="Use Flash Attention 2 for faster inference")
args = parser.parse_args()

# Add model directory to Python path (resolve relative imports)
model_path = "/your/path/to/embodied-eval-main/embodied_eval/data/Step3-VL-10B"
if os.path.exists(model_path) and model_path not in sys.path:
    sys.path.insert(0, model_path)
    print(f"Added model directory to Python path: {model_path}")

# Set max_new_tokens based on arguments
if args.quick_test:
    max_new_tokens = 50
    print("Quick test mode: using max_new_tokens=50")
else:
    max_new_tokens = args.max_tokens

key_mapping = {
    "^vision_model": "model.vision_model",
    r"^model(?!\.(language_model|vision_model))": "model.language_model",
    "vit_large_projector": "model.vit_large_projector",
}

print("=" * 60)
print("Step3-VL-10B Official Usage Test")
print("=" * 60)

print("\n[1/5] Loading processor...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print("✓ Processor loaded successfully")

print("\n[2/5] Loading model...")
if args.use_flash_attention_2:
    print("  Attempting to load with Flash Attention 2 (faster inference)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            key_mapping=key_mapping,
            attn_implementation="flash_attention_2"
        ).eval()
        print("  ✓ Successfully loaded with Flash Attention 2")
    except (ImportError, Exception) as e:
        print(f"  ⚠ Failed to load with Flash Attention 2: {e}")
        print("  Falling back to standard attention...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
            key_mapping=key_mapping
        ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto",
        key_mapping=key_mapping
    ).eval()
print(f"✓ Model loaded successfully on device: {model.device}")

print("\n[3/5] Preparing messages...")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "What's in this picture?"}
        ]
    },
]
print("✓ Messages prepared")

print("\n[4/5] Applying chat template...")
try:
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    print(f"✓ Chat template applied successfully")
    print(f"  Input keys: {list(inputs.keys())}")
    input_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    print(f"  Input length: {input_length} tokens")
    if "pixel_values" in inputs:
        print(f"  Image shape: {inputs['pixel_values'].shape}")
except Exception as e:
    print(f"✗ Error applying chat template: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[5/5] Generating response...")
print("  Parameters: max_new_tokens=1024, do_sample=False")
print("  (This may take a while for large models...)")

# First, test with a small max_new_tokens to verify generation works
print("\n  Testing with small max_new_tokens (10) to verify generation works...")
test_start = time.time()
try:
    with torch.no_grad():
        test_ids = model.generate(
            **inputs,
            max_new_tokens=10,  # Small test first
            do_sample=False,
            use_cache=True
        )
    test_time = time.time() - test_start
    test_output_length = test_ids.shape[-1] - input_length
    print(f"  ✓ Test generation completed in {test_time:.2f}s (generated {test_output_length} tokens)")
    # Decode a small sample to verify it's working
    test_decoded = processor.decode(
        test_ids[0, inputs["input_ids"].shape[-1]:], 
        skip_special_tokens=True
    )
    print(f"  Sample output: {test_decoded[:50]}...")
except Exception as e:
    test_time = time.time() - test_start
    print(f"  ✗ Test generation failed (after {test_time:.2f}s): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now do the full generation
print(f"\n  Starting full generation (max_new_tokens={max_new_tokens})...")
if max_new_tokens >= 1024:
    print("  (Please wait, this may take 10-30 minutes for large models with long outputs)")
else:
    print(f"  (Estimated time: ~{max_new_tokens * 2.8 / 60:.1f} minutes based on test speed)")
start_time = time.time()
try:
    with torch.no_grad():
        # Use minimal parameters as per official usage
        # Note: We can't easily add progress monitoring without streamer,
        # but we can at least catch errors and show timing
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True  # Enable cache for better performance
        )
    elapsed_time = time.time() - start_time
    output_length = generate_ids.shape[-1] - input_length
    print(f"✓ Generation completed in {elapsed_time:.2f} seconds")
    print(f"  Generated {output_length} tokens")
    print(f"  Output shape: {generate_ids.shape}")
except KeyboardInterrupt:
    elapsed_time = time.time() - start_time
    print(f"\n✗ Generation interrupted by user (after {elapsed_time:.2f}s)")
    sys.exit(1)
except Exception as e:
    elapsed_time = time.time() - start_time
    print(f"✗ Error during generation (after {elapsed_time:.2f}s): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nDecoding response...")
try:
    decoded = processor.decode(
        generate_ids[0, inputs["input_ids"].shape[-1]:], 
        skip_special_tokens=True
    )
    print("✓ Decoding completed")
except Exception as e:
    print(f"✗ Error during decoding: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("RESPONSE:")
print("=" * 60)
print(decoded)
print("=" * 60)
print("\n✓ Test completed successfully!")
