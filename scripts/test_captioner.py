#!/usr/bin/env python3
"""
Test the captioner on sample images to understand output format.
Run this first to see what vocabulary/style the model actually uses.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Model options - try these in order of preference
MODELS = [
    "Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5",
    "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it",
    "Qwen/Qwen3-VL-8B-Instruct",  # Fallback to base
]


def load_model(model_id: str = None):
    """Load model, trying alternatives if needed."""

    models_to_try = [model_id] if model_id else MODELS

    for mid in models_to_try:
        try:
            print(f"Trying to load: {mid}")

            # Check VRAM
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"Available VRAM: {vram_gb:.1f} GB")

                if vram_gb >= 20:
                    dtype = torch.bfloat16
                    quantize = False
                else:
                    dtype = torch.bfloat16
                    quantize = True
            else:
                print("No CUDA - using CPU (will be slow)")
                dtype = torch.float32
                quantize = False

            model_kwargs = {
                "torch_dtype": dtype,
                "device_map": "auto",
                "trust_remote_code": True,
            }

            if quantize:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                print("Using 4-bit quantization")

            model = Qwen2VLForConditionalGeneration.from_pretrained(mid, **model_kwargs)
            processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)

            print(f"Loaded: {mid}")
            return model, processor, mid

        except Exception as e:
            print(f"Failed to load {mid}: {e}")
            continue

    raise RuntimeError("Could not load any model")


def caption_image(model, processor, image_path: Path, prompt: str = None) -> str:
    """Generate caption for an image."""

    if prompt is None:
        prompt = "Describe this image in detail."

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    caption = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return caption.strip()


def find_sample_images(directory: Path, limit: int = 10) -> list[Path]:
    """Find sample images to test with."""

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []

    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
        images.extend(directory.rglob(f"*{ext.upper()}"))

    images = sorted(set(images))[:limit]
    return images


def main():
    parser = argparse.ArgumentParser(description="Test captioner on sample images")
    parser.add_argument(
        "--images",
        type=str,
        help="Directory containing test images, or path to single image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max images to process"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt to use"
    )
    args = parser.parse_args()

    # Find images
    if args.images:
        img_path = Path(args.images)
        if img_path.is_file():
            images = [img_path]
        else:
            images = find_sample_images(img_path, args.limit)
    else:
        # Try to find images in data/raw
        data_dir = PROJECT_DIR / "data" / "raw"
        if data_dir.exists():
            images = find_sample_images(data_dir, args.limit)
        else:
            print("No images specified and no data/raw directory found.")
            print("Usage: python test_captioner.py --images /path/to/images")
            return

    if not images:
        print("No images found!")
        return

    print(f"Found {len(images)} images to test")

    # Load model
    model, processor, model_id = load_model(args.model)

    # Process images
    results = []

    for i, img_path in enumerate(images):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(images)}] {img_path.name}")
        print('='*60)

        try:
            caption = caption_image(model, processor, img_path, args.prompt)
            print(f"\nCAPTION:\n{caption}")

            results.append({
                "image": str(img_path),
                "caption": caption,
            })

        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "image": str(img_path),
                "error": str(e),
            })

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"caption_test_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({
            "model": model_id,
            "prompt": args.prompt or "Describe this image in detail.",
            "results": results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
