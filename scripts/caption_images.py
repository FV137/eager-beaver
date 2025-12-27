#!/usr/bin/env python3
"""
Caption images using Qwen3-VL-8B-NSFW-Caption model.
Generates detailed captions with consistent vocabulary.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

# Project paths
PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
CONFIG_DIR = PROJECT_DIR / "configs"
OUTPUT_DIR = DATA_DIR / "processed"

# Model config
MODEL_ID = "Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5"


def load_taxonomy():
    """Load vocabulary taxonomy for consistent captioning."""
    taxonomy_path = CONFIG_DIR / "taxonomy.json"
    if taxonomy_path.exists():
        with open(taxonomy_path) as f:
            return json.load(f)
    return None


def build_caption_prompt(taxonomy=None):
    """Build the captioning prompt with optional taxonomy guidance."""

    base_prompt = """Describe this image in detail. Include:
- Subject(s): appearance, pose, expression, body type
- Clothing: specific garment types, colors, styles, coverage
- Setting: environment, lighting, atmosphere
- Composition: framing, angle, focus

Be precise and use specific terminology. Describe what you see objectively."""

    if taxonomy:
        vocab_section = "\n\nUse these specific terms when applicable:\n"
        for category, terms in taxonomy.items():
            vocab_section += f"- {category}: {', '.join(terms)}\n"
        base_prompt += vocab_section

    return base_prompt


def load_model(model_id: str, device: str = "cuda"):
    """Load the Qwen3-VL model and processor."""
    print(f"Loading model: {model_id}")

    # Determine dtype and loading strategy based on VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Available VRAM: {vram_gb:.1f} GB")

        if vram_gb >= 20:
            # Full bf16 for 24GB+ cards
            dtype = torch.bfloat16
            load_in_4bit = False
        else:
            # 4-bit quantization for smaller cards
            dtype = torch.bfloat16
            load_in_4bit = True
    else:
        dtype = torch.float32
        load_in_4bit = False
        device = "cpu"

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        **model_kwargs
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    return model, processor


def caption_image(model, processor, image_path: Path, prompt: str) -> str:
    """Generate caption for a single image."""

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
            max_new_tokens=512,
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


def caption_images_batch(model, processor, image_paths: list, prompt: str, batch_size: int = 4) -> list:
    """Generate captions for a batch of images - 3-5x faster than single image processing."""

    captions = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        # Load images
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(image)
            except Exception:
                batch_images.append(None)

        # Create batch messages
        batch_messages = []
        batch_texts = []
        all_image_inputs = []
        all_video_inputs = []

        for image in batch_images:
            if image is None:
                continue

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            batch_messages.append(messages)
            batch_texts.append(text)
            all_image_inputs.extend(image_inputs)
            all_video_inputs.extend(video_inputs)

        # Process batch
        if batch_texts:
            inputs = processor(
                text=batch_texts,
                images=all_image_inputs,
                videos=all_video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            batch_captions = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            captions.extend([cap.strip() for cap in batch_captions])

    return captions


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    model,
    processor,
    prompt: str,
    resume: bool = True,
    batch_size: int = 4,
):
    """Process all images in a directory with batch processing for 3-5x speedup."""

    output_dir.mkdir(parents=True, exist_ok=True)
    captions_file = output_dir / "captions.json"

    # Load existing captions if resuming
    if resume and captions_file.exists():
        with open(captions_file) as f:
            captions = json.load(f)
        print(f"Resuming with {len(captions)} existing captions")
    else:
        captions = {}

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []

    for ext in image_extensions:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))

    images = sorted(set(images))
    print(f"Found {len(images)} images")

    # Filter already processed
    to_process = [
        img for img in images
        if img.name not in captions
    ]
    print(f"Processing {len(to_process)} new images with batch_size={batch_size}")

    # Process in batches with progress bar
    for batch_start in tqdm(range(0, len(to_process), batch_size), desc="Captioning batches"):
        batch_paths = to_process[batch_start:batch_start + batch_size]

        try:
            # Use batch processing
            batch_captions = caption_images_batch(model, processor, batch_paths, prompt, batch_size=len(batch_paths))

            # Store results
            for img_path, caption in zip(batch_paths, batch_captions):
                captions[img_path.name] = {
                    "caption": caption,
                    "path": str(img_path.relative_to(input_dir)),
                }

            # Save periodically (every ~50 images)
            if len(captions) % 50 < batch_size:
                with open(captions_file, "w") as f:
                    json.dump(captions, f, indent=2)

        except Exception as e:
            print(f"\nError processing batch starting at {batch_start}: {e}")
            # Fallback to single-image processing for this batch
            for img_path in batch_paths:
                try:
                    caption = caption_image(model, processor, img_path, prompt)
                    captions[img_path.name] = {
                        "caption": caption,
                        "path": str(img_path.relative_to(input_dir)),
                    }
                except Exception as e2:
                    print(f"\nError processing {img_path.name}: {e2}")
                    continue

    # Final save
    with open(captions_file, "w") as f:
        json.dump(captions, f, indent=2)

    print(f"\nSaved {len(captions)} captions to {captions_file}")
    return captions


def main():
    parser = argparse.ArgumentParser(description="Caption NSFW images")
    parser.add_argument(
        "--dataset",
        choices=["civitai", "nsfw_t2i", "all", "custom"],
        default="all",
        help="Which dataset to process"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Custom input directory (use with --dataset custom)"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="custom",
        help="Name for output folder when using custom input"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, don't resume from existing captions"
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help="Model ID to use for captioning"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for GPU inference (default: 4, increase for faster processing)"
    )
    args = parser.parse_args()

    # Load taxonomy
    taxonomy = load_taxonomy()
    if taxonomy:
        print(f"Loaded taxonomy with {len(taxonomy)} categories")

    # Build prompt
    prompt = build_caption_prompt(taxonomy)

    # Load model
    model, processor = load_model(args.model)

    # Process datasets
    datasets_to_process = []
    if args.dataset == "custom":
        if not args.input:
            print("--input required when using --dataset custom")
            return
        datasets_to_process.append((args.output_name, Path(args.input)))
    if args.dataset in ["civitai", "all"]:
        datasets_to_process.append(("civitai", DATA_DIR / "raw" / "civitai"))
    if args.dataset in ["nsfw_t2i", "all"]:
        datasets_to_process.append(("nsfw_t2i", DATA_DIR / "raw" / "nsfw_t2i"))

    for name, input_dir in datasets_to_process:
        if not input_dir.exists():
            print(f"Dataset {name} not found at {input_dir}, skipping...")
            continue

        print(f"\n{'='*50}")
        print(f"Processing: {name}")
        print(f"{'='*50}")

        output_dir = OUTPUT_DIR / name
        process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            processor=processor,
            prompt=prompt,
            resume=not args.no_resume,
            batch_size=args.batch_size,
        )

    print("\nCaptioning complete!")


if __name__ == "__main__":
    main()
