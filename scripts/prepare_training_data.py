#!/usr/bin/env python3
"""
Prepare captioned data for Ministral 3 vision training.
Converts captions to the format expected by HuggingFace training.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from PIL import Image
import random

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "training_ready"


def load_captions(dataset_name: str) -> dict:
    """Load captions from processed dataset."""
    captions_file = PROCESSED_DIR / dataset_name / "captions.json"
    if not captions_file.exists():
        print(f"No captions found for {dataset_name}")
        return {}

    with open(captions_file) as f:
        return json.load(f)


def create_conversation_format(image_path: str, caption: str) -> dict:
    """
    Create training example in conversation format for Ministral 3.
    Uses the open system prompt capability.
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful vision assistant that provides detailed, accurate descriptions of images. Be specific and objective in your descriptions."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            },
            {
                "role": "assistant",
                "content": caption
            }
        ]
    }


def create_simple_format(image_path: str, caption: str) -> dict:
    """Simple image-caption pair format."""
    return {
        "image": image_path,
        "caption": caption
    }


def filter_captions(
    captions: dict,
    min_length: int = 50,
    max_length: int = 2000,
) -> dict:
    """Filter captions by quality criteria."""
    filtered = {}

    for img_name, data in captions.items():
        caption = data.get("caption", "") if isinstance(data, dict) else data

        # Length filter
        if len(caption) < min_length:
            continue
        if len(caption) > max_length:
            continue

        # Could add more quality filters here:
        # - Keyword presence
        # - Vocabulary usage
        # - Sentence structure

        filtered[img_name] = data

    return filtered


def prepare_dataset(
    dataset_name: str,
    output_format: str = "conversation",
    train_split: float = 0.9,
    copy_images: bool = False,
    min_caption_length: int = 50,
):
    """Prepare a single dataset for training."""
    print(f"\n{'='*50}")
    print(f"Preparing: {dataset_name}")
    print(f"{'='*50}")

    # Load captions
    captions = load_captions(dataset_name)
    if not captions:
        return

    print(f"Loaded {len(captions)} captions")

    # Filter
    captions = filter_captions(captions, min_length=min_caption_length)
    print(f"After filtering: {len(captions)} captions")

    # Find source images
    raw_dir = DATA_DIR / "raw" / dataset_name

    # Create output structure
    output_base = OUTPUT_DIR / dataset_name
    output_base.mkdir(parents=True, exist_ok=True)

    if copy_images:
        images_out = output_base / "images"
        images_out.mkdir(exist_ok=True)

    # Process examples
    examples = []

    for img_name, data in tqdm(captions.items(), desc="Processing"):
        caption = data.get("caption", "") if isinstance(data, dict) else data
        rel_path = data.get("path", img_name) if isinstance(data, dict) else img_name

        # Find source image
        src_image = raw_dir / rel_path
        if not src_image.exists():
            # Try common subdirectories
            for subdir in ["images", "."]:
                alt_path = raw_dir / subdir / img_name
                if alt_path.exists():
                    src_image = alt_path
                    break

        if not src_image.exists():
            continue

        # Determine output image path
        if copy_images:
            dst_image = images_out / img_name
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)
            image_ref = f"images/{img_name}"
        else:
            image_ref = str(src_image.absolute())

        # Create training example
        if output_format == "conversation":
            example = create_conversation_format(image_ref, caption)
        else:
            example = create_simple_format(image_ref, caption)

        examples.append(example)

    print(f"Created {len(examples)} training examples")

    # Shuffle and split
    random.shuffle(examples)
    split_idx = int(len(examples) * train_split)

    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    print(f"Train: {len(train_examples)}, Validation: {len(val_examples)}")

    # Save
    with open(output_base / "train.json", "w") as f:
        json.dump(train_examples, f, indent=2)

    with open(output_base / "validation.json", "w") as f:
        json.dump(val_examples, f, indent=2)

    # Also save as JSONL for compatibility
    with open(output_base / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(output_base / "validation.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved to {output_base}")

    return output_base


def merge_datasets(dataset_names: list[str], output_name: str = "combined"):
    """Merge multiple prepared datasets into one."""
    print(f"\n{'='*50}")
    print(f"Merging datasets: {dataset_names}")
    print(f"{'='*50}")

    all_train = []
    all_val = []

    for name in dataset_names:
        train_file = OUTPUT_DIR / name / "train.json"
        val_file = OUTPUT_DIR / name / "validation.json"

        if train_file.exists():
            with open(train_file) as f:
                all_train.extend(json.load(f))

        if val_file.exists():
            with open(val_file) as f:
                all_val.extend(json.load(f))

    # Shuffle combined
    random.shuffle(all_train)
    random.shuffle(all_val)

    # Save merged
    output_base = OUTPUT_DIR / output_name
    output_base.mkdir(parents=True, exist_ok=True)

    with open(output_base / "train.json", "w") as f:
        json.dump(all_train, f, indent=2)

    with open(output_base / "validation.json", "w") as f:
        json.dump(all_val, f, indent=2)

    with open(output_base / "train.jsonl", "w") as f:
        for ex in all_train:
            f.write(json.dumps(ex) + "\n")

    with open(output_base / "validation.jsonl", "w") as f:
        for ex in all_val:
            f.write(json.dumps(ex) + "\n")

    print(f"Merged: {len(all_train)} train, {len(all_val)} validation")
    print(f"Saved to {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument(
        "--dataset",
        choices=["civitai", "nsfw_t2i", "all"],
        default="all",
        help="Which dataset to prepare"
    )
    parser.add_argument(
        "--format",
        choices=["conversation", "simple"],
        default="conversation",
        help="Output format"
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images to output dir (for portability)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Also create merged dataset"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Minimum caption length"
    )
    args = parser.parse_args()

    datasets = []
    if args.dataset in ["civitai", "all"]:
        datasets.append("civitai")
    if args.dataset in ["nsfw_t2i", "all"]:
        datasets.append("nsfw_t2i")

    for name in datasets:
        prepare_dataset(
            name,
            output_format=args.format,
            copy_images=args.copy_images,
            min_caption_length=args.min_length,
        )

    if args.merge and len(datasets) > 1:
        merge_datasets(datasets)

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
