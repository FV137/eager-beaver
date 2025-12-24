#!/usr/bin/env python3
"""
Download NSFW datasets for vision training pipeline.
Datasets:
  - wallstoneai/civitai-top-nsfw-images-with-metadata (~6K images)
  - zxbsmk/NSFW-T2I (~38K images)
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm
from PIL import Image
import io

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def download_civitai_dataset():
    """Download the CivitAI NSFW dataset with metadata."""
    print("\n=== Downloading CivitAI Dataset ===")

    output_dir = DATA_DIR / "civitai"
    output_dir.mkdir(parents=True, exist_ok=True)

    # This dataset has images/ folder and prompts.json
    snapshot_download(
        repo_id="wallstoneai/civitai-top-nsfw-images-with-metadata",
        repo_type="dataset",
        local_dir=output_dir,
        ignore_patterns=["*.md"],
    )

    # Count images
    images_dir = output_dir / "images"
    if images_dir.exists():
        count = len(list(images_dir.glob("*")))
        print(f"Downloaded {count} images to {output_dir}")

    return output_dir


def download_nsfw_t2i_dataset():
    """Download the NSFW-T2I dataset."""
    print("\n=== Downloading NSFW-T2I Dataset ===")

    output_dir = DATA_DIR / "nsfw_t2i"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load streaming to handle large dataset
    try:
        ds = load_dataset(
            "zxbsmk/NSFW-T2I",
            split="train",
            streaming=True
        )

        # Save images and captions
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        captions = {}

        print("Downloading images (this may take a while)...")
        for i, example in enumerate(tqdm(ds, desc="Processing")):
            try:
                # Get image
                if "jpg" in example and example["jpg"] is not None:
                    img = example["jpg"]
                    if isinstance(img, bytes):
                        img = Image.open(io.BytesIO(img))

                    img_path = images_dir / f"{i:06d}.jpg"
                    img.save(img_path, "JPEG", quality=95)

                    # Get caption if available
                    caption = example.get("txt", "")
                    if caption:
                        captions[f"{i:06d}.jpg"] = caption

            except Exception as e:
                print(f"Error processing image {i}: {e}")
                continue

            # Optional: limit for testing
            # if i >= 1000:
            #     break

        # Save captions
        captions_path = output_dir / "original_captions.json"
        with open(captions_path, "w") as f:
            json.dump(captions, f, indent=2)

        print(f"Downloaded {len(captions)} images to {output_dir}")

    except Exception as e:
        print(f"Error with streaming, trying direct download: {e}")
        # Fallback to snapshot
        snapshot_download(
            repo_id="zxbsmk/NSFW-T2I",
            repo_type="dataset",
            local_dir=output_dir,
        )

    return output_dir


def main():
    print("NSFW Dataset Downloader")
    print("=" * 50)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download both datasets
    civitai_dir = download_civitai_dataset()
    nsfw_t2i_dir = download_nsfw_t2i_dataset()

    print("\n" + "=" * 50)
    print("Download complete!")
    print(f"CivitAI: {civitai_dir}")
    print(f"NSFW-T2I: {nsfw_t2i_dir}")


if __name__ == "__main__":
    main()
