#!/usr/bin/env python3
"""
Upload prepared dataset to HuggingFace Hub for training.
"""

import os
import json
import argparse
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import HfApi, create_repo
from PIL import Image
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
TRAINING_DIR = DATA_DIR / "training_ready"


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def create_hf_dataset(
    dataset_name: str,
    include_images: bool = True,
) -> DatasetDict:
    """Create HuggingFace Dataset from prepared data."""

    dataset_dir = TRAINING_DIR / dataset_name
    train_file = dataset_dir / "train.jsonl"
    val_file = dataset_dir / "validation.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"No training data at {train_file}")

    train_examples = load_jsonl(train_file)
    val_examples = load_jsonl(val_file) if val_file.exists() else []

    print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")

    def process_examples(examples: list[dict]) -> dict:
        """Convert to columnar format for HF Dataset."""
        processed = {
            "image": [],
            "conversations": [],
        }

        for ex in tqdm(examples, desc="Processing"):
            messages = ex.get("messages", [])

            # Extract image path from user message
            image_path = None
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            image_path = item.get("image")
                            break

            if image_path and include_images:
                try:
                    img = Image.open(image_path)
                    processed["image"].append(img)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
            else:
                processed["image"].append(None)

            # Store conversation
            processed["conversations"].append(json.dumps(messages))

        return processed

    train_data = process_examples(train_examples)
    val_data = process_examples(val_examples) if val_examples else None

    # Create datasets
    train_ds = Dataset.from_dict(train_data)

    if val_data:
        val_ds = Dataset.from_dict(val_data)
        dataset = DatasetDict({"train": train_ds, "validation": val_ds})
    else:
        dataset = DatasetDict({"train": train_ds})

    return dataset


def upload_to_hub(
    dataset: DatasetDict,
    repo_id: str,
    private: bool = True,
):
    """Upload dataset to HuggingFace Hub."""

    api = HfApi()

    # Create repo if needed
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Push dataset
    print(f"Uploading to {repo_id}...")
    dataset.push_to_hub(
        repo_id,
        private=private,
    )

    print(f"Dataset uploaded: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace")
    parser.add_argument(
        "--dataset",
        default="combined",
        help="Which prepared dataset to upload"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g., username/dataset-name)"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make dataset public (default: private)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Don't include images (just references)"
    )
    args = parser.parse_args()

    # Create HF dataset
    dataset = create_hf_dataset(
        args.dataset,
        include_images=not args.no_images,
    )

    # Upload
    upload_to_hub(
        dataset,
        args.repo_id,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
