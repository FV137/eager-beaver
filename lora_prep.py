#!/usr/bin/env python3
"""
LoRA Prep - Intelligent LoRA dataset preparation with shot classification.

Analyzes images, classifies shots (close/mid/far), extracts concepts from captions,
and creates well-organized datasets for LoRA training.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import re

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
from PIL import Image

console = Console()

# Project paths
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "lora_datasets"


# ============================================================================
# Shot Classification
# ============================================================================

def classify_shot_type(face_bbox: List[float], image_size: Tuple[int, int]) -> Dict:
    """
    Classify shot type based on face size relative to image.

    Returns:
        Dictionary with shot_type, face_ratio, and confidence
    """
    x1, y1, x2, y2 = face_bbox
    face_width = x2 - x1
    face_height = y2 - y1
    face_area = face_width * face_height

    img_width, img_height = image_size
    img_area = img_width * img_height

    face_ratio = face_area / img_area

    # Classification thresholds
    # close-up: face takes up >25% of image (headshot/portrait)
    # mid-shot: face is 8-25% of image (waist-up, upper body)
    # far-shot: face is <8% of image (full body, environmental)

    if face_ratio > 0.25:
        shot_type = "close"
        confidence = min(0.95, 0.7 + (face_ratio - 0.25) * 0.5)
    elif face_ratio > 0.08:
        shot_type = "mid"
        # Highest confidence in middle of range
        mid_point = 0.165  # midpoint of 0.08-0.25
        distance_from_mid = abs(face_ratio - mid_point)
        confidence = max(0.6, 0.9 - distance_from_mid * 3)
    else:
        shot_type = "far"
        confidence = min(0.85, 0.7 - face_ratio * 5)

    return {
        "shot_type": shot_type,
        "face_ratio": float(face_ratio),
        "confidence": float(confidence),
        "face_bbox": face_bbox,
    }


# ============================================================================
# Concept Extraction
# ============================================================================

def extract_concepts_from_caption(caption: str, taxonomy: Optional[Dict] = None) -> Dict:
    """
    Extract structured concepts from image caption.

    Args:
        caption: Text caption describing the image
        taxonomy: Optional taxonomy dictionary for guided extraction

    Returns:
        Dictionary with extracted concepts
    """
    caption_lower = caption.lower()

    concepts = {
        "raw_caption": caption,
        "concepts": [],
        "pose": None,
        "setting": None,
        "lighting": None,
        "clothing": [],
        "attributes": [],
    }

    # Common pose keywords
    pose_keywords = {
        "standing": ["standing", "upright"],
        "sitting": ["sitting", "seated"],
        "lying": ["lying", "laying", "reclined"],
        "kneeling": ["kneeling"],
        "crouching": ["crouching", "squatting"],
        "leaning": ["leaning"],
    }

    # Setting keywords
    setting_keywords = {
        "indoor": ["indoor", "inside", "room", "bedroom", "living room", "kitchen"],
        "outdoor": ["outdoor", "outside", "beach", "park", "garden", "street"],
        "studio": ["studio", "plain background", "solid background"],
    }

    # Lighting keywords
    lighting_keywords = {
        "natural": ["natural light", "daylight", "sunlight", "golden hour"],
        "artificial": ["artificial", "studio lighting", "flash"],
        "soft": ["soft lighting", "diffused"],
        "dramatic": ["dramatic lighting", "low key", "high contrast"],
    }

    # Extract pose
    for pose, keywords in pose_keywords.items():
        if any(kw in caption_lower for kw in keywords):
            concepts["pose"] = pose
            concepts["concepts"].append(pose)
            break

    # Extract setting
    for setting, keywords in setting_keywords.items():
        if any(kw in caption_lower for kw in keywords):
            concepts["setting"] = setting
            concepts["concepts"].append(setting)
            break

    # Extract lighting
    for lighting, keywords in lighting_keywords.items():
        if any(kw in caption_lower for kw in keywords):
            concepts["lighting"] = lighting
            concepts["concepts"].append(lighting)
            break

    # Extract from taxonomy if provided
    if taxonomy:
        for category, terms in taxonomy.items():
            if category.startswith("_"):  # Skip metadata
                continue

            for term in terms:
                if term.lower() in caption_lower:
                    concepts["concepts"].append(term)

                    # Categorize
                    if "clothing" in category or "swimwear" in category or "lingerie" in category:
                        concepts["clothing"].append(term)
                    elif category in ["body_descriptors", "expressions"]:
                        concepts["attributes"].append(term)

    # Additional common attributes
    attribute_keywords = [
        "smiling", "happy", "serious", "looking at camera",
        "blonde", "brunette", "long hair", "short hair",
        "blue eyes", "green eyes", "brown eyes",
    ]

    for attr in attribute_keywords:
        if attr in caption_lower and attr not in concepts["concepts"]:
            concepts["concepts"].append(attr)
            concepts["attributes"].append(attr)

    # Remove duplicates
    concepts["concepts"] = list(set(concepts["concepts"]))
    concepts["clothing"] = list(set(concepts["clothing"]))
    concepts["attributes"] = list(set(concepts["attributes"]))

    return concepts


# ============================================================================
# Dataset Preparation
# ============================================================================

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory (default: auto-generated)')
@click.option('--name', '-n', required=True, help='Person/concept name (used for trigger word)')
@click.option('--trigger', '-t', help='Custom trigger word (default: auto from name)')
@click.option('--facevault-cache', type=click.Path(), help='Path to FaceVault face_cache.json for shot classification')
@click.option('--captions', type=click.Path(), help='Path to captions.json from caption_images.py')
@click.option('--taxonomy', type=click.Path(), help='Path to taxonomy.json for concept extraction')
@click.option('--organize-by-shot', is_flag=True, default=True, help='Organize images into close/mid/far folders')
@click.option('--min-per-shot', type=int, default=3, help='Minimum images per shot type (warning if below)')
@click.option('--symlink', is_flag=True, help='Create symlinks instead of copying files')
def prepare(
    input_dir: str,
    output: Optional[str],
    name: str,
    trigger: Optional[str],
    facevault_cache: Optional[str],
    captions: Optional[str],
    taxonomy: Optional[str],
    organize_by_shot: bool,
    min_per_shot: int,
    symlink: bool,
):
    """
    Prepare LoRA training dataset with shot classification and concept extraction.

    Examples:

        # From FaceVault export
        lora_prep outputs/facevault/lora_ready/Emma --name Emma

        # With captions
        lora_prep /path/to/images --name Emma \\
          --captions outputs/processed/emma/captions.json

        # Full pipeline integration
        lora_prep outputs/facevault/organized/Emma --name Emma \\
          --facevault-cache outputs/facevault/face_cache.json \\
          --captions outputs/processed/emma/captions.json \\
          --taxonomy configs/taxonomy.json
    """

    input_path = Path(input_dir)

    # Auto-generate output path if not provided
    if not output:
        trigger_word = trigger or name.lower().replace(" ", "_")
        output_path = OUTPUT_DIR / trigger_word
    else:
        output_path = Path(output)

    output_path.mkdir(parents=True, exist_ok=True)

    # Generate trigger word
    trigger_word = trigger or name.lower().replace(" ", "_")

    # Header
    console.print(Panel.fit(
        f"[bold cyan]LoRA Dataset Preparation[/bold cyan]\n"
        f"Person: {name}\n"
        f"Trigger: {trigger_word}\n"
        f"Input: {input_path}\n"
        f"Output: {output_path}",
        border_style="cyan"
    ))
    console.print()

    # Load optional data
    face_cache = None
    if facevault_cache:
        with open(facevault_cache) as f:
            face_cache = json.load(f)
        console.print(f"[green]✓[/green] Loaded FaceVault cache")

    caption_data = None
    if captions:
        with open(captions) as f:
            caption_data = json.load(f)
        console.print(f"[green]✓[/green] Loaded captions for {len(caption_data)} images")

    taxonomy_data = None
    if taxonomy:
        with open(taxonomy) as f:
            taxonomy_data = json.load(f)
        console.print(f"[green]✓[/green] Loaded taxonomy")

    console.print()

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    for ext in image_extensions:
        images.extend(input_path.rglob(f"*{ext}"))
        images.extend(input_path.rglob(f"*{ext.upper()}"))
    images = sorted(set(images))

    console.print(f"[cyan]Found {len(images)} images[/cyan]\n")

    # Process images
    shot_data = {}
    concept_data = {}
    shot_distribution = defaultdict(int)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Analyzing images...", total=len(images))

        for img_path in images:
            img_name = img_path.name

            # Classify shot type
            shot_type = "unknown"
            shot_info = None

            if face_cache:
                # Find face data for this image
                img_key = str(img_path)
                if img_key not in face_cache:
                    # Try relative paths
                    img_key = str(img_path.relative_to(PROJECT_DIR))

                if img_key in face_cache and face_cache[img_key].get("faces"):
                    faces = face_cache[img_key]["faces"]
                    if faces:
                        # Use first/largest face
                        face = max(faces, key=lambda f: f.get("size", [0, 0])[0] * f.get("size", [0, 0])[1])

                        # Get image dimensions
                        with Image.open(img_path) as img:
                            img_size = img.size

                        shot_info = classify_shot_type(face["bbox"], img_size)
                        shot_type = shot_info["shot_type"]

            # If no face cache, try to infer from image
            if shot_type == "unknown":
                # Default to mid-shot with low confidence
                shot_info = {
                    "shot_type": "mid",
                    "face_ratio": 0.0,
                    "confidence": 0.3,
                    "face_bbox": None,
                }
                shot_type = "mid"

            shot_data[img_name] = shot_info
            shot_distribution[shot_type] += 1

            # Extract concepts from caption if available
            if caption_data and img_name in caption_data:
                caption = caption_data[img_name].get("caption", "")
                concepts = extract_concepts_from_caption(caption, taxonomy_data)
                concept_data[img_name] = concepts

            progress.update(task, advance=1)

    console.print()

    # Show shot distribution
    dist_table = Table(title="Shot Distribution", box=box.ROUNDED, border_style="cyan")
    dist_table.add_column("Shot Type", style="cyan")
    dist_table.add_column("Count", justify="right", style="yellow")
    dist_table.add_column("Percentage", justify="right", style="green")

    total_images = len(images)
    for shot_type in ["close", "mid", "far", "unknown"]:
        count = shot_distribution[shot_type]
        pct = (count / total_images * 100) if total_images > 0 else 0

        style = ""
        if count < min_per_shot and shot_type != "unknown":
            style = "red"

        dist_table.add_row(
            shot_type.capitalize(),
            str(count),
            f"{pct:.1f}%",
            style=style
        )

    console.print(dist_table)
    console.print()

    # Warnings
    for shot_type in ["close", "mid", "far"]:
        if shot_distribution[shot_type] < min_per_shot:
            console.print(
                f"[yellow]⚠️  Warning: Only {shot_distribution[shot_type]} {shot_type} shots "
                f"(recommended: {min_per_shot}+)[/yellow]"
            )

    if shot_distribution["unknown"] > total_images * 0.3:
        console.print(
            f"[yellow]⚠️  Warning: {shot_distribution['unknown']} images couldn't be classified. "
            f"Provide --facevault-cache for better classification.[/yellow]"
        )

    console.print()

    # Organize files
    console.print("[cyan]Organizing dataset...[/cyan]")

    if organize_by_shot:
        # Create shot type directories
        for shot_type in ["close", "mid", "far", "unknown"]:
            (output_path / shot_type).mkdir(exist_ok=True)
            (output_path / "captions").mkdir(exist_ok=True)

    # Copy/link files
    file_index = defaultdict(int)

    for img_path in images:
        img_name = img_path.name
        shot_type = shot_data[img_name]["shot_type"]

        # Generate new filename
        file_index[shot_type] += 1
        new_name = f"{trigger_word}_{shot_type}_{file_index[shot_type]:04d}{img_path.suffix}"

        if organize_by_shot:
            dst = output_path / shot_type / new_name
        else:
            dst = output_path / new_name

        # Copy or symlink
        if symlink:
            dst.symlink_to(img_path.absolute())
        else:
            shutil.copy2(img_path, dst)

        # Create caption file if we have caption data
        if img_name in concept_data:
            caption = concept_data[img_name]["raw_caption"]
            caption_file = output_path / "captions" / f"{new_name.rsplit('.', 1)[0]}.txt"
            with open(caption_file, "w") as f:
                f.write(caption)

    # Save metadata
    metadata = {
        "person": name,
        "trigger_word": trigger_word,
        "total_images": len(images),
        "shot_distribution": dict(shot_distribution),
        "organized_by_shot": organize_by_shot,
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }

    # Extract common concepts
    if concept_data:
        all_concepts = []
        for concepts in concept_data.values():
            all_concepts.extend(concepts["concepts"])

        concept_counts = defaultdict(int)
        for concept in all_concepts:
            concept_counts[concept] += 1

        # Top 20 concepts
        top_concepts = sorted(concept_counts.items(), key=lambda x: -x[1])[:20]
        metadata["top_concepts"] = [
            {"concept": c, "count": cnt, "frequency": cnt/len(images)}
            for c, cnt in top_concepts
        ]

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save shot data
    with open(output_path / "shots.json", "w") as f:
        json.dump(shot_data, f, indent=2)

    # Save concept data
    if concept_data:
        with open(output_path / "concepts.json", "w") as f:
            json.dump(concept_data, f, indent=2)

    # Create preview manifest for future UI
    preview_manifest = {
        "dataset": {
            "name": name,
            "trigger": trigger_word,
            "total_images": len(images),
        },
        "images": []
    }

    for img_path in images:
        img_name = img_path.name
        shot_info = shot_data[img_name]

        entry = {
            "filename": img_name,
            "shot_type": shot_info["shot_type"],
            "shot_confidence": shot_info["confidence"],
        }

        if img_name in concept_data:
            entry["caption"] = concept_data[img_name]["raw_caption"]
            entry["concepts"] = concept_data[img_name]["concepts"]
            entry["pose"] = concept_data[img_name]["pose"]
            entry["setting"] = concept_data[img_name]["setting"]

        preview_manifest["images"].append(entry)

    with open(output_path / "preview_manifest.json", "w") as f:
        json.dump(preview_manifest, f, indent=2)

    # Summary
    console.print()
    summary = Table(show_header=False, box=box.ROUNDED, border_style="green")
    summary.add_row("📦 Dataset", name)
    summary.add_row("🎯 Trigger word", f"[bold]{trigger_word}[/bold]")
    summary.add_row("📸 Total images", str(len(images)))
    summary.add_row("📊 Close shots", str(shot_distribution["close"]))
    summary.add_row("📊 Mid shots", str(shot_distribution["mid"]))
    summary.add_row("📊 Far shots", str(shot_distribution["far"]))
    summary.add_row("💾 Saved to", str(output_path))

    console.print(Panel(summary, title="[bold green]LoRA Dataset Ready!", border_style="green"))

    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print(f"  1. Review images in: {output_path}")
    console.print(f"  2. Check metadata: {output_path / 'metadata.json'}")
    console.print(f"  3. Train LoRA with trigger word: [bold]{trigger_word}[/bold]")
    if concept_data:
        console.print(f"  4. Top concepts available in metadata for fine-tuning")


@click.group()
@click.version_option(version="1.0.0", prog_name="LoRA Prep")
def cli():
    """
    🎨 LoRA Prep - Intelligent LoRA dataset preparation.

    Analyzes images, classifies shots (close/mid/far), extracts concepts,
    and creates well-organized datasets for LoRA training.
    """
    pass


cli.add_command(prepare)


if __name__ == "__main__":
    cli()
