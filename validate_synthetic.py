#!/usr/bin/env python3
"""
Synthetic Validation - Quality check generated images using YOLO models.

Validates synthetic images by:
- Running same YOLO detectors as gap analysis
- Checking if target attributes were achieved
- Scoring quality (blur, brightness, face confidence)
- Auto-filtering failures, queuing successes for review

Ensures only high-quality synthetic images enter the training loop.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Confirm

# Import from gap_analysis
import sys
sys.path.insert(0, str(Path(__file__).parent))

from gap_analysis import (
    load_yolo_model,
    analyze_face_angles,
    analyze_eyes,
    analyze_hands,
    compute_blur_score
)

console = Console()


# ============================================================================
# Validation Thresholds
# ============================================================================

VALIDATION_THRESHOLDS = {
    "face_confidence_min": 0.5,
    "blur_score_min": 100.0,  # Laplacian variance
    "brightness_min": 30.0,
    "brightness_max": 225.0,
}


# ============================================================================
# Image Quality Scoring
# ============================================================================

def score_image_quality(image_path: str) -> Dict:
    """
    Score image technical quality.

    Returns dict with:
        - blur_score: Laplacian variance
        - brightness: Mean pixel value
        - quality_score: Combined 0-1 score
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return {"blur_score": 0.0, "brightness": 0.0, "quality_score": 0.0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur score (Laplacian variance)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness
    brightness = np.mean(gray)

    # Normalize and combine scores
    blur_normalized = min(blur_score / 500.0, 1.0)  # 500+ is sharp
    brightness_normalized = 1.0 - abs(brightness - 127.5) / 127.5  # Closer to middle is better

    quality_score = (blur_normalized * 0.6 + brightness_normalized * 0.4)

    return {
        "blur_score": float(blur_score),
        "brightness": float(brightness),
        "quality_score": float(quality_score)
    }


# ============================================================================
# Target Validation
# ============================================================================

def validate_target_achieved(
    image_path: str,
    metadata: Dict,
    models: Dict
) -> Tuple[bool, str]:
    """
    Check if image achieved its generation target.

    Args:
        image_path: Path to synthetic image
        metadata: Generation metadata (gap_type, gap_target, etc.)
        models: Loaded YOLO models

    Returns:
        (success: bool, reason: str)
    """

    gap_type = metadata.get("gap_type", "")
    gap_target = metadata.get("gap_target", "")

    # Face angle validation
    if gap_type == "face_angle":
        if not models.get("face"):
            return True, "No face model - skipping validation"

        face_result = analyze_face_angles(str(image_path), models["face"])
        detected_angle = face_result.get("angle", "unknown")

        # Fuzzy matching (profile matches both profile_left/right)
        if gap_target in detected_angle or detected_angle in gap_target:
            return True, f"Achieved target angle: {detected_angle}"
        else:
            return False, f"Wrong angle: {detected_angle} (wanted {gap_target})"

    # Hand visibility validation
    elif gap_type == "composition" and "hands" in gap_target:
        if not models.get("hands"):
            return True, "No hands model - skipping validation"

        hands_result = analyze_hands(str(image_path), models["hands"])
        if hands_result.get("visible", False):
            return True, f"Hands detected: {hands_result['hands_detected']}"
        else:
            return False, "No hands detected"

    # Eyes visibility validation
    elif gap_type == "composition" and "eyes" in gap_target:
        if not models.get("eyes"):
            return True, "No eyes model - skipping validation"

        eyes_result = analyze_eyes(str(image_path), models["eyes"])
        if eyes_result.get("visible", False):
            return True, f"Eyes detected: {eyes_result['eyes_detected']}"
        else:
            return False, "No eyes detected"

    # Default: pass validation
    return True, "No specific validation required"


# ============================================================================
# Batch Validation
# ============================================================================

def validate_synthetic_batch(
    synthetic_dir: str,
    models_dir: str = "models",
    output_approved: str = None,
    output_rejected: str = None,
    auto_filter: bool = True
) -> Dict:
    """
    Validate batch of synthetic images.

    Args:
        synthetic_dir: Directory with generated images
        models_dir: Directory with YOLO models
        output_approved: Directory for approved images (optional)
        output_rejected: Directory for rejected images (optional)
        auto_filter: Automatically move images based on validation

    Returns:
        Validation report dict
    """

    synthetic_path = Path(synthetic_dir)
    metadata_file = synthetic_path / "generation_metadata.json"

    if not metadata_file.exists():
        console.print("[red]Error: generation_metadata.json not found[/red]")
        console.print(f"[dim]Expected at: {metadata_file}[/dim]")
        return {}

    # Load metadata
    with open(metadata_file) as f:
        metadata_list = json.load(f)

    console.print(Panel.fit(
        "[bold cyan]Synthetic Validation[/bold cyan]\n"
        f"Images: {len(metadata_list)}\n"
        f"Auto-filter: {'ON' if auto_filter else 'OFF'}",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Load YOLO models
    console.print("[cyan]Loading YOLO models...[/cyan]")
    models = {}
    model_files = {
        "face": "yolov5l-face.pt",
        "eyes": "eyes_yolov.pt",
        "hands": "hand_yolov8n.pt",
    }

    models_path = Path(models_dir)
    for name, filename in model_files.items():
        model_path = models_path / filename
        if model_path.exists():
            try:
                models[name] = load_yolo_model(str(model_path))
                console.print(f"  [green]✓[/green] Loaded {name}")
            except Exception:
                models[name] = None
        else:
            console.print(f"  [dim]○ {name} not found (optional)[/dim]")
            models[name] = None

    console.print()

    # Validate each image
    results = {
        "total": len(metadata_list),
        "approved": [],
        "rejected": [],
        "quality_scores": []
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Validating images...", total=len(metadata_list))

        for metadata in metadata_list:
            filename = metadata.get("file", "")
            image_path = synthetic_path / filename

            if not image_path.exists():
                console.print(f"[yellow]Warning: {filename} not found[/yellow]")
                progress.update(task, advance=1)
                continue

            # Quality scoring
            quality = score_image_quality(str(image_path))

            # Target validation
            target_achieved, reason = validate_target_achieved(
                str(image_path),
                metadata,
                models
            )

            # Pass/fail decision
            passed = True
            fail_reasons = []

            # Check quality thresholds
            if quality["blur_score"] < VALIDATION_THRESHOLDS["blur_score_min"]:
                passed = False
                fail_reasons.append(f"Too blurry ({quality['blur_score']:.1f})")

            if quality["brightness"] < VALIDATION_THRESHOLDS["brightness_min"]:
                passed = False
                fail_reasons.append(f"Too dark ({quality['brightness']:.1f})")

            if quality["brightness"] > VALIDATION_THRESHOLDS["brightness_max"]:
                passed = False
                fail_reasons.append(f"Too bright ({quality['brightness']:.1f})")

            # Check target achievement
            if not target_achieved:
                passed = False
                fail_reasons.append(reason)

            # Store result
            result_entry = {
                "file": filename,
                "passed": passed,
                "quality": quality,
                "target_achieved": target_achieved,
                "reason": reason if target_achieved else fail_reasons[0] if fail_reasons else "",
                "metadata": metadata
            }

            if passed:
                results["approved"].append(result_entry)
            else:
                results["rejected"].append(result_entry)

            results["quality_scores"].append(quality["quality_score"])

            progress.update(task, advance=1)

    # Calculate statistics
    results["statistics"] = {
        "approval_rate": len(results["approved"]) / len(metadata_list) * 100 if metadata_list else 0,
        "avg_quality_score": float(np.mean(results["quality_scores"])) if results["quality_scores"] else 0,
        "total_approved": len(results["approved"]),
        "total_rejected": len(results["rejected"]),
    }

    # Display report
    display_validation_report(results)

    # Auto-filter if requested
    if auto_filter and (output_approved or output_rejected):
        filter_images(
            results,
            synthetic_path,
            output_approved,
            output_rejected
        )

    # Save validation report
    report_file = synthetic_path / "validation_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]✓ Validation report saved: {report_file}[/green]")

    return results


def display_validation_report(results: Dict):
    """Display formatted validation report."""
    stats = results["statistics"]

    console.print()
    console.print(Panel.fit(
        f"[bold green]Validation Complete[/bold green]\n"
        f"Approved: {stats['total_approved']} ({stats['approval_rate']:.1f}%)\n"
        f"Rejected: {stats['total_rejected']}\n"
        f"Avg Quality: {stats['avg_quality_score']:.2f}",
        border_style="green"
    ))
    console.print()

    # Show sample rejections
    if results["rejected"]:
        console.print("[bold red]Sample Rejections:[/bold red]\n")
        reject_table = Table(box=box.ROUNDED, border_style="red")
        reject_table.add_column("File", style="dim", max_width=30)
        reject_table.add_column("Reason", style="red")

        for entry in results["rejected"][:10]:
            reject_table.add_row(
                entry["file"],
                entry["reason"]
            )

        if len(results["rejected"]) > 10:
            reject_table.add_row("...", f"[dim](+{len(results['rejected']) - 10} more)[/dim]")

        console.print(reject_table)
        console.print()


def filter_images(
    results: Dict,
    source_dir: Path,
    approved_dir: Optional[str],
    rejected_dir: Optional[str]
):
    """Move images to approved/rejected directories."""

    if approved_dir:
        approved_path = Path(approved_dir)
        approved_path.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Moving approved images to {approved_path}...[/cyan]")
        for entry in results["approved"]:
            src = source_dir / entry["file"]
            dst = approved_path / entry["file"]
            if src.exists():
                shutil.copy2(src, dst)

        console.print(f"[green]✓ Moved {len(results['approved'])} approved images[/green]")

    if rejected_dir:
        rejected_path = Path(rejected_dir)
        rejected_path.mkdir(parents=True, exist_ok=True)

        console.print(f"[cyan]Moving rejected images to {rejected_path}...[/cyan]")
        for entry in results["rejected"]:
            src = source_dir / entry["file"]
            dst = rejected_path / entry["file"]
            if src.exists():
                shutil.copy2(src, dst)

        console.print(f"[green]✓ Moved {len(results['rejected'])} rejected images[/green]")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate synthetic images with YOLO models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate synthetic batch
  python validate_synthetic.py outputs/synthetic/person

  # Validate and auto-filter
  python validate_synthetic.py outputs/synthetic/person \\
    --approved outputs/synthetic/approved \\
    --rejected outputs/synthetic/rejected \\
    --auto-filter

  # Just report, no filtering
  python validate_synthetic.py outputs/synthetic/person --no-auto-filter
        """
    )

    parser.add_argument("synthetic_dir", help="Directory with synthetic images")
    parser.add_argument("--models-dir", default="models",
                       help="Directory with YOLO models")
    parser.add_argument("--approved", help="Output directory for approved images")
    parser.add_argument("--rejected", help="Output directory for rejected images")
    parser.add_argument("--auto-filter", action="store_true", default=True,
                       help="Automatically move images (default: True)")
    parser.add_argument("--no-auto-filter", dest="auto_filter", action="store_false",
                       help="Don't move images, just report")

    args = parser.parse_args()

    validate_synthetic_batch(
        synthetic_dir=args.synthetic_dir,
        models_dir=args.models_dir,
        output_approved=args.approved,
        output_rejected=args.rejected,
        auto_filter=args.auto_filter
    )
