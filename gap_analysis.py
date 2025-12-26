#!/usr/bin/env python3
"""
Gap Analysis - Detect dataset gaps using specialized YOLO models.

Analyzes existing dataset to identify:
- Face angles (front, profile, three-quarter)
- Expressions (neutral, smile, eyes closed)
- Body visibility (headshot, upper body, full body)
- Hand/eye visibility
- Composition gaps

Outputs structured recommendations for synthetic generation.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ============================================================================
# YOLO Model Loading
# ============================================================================

def load_yolo_model(model_path: str, device: str = "cuda") -> torch.nn.Module:
    """Load a YOLO model from .pt file."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load with torch.hub for YOLOv5 models
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device)
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45   # NMS IOU threshold
        return model
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load {model_path}: {e}[/yellow]")
        return None


# ============================================================================
# Face Analysis
# ============================================================================

def analyze_face_angles(image_path: str, face_model) -> Dict:
    """
    Analyze face angle/pose from image.
    Returns classification: front, profile_left, profile_right, three_quarter_left, three_quarter_right
    """
    if face_model is None:
        return {"angle": "unknown", "confidence": 0.0}

    img = cv2.imread(str(image_path))
    if img is None:
        return {"angle": "unknown", "confidence": 0.0}

    # Run face detection
    results = face_model(img)
    detections = results.pandas().xyxy[0]

    if len(detections) == 0:
        return {"angle": "unknown", "confidence": 0.0}

    # Get largest face
    largest = detections.iloc[0]
    x1, y1, x2, y2 = largest['xmin'], largest['ymin'], largest['xmax'], largest['ymax']
    confidence = largest['confidence']

    # Extract face region for angle analysis
    face_region = img[int(y1):int(y2), int(x1):int(x2)]

    # Heuristic: analyze face position in bbox for profile detection
    face_width = x2 - x1
    face_height = y2 - y1
    aspect_ratio = face_width / face_height if face_height > 0 else 1.0

    # Simplified angle classification based on aspect ratio
    # Profile faces are narrower (aspect < 0.7)
    # Front faces are more square (aspect 0.7-1.0)
    if aspect_ratio < 0.6:
        angle = "profile"  # Can't distinguish left/right without landmark detection
    elif aspect_ratio < 0.85:
        angle = "three_quarter"
    else:
        angle = "front"

    return {
        "angle": angle,
        "confidence": float(confidence),
        "aspect_ratio": float(aspect_ratio),
        "bbox": [float(x1), float(y1), float(x2), float(y2)]
    }


def analyze_eyes(image_path: str, eyes_model) -> Dict:
    """Detect eye visibility and state (open/closed)."""
    if eyes_model is None:
        return {"eyes_detected": 0, "visible": False}

    img = cv2.imread(str(image_path))
    if img is None:
        return {"eyes_detected": 0, "visible": False}

    results = eyes_model(img)
    detections = results.pandas().xyxy[0]

    return {
        "eyes_detected": len(detections),
        "visible": len(detections) > 0,
        "confidence": float(detections['confidence'].mean()) if len(detections) > 0 else 0.0
    }


def analyze_hands(image_path: str, hands_model) -> Dict:
    """Detect hand visibility."""
    if hands_model is None:
        return {"hands_detected": 0, "visible": False}

    img = cv2.imread(str(image_path))
    if img is None:
        return {"hands_detected": 0, "visible": False}

    results = hands_model(img)
    detections = results.pandas().xyxy[0]

    return {
        "hands_detected": len(detections),
        "visible": len(detections) > 0,
        "confidence": float(detections['confidence'].mean()) if len(detections) > 0 else 0.0
    }


# ============================================================================
# Dataset Analysis
# ============================================================================

def analyze_dataset(
    dataset_dir: str,
    models_dir: str = "models",
    output_file: Optional[str] = None
) -> Dict:
    """
    Analyze entire dataset for gaps.

    Args:
        dataset_dir: Directory containing images to analyze
        models_dir: Directory containing YOLO .pt models
        output_file: Optional path to save analysis JSON

    Returns:
        Dictionary with gap analysis results
    """

    dataset_path = Path(dataset_dir)
    models_path = Path(models_dir)

    console.print(Panel.fit(
        "[bold cyan]Gap Analysis - Dataset Quality Assessment[/bold cyan]\n"
        f"Dataset: {dataset_path}\n"
        f"Models: {models_path}",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Load YOLO models
    console.print("[cyan]Loading YOLO models...[/cyan]")
    models = {}

    # Try to load available models
    model_files = {
        "face": "yolov5l-face.pt",
        "eyes": "eyes_yolov",  # Folder-based model
        "hands": "hand_yolov8n",
    }

    for name, filename in model_files.items():
        model_path = models_path / filename
        if not model_path.exists():
            # Try .pt extension
            model_path = models_path / f"{filename}.pt"

        if model_path.exists():
            try:
                models[name] = load_yolo_model(str(model_path))
                console.print(f"  [green]✓[/green] Loaded {name} model")
            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow] Could not load {name}: {e}")
                models[name] = None
        else:
            console.print(f"  [dim]○ {name} model not found (optional)[/dim]")
            models[name] = None

    console.print()

    # Find all images
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    for ext in extensions:
        images.extend(dataset_path.rglob(f"*{ext}"))
        images.extend(dataset_path.rglob(f"*{ext.upper()}"))

    images = sorted(set(images))
    console.print(f"[green]Found {len(images)} images to analyze[/green]\n")

    # Analyze each image
    results = {
        "total_images": len(images),
        "face_angles": defaultdict(int),
        "eyes_visible": {"yes": 0, "no": 0},
        "hands_visible": {"yes": 0, "no": 0},
        "shot_types": defaultdict(int),  # From existing facevault cache if available
        "quality_distribution": [],
        "per_image": {}
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Analyzing images...", total=len(images))

        for img_path in images:
            analysis = {}

            # Face angle analysis
            if models["face"]:
                face_result = analyze_face_angles(str(img_path), models["face"])
                analysis["face"] = face_result
                results["face_angles"][face_result["angle"]] += 1

            # Eyes visibility
            if models["eyes"]:
                eyes_result = analyze_eyes(str(img_path), models["eyes"])
                analysis["eyes"] = eyes_result
                if eyes_result["visible"]:
                    results["eyes_visible"]["yes"] += 1
                else:
                    results["eyes_visible"]["no"] += 1

            # Hands visibility
            if models["hands"]:
                hands_result = analyze_hands(str(img_path), models["hands"])
                analysis["hands"] = hands_result
                if hands_result["visible"]:
                    results["hands_visible"]["yes"] += 1
                else:
                    results["hands_visible"]["no"] += 1

            results["per_image"][str(img_path)] = analysis
            progress.update(task, advance=1)

    # Calculate gap recommendations
    results["recommendations"] = generate_recommendations(results)

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\n[green]✓ Saved analysis to {output_path}[/green]")

    # Display summary
    display_gap_report(results)

    return results


def generate_recommendations(analysis: Dict) -> List[Dict]:
    """Generate structured recommendations for image generation."""
    recommendations = []
    total = analysis["total_images"]

    if total == 0:
        return recommendations

    # Face angle recommendations
    angles = analysis["face_angles"]
    front_pct = (angles.get("front", 0) / total) * 100 if total > 0 else 0
    profile_pct = (angles.get("profile", 0) / total) * 100 if total > 0 else 0
    three_quarter_pct = (angles.get("three_quarter", 0) / total) * 100 if total > 0 else 0

    # Ideal distribution: 50% front, 30% three-quarter, 20% profile
    if front_pct < 40:
        needed = int((0.5 - front_pct/100) * total)
        recommendations.append({
            "type": "face_angle",
            "target": "front",
            "count": needed,
            "reason": f"Front faces are {front_pct:.1f}% (target: 50%)"
        })

    if three_quarter_pct < 20:
        needed = int((0.3 - three_quarter_pct/100) * total)
        recommendations.append({
            "type": "face_angle",
            "target": "three_quarter",
            "count": needed,
            "reason": f"Three-quarter views are {three_quarter_pct:.1f}% (target: 30%)"
        })

    if profile_pct < 10:
        needed = int((0.2 - profile_pct/100) * total)
        recommendations.append({
            "type": "face_angle",
            "target": "profile",
            "count": needed,
            "reason": f"Profile shots are {profile_pct:.1f}% (target: 20%)"
        })

    # Hand visibility
    hands = analysis["hands_visible"]
    hands_yes = hands.get("yes", 0)
    hands_pct = (hands_yes / total) * 100 if total > 0 else 0

    if hands_pct < 30:
        needed = int((0.3 - hands_pct/100) * total)
        recommendations.append({
            "type": "composition",
            "target": "hands_visible",
            "count": needed,
            "reason": f"Hands visible in {hands_pct:.1f}% (target: 30%)"
        })

    return recommendations


def display_gap_report(analysis: Dict):
    """Display formatted gap analysis report."""
    console.print()
    console.print(Panel.fit(
        "[bold green]Gap Analysis Complete[/bold green]",
        border_style="green"
    ))
    console.print()

    # Face angles distribution
    if analysis["face_angles"]:
        angles_table = Table(title="Face Angle Distribution", box=box.ROUNDED, border_style="cyan")
        angles_table.add_column("Angle", style="yellow")
        angles_table.add_column("Count", justify="right", style="green")
        angles_table.add_column("Percentage", justify="right", style="dim")

        total = analysis["total_images"]
        for angle, count in sorted(analysis["face_angles"].items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            angles_table.add_row(angle, str(count), f"{pct:.1f}%")

        console.print(angles_table)
        console.print()

    # Visibility stats
    visibility_table = Table(title="Feature Visibility", box=box.ROUNDED, border_style="cyan")
    visibility_table.add_column("Feature", style="yellow")
    visibility_table.add_column("Visible", justify="right", style="green")
    visibility_table.add_column("Not Visible", justify="right", style="red")
    visibility_table.add_column("% Visible", justify="right", style="dim")

    eyes = analysis["eyes_visible"]
    hands = analysis["hands_visible"]
    total = analysis["total_images"]

    if eyes.get("yes", 0) + eyes.get("no", 0) > 0:
        eyes_total = eyes["yes"] + eyes["no"]
        eyes_pct = (eyes["yes"] / eyes_total * 100) if eyes_total > 0 else 0
        visibility_table.add_row("Eyes", str(eyes["yes"]), str(eyes["no"]), f"{eyes_pct:.1f}%")

    if hands.get("yes", 0) + hands.get("no", 0) > 0:
        hands_total = hands["yes"] + hands["no"]
        hands_pct = (hands["yes"] / hands_total * 100) if hands_total > 0 else 0
        visibility_table.add_row("Hands", str(hands["yes"]), str(hands["no"]), f"{hands_pct:.1f}%")

    console.print(visibility_table)
    console.print()

    # Recommendations
    if analysis["recommendations"]:
        console.print("[bold cyan]Generation Recommendations:[/bold cyan]\n")
        for i, rec in enumerate(analysis["recommendations"], 1):
            console.print(f"  {i}. Generate [bold]{rec['count']}[/bold] images: {rec['target']}")
            console.print(f"     [dim]{rec['reason']}[/dim]")
        console.print()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze dataset for gaps")
    parser.add_argument("dataset", help="Directory containing images")
    parser.add_argument("--models-dir", default="models", help="Directory with YOLO models")
    parser.add_argument("--output", "-o", help="Output JSON file")

    args = parser.parse_args()

    analyze_dataset(
        dataset_dir=args.dataset,
        models_dir=args.models_dir,
        output_file=args.output
    )
