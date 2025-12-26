#!/usr/bin/env python3
"""
Pose Analysis - Human pose detection and classification.

Detects:
- Body keypoints (COCO 17-point format)
- Pose types (standing, sitting, lying, action poses)
- Body orientation (frontal, back, side)
- Hand positions (raised, crossed, gesturing)
- Composition (full body, upper body, portrait)

Integrates with gap analysis for pose-aware dataset balancing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum

import cv2
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ============================================================================
# Pose Enums and Constants
# ============================================================================

class PoseType(str, Enum):
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    KNEELING = "kneeling"
    CROUCHING = "crouching"
    ACTION = "action"  # Running, jumping, dancing, etc.
    UNKNOWN = "unknown"


class BodyOrientation(str, Enum):
    FRONTAL = "frontal"  # Facing camera
    BACK = "back"  # Back to camera
    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    THREE_QUARTER = "three_quarter"
    UNKNOWN = "unknown"


class HandPosition(str, Enum):
    RELAXED = "relaxed"  # Arms at sides
    RAISED = "raised"  # Hands up
    CROSSED = "crossed"  # Arms crossed
    ON_HIPS = "on_hips"
    GESTURING = "gesturing"  # Active hand position
    HIDDEN = "hidden"  # Hands not visible
    UNKNOWN = "unknown"


# COCO keypoint indices
COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye", 2: "right_eye",
    3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder",
    7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist",
    11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee",
    15: "left_ankle", 16: "right_ankle"
}


# ============================================================================
# Pose Detection Models
# ============================================================================

def load_pose_model(model_path: str, device: str = "cuda"):
    """
    Load pose estimation model.

    Supports:
    - YOLOv8-pose
    - Human parsing models
    - Keypoint detection models
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        # Try loading as YOLO pose model
        if "yolo" in model_path.stem.lower():
            model = torch.hub.load('ultralytics/yolov5', 'custom',
                                  path=str(model_path), device=device)
            console.print(f"[green]✓ Loaded YOLO pose model[/green]")
            return model

        # Try loading as torch model directly
        model = torch.load(model_path, map_location=device)
        console.print(f"[green]✓ Loaded pose model[/green]")
        return model

    except Exception as e:
        console.print(f"[yellow]Warning: Could not load pose model: {e}[/yellow]")
        return None


def detect_keypoints_yolo(image_path: str, model) -> Optional[np.ndarray]:
    """
    Detect keypoints using YOLOv8-pose model.

    Returns:
        Array of shape (17, 3) - [x, y, confidence] for each keypoint
        or None if no person detected
    """
    if model is None:
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    try:
        # Run inference
        results = model(img)

        # Extract keypoints (YOLOv8-pose specific)
        if hasattr(results, 'keypoints'):
            keypoints = results.keypoints[0]  # First person
            return keypoints.cpu().numpy()

        return None

    except Exception as e:
        return None


def detect_keypoints_opencv(image_path: str) -> Optional[np.ndarray]:
    """
    Fallback: Detect keypoints using OpenCV's OpenPose implementation.

    Returns:
        Array of shape (17, 3) or None
    """
    # This would require OpenPose weights
    # For now, return None as fallback
    return None


# ============================================================================
# Pose Classification from Keypoints
# ============================================================================

def classify_pose_type(keypoints: np.ndarray) -> PoseType:
    """
    Classify pose type from keypoints.

    Args:
        keypoints: Array (17, 3) with [x, y, confidence]

    Returns:
        PoseType enum
    """
    if keypoints is None or len(keypoints) == 0:
        return PoseType.UNKNOWN

    # Extract key positions
    nose = keypoints[0]
    shoulders = keypoints[[5, 6]]
    hips = keypoints[[11, 12]]
    knees = keypoints[[13, 14]]
    ankles = keypoints[[15, 16]]

    # Check if keypoints are detected (confidence > 0.3)
    nose_visible = nose[2] > 0.3
    shoulders_visible = np.any(shoulders[:, 2] > 0.3)
    hips_visible = np.any(hips[:, 2] > 0.3)
    knees_visible = np.any(knees[:, 2] > 0.3)
    ankles_visible = np.any(ankles[:, 2] > 0.3)

    if not (nose_visible and shoulders_visible):
        return PoseType.UNKNOWN

    # Calculate vertical spans
    if hips_visible and shoulders_visible:
        shoulder_y = np.mean(shoulders[shoulders[:, 2] > 0.3, 1])
        hip_y = np.mean(hips[hips[:, 2] > 0.3, 1])
        torso_length = abs(hip_y - shoulder_y)

        if knees_visible:
            knee_y = np.mean(knees[knees[:, 2] > 0.3, 1])
            upper_leg_length = abs(knee_y - hip_y)

            # Standing: knees below hips, ankles visible
            if knee_y > hip_y and ankles_visible:
                ankle_y = np.mean(ankles[ankles[:, 2] > 0.3, 1])

                # Check if upright (small knee bend)
                if ankle_y > knee_y and (ankle_y - knee_y) > torso_length * 0.5:
                    return PoseType.STANDING
                # Deep crouch
                elif (knee_y - hip_y) < torso_length * 0.3:
                    return PoseType.CROUCHING

            # Sitting: knees roughly level with hips
            elif abs(knee_y - hip_y) < torso_length * 0.5:
                return PoseType.SITTING

            # Kneeling: knees visible but ankles behind
            elif knee_y > hip_y and not ankles_visible:
                return PoseType.KNEELING

    # Lying: nose and shoulders at similar height, body horizontal
    if nose_visible and shoulders_visible:
        nose_y = nose[1]
        shoulder_y = np.mean(shoulders[shoulders[:, 2] > 0.3, 1])

        if abs(nose_y - shoulder_y) < 50:  # Close vertical alignment
            return PoseType.LYING

    # Action pose: large variation in keypoint positions
    if knees_visible and ankles_visible:
        all_y = []
        for kp in [shoulders, hips, knees, ankles]:
            visible = kp[kp[:, 2] > 0.3]
            if len(visible) > 0:
                all_y.append(visible[:, 1])

        if len(all_y) > 0:
            all_y = np.concatenate(all_y)
            y_variance = np.var(all_y)

            # High variance suggests dynamic pose
            if y_variance > 5000:
                return PoseType.ACTION

    return PoseType.UNKNOWN


def classify_body_orientation(keypoints: np.ndarray) -> BodyOrientation:
    """
    Classify body orientation from keypoints.

    Args:
        keypoints: Array (17, 3)

    Returns:
        BodyOrientation enum
    """
    if keypoints is None or len(keypoints) == 0:
        return BodyOrientation.UNKNOWN

    # Extract bilateral keypoints
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    # Check visibility
    left_visible = left_shoulder[2] > 0.3 and left_hip[2] > 0.3
    right_visible = right_shoulder[2] > 0.3 and right_hip[2] > 0.3

    # Both sides visible → frontal or back
    if left_visible and right_visible:
        # Calculate shoulder width
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        # Check if face keypoints visible (frontal vs back)
        nose = keypoints[0]
        eyes = keypoints[[1, 2]]

        face_visible = nose[2] > 0.3 or np.any(eyes[:, 2] > 0.3)

        if face_visible:
            return BodyOrientation.FRONTAL
        else:
            return BodyOrientation.BACK

    # Only one side visible → side view
    elif left_visible and not right_visible:
        return BodyOrientation.SIDE_LEFT

    elif right_visible and not left_visible:
        return BodyOrientation.SIDE_RIGHT

    # Partial visibility → three-quarter view
    elif (left_shoulder[2] > 0.3 or right_shoulder[2] > 0.3):
        return BodyOrientation.THREE_QUARTER

    return BodyOrientation.UNKNOWN


def classify_hand_position(keypoints: np.ndarray) -> HandPosition:
    """
    Classify hand/arm position from keypoints.

    Args:
        keypoints: Array (17, 3)

    Returns:
        HandPosition enum
    """
    if keypoints is None or len(keypoints) == 0:
        return HandPosition.UNKNOWN

    # Extract arm keypoints
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]

    # Check visibility
    wrists_visible = (left_wrist[2] > 0.3) or (right_wrist[2] > 0.3)
    elbows_visible = (left_elbow[2] > 0.3) or (right_elbow[2] > 0.3)

    if not (wrists_visible or elbows_visible):
        return HandPosition.HIDDEN

    # Calculate positions
    shoulder_y = np.mean([left_shoulder[1], right_shoulder[1]])
    hip_y = np.mean([left_hip[1], right_hip[1]]) if left_hip[2] > 0.3 and right_hip[2] > 0.3 else shoulder_y + 100

    # Raised hands: wrists above shoulders
    if wrists_visible:
        wrist_ys = []
        if left_wrist[2] > 0.3:
            wrist_ys.append(left_wrist[1])
        if right_wrist[2] > 0.3:
            wrist_ys.append(right_wrist[1])

        avg_wrist_y = np.mean(wrist_ys)

        if avg_wrist_y < shoulder_y:
            return HandPosition.RAISED

        # On hips: wrists near hip level
        if abs(avg_wrist_y - hip_y) < 50:
            return HandPosition.ON_HIPS

    # Crossed arms: elbows close together horizontally
    if elbows_visible and left_elbow[2] > 0.3 and right_elbow[2] > 0.3:
        elbow_distance = abs(right_elbow[0] - left_elbow[0])
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        if elbow_distance < shoulder_width * 0.6:
            return HandPosition.CROSSED

    # Gesturing: wrists away from body, active position
    if wrists_visible and left_wrist[2] > 0.3 and right_wrist[2] > 0.3:
        # Calculate wrist spread
        wrist_distance = abs(right_wrist[0] - left_wrist[0])
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])

        if wrist_distance > shoulder_width * 1.5:
            return HandPosition.GESTURING

    # Default: relaxed at sides
    return HandPosition.RELAXED


# ============================================================================
# Complete Pose Analysis
# ============================================================================

def analyze_pose(image_path: str, pose_model=None) -> Dict:
    """
    Complete pose analysis of image.

    Returns dict with:
        - keypoints: Raw keypoint array
        - pose_type: Standing/sitting/etc.
        - orientation: Frontal/back/side
        - hand_position: Raised/crossed/etc.
        - confidence: Overall detection confidence
    """
    result = {
        "keypoints": None,
        "pose_type": PoseType.UNKNOWN,
        "orientation": BodyOrientation.UNKNOWN,
        "hand_position": HandPosition.UNKNOWN,
        "confidence": 0.0,
        "has_pose": False
    }

    # Detect keypoints
    if pose_model is not None:
        keypoints = detect_keypoints_yolo(str(image_path), pose_model)
    else:
        keypoints = detect_keypoints_opencv(str(image_path))

    if keypoints is None or len(keypoints) == 0:
        return result

    result["keypoints"] = keypoints.tolist()
    result["has_pose"] = True

    # Calculate overall confidence
    confidences = keypoints[:, 2]
    result["confidence"] = float(np.mean(confidences[confidences > 0]))

    # Classify pose
    result["pose_type"] = classify_pose_type(keypoints)
    result["orientation"] = classify_body_orientation(keypoints)
    result["hand_position"] = classify_hand_position(keypoints)

    return result


# ============================================================================
# Dataset Pose Distribution Analysis
# ============================================================================

def analyze_pose_distribution(
    dataset_dir: str,
    pose_model_path: Optional[str] = None,
    output_file: Optional[str] = None,
    device: str = "cuda"
) -> Dict:
    """
    Analyze pose distribution across entire dataset.

    Returns:
        Dict with distribution stats and recommendations
    """
    dataset_path = Path(dataset_dir)

    console.print(Panel.fit(
        "[bold cyan]Pose Distribution Analysis[/bold cyan]\n"
        f"Dataset: {dataset_path}",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Load pose model if provided
    pose_model = None
    if pose_model_path:
        try:
            pose_model = load_pose_model(pose_model_path, device)
        except Exception as e:
            console.print(f"[yellow]Could not load pose model: {e}[/yellow]")
            console.print("[dim]Continuing with limited analysis...[/dim]\n")

    # Find images
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    images = []
    for ext in extensions:
        images.extend(dataset_path.rglob(f"*{ext}"))
        images.extend(dataset_path.rglob(f"*{ext.upper()}"))

    images = sorted(set(images))
    console.print(f"[green]Found {len(images)} images[/green]\n")

    if not images:
        console.print("[red]No images found[/red]")
        return {}

    # Analyze each image
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

    results = {
        "total_images": len(images),
        "pose_types": defaultdict(int),
        "orientations": defaultdict(int),
        "hand_positions": defaultdict(int),
        "has_pose_count": 0,
        "per_image": {}
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Analyzing poses...", total=len(images))

        for img_path in images:
            analysis = analyze_pose(str(img_path), pose_model)

            if analysis["has_pose"]:
                results["has_pose_count"] += 1
                results["pose_types"][analysis["pose_type"]] += 1
                results["orientations"][analysis["orientation"]] += 1
                results["hand_positions"][analysis["hand_position"]] += 1

            results["per_image"][str(img_path)] = analysis
            progress.update(task, advance=1)

    # Generate recommendations
    results["recommendations"] = generate_pose_recommendations(results)

    # Display report
    display_pose_report(results)

    # Save if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert defaultdicts to regular dicts for JSON
        results_serializable = {
            "total_images": results["total_images"],
            "pose_types": dict(results["pose_types"]),
            "orientations": dict(results["orientations"]),
            "hand_positions": dict(results["hand_positions"]),
            "has_pose_count": results["has_pose_count"],
            "recommendations": results["recommendations"],
            "per_image": results["per_image"]
        }

        with open(output_path, "w") as f:
            json.dump(results_serializable, f, indent=2, default=str)

        console.print(f"\n[green]✓ Saved analysis to {output_path}[/green]")

    return results


def generate_pose_recommendations(analysis: Dict) -> List[Dict]:
    """Generate pose-specific recommendations."""
    recommendations = []
    total = analysis["total_images"]

    if total == 0:
        return recommendations

    # Pose type recommendations
    pose_types = analysis["pose_types"]
    standing_pct = (pose_types.get(PoseType.STANDING, 0) / total) * 100
    sitting_pct = (pose_types.get(PoseType.SITTING, 0) / total) * 100
    action_pct = (pose_types.get(PoseType.ACTION, 0) / total) * 100

    # Ideal: 60% standing, 20% sitting, 15% action, 5% other
    if standing_pct < 50:
        needed = int((0.6 - standing_pct/100) * total)
        recommendations.append({
            "type": "pose_type",
            "target": PoseType.STANDING,
            "count": needed,
            "reason": f"Standing poses are {standing_pct:.1f}% (target: 60%)"
        })

    if sitting_pct < 10:
        needed = int((0.2 - sitting_pct/100) * total)
        recommendations.append({
            "type": "pose_type",
            "target": PoseType.SITTING,
            "count": needed,
            "reason": f"Sitting poses are {sitting_pct:.1f}% (target: 20%)"
        })

    if action_pct < 5:
        needed = int((0.15 - action_pct/100) * total)
        recommendations.append({
            "type": "pose_type",
            "target": PoseType.ACTION,
            "count": needed,
            "reason": f"Action poses are {action_pct:.1f}% (target: 15%)"
        })

    # Hand position recommendations
    hand_positions = analysis["hand_positions"]
    raised_pct = (hand_positions.get(HandPosition.RAISED, 0) / total) * 100
    crossed_pct = (hand_positions.get(HandPosition.CROSSED, 0) / total) * 100

    if raised_pct < 10:
        needed = int((0.15 - raised_pct/100) * total)
        recommendations.append({
            "type": "hand_position",
            "target": HandPosition.RAISED,
            "count": needed,
            "reason": f"Raised hands are {raised_pct:.1f}% (target: 15%)"
        })

    if crossed_pct < 5:
        needed = int((0.1 - crossed_pct/100) * total)
        recommendations.append({
            "type": "hand_position",
            "target": HandPosition.CROSSED,
            "count": needed,
            "reason": f"Crossed arms are {crossed_pct:.1f}% (target: 10%)"
        })

    return recommendations


def display_pose_report(analysis: Dict):
    """Display formatted pose analysis report."""
    total = analysis["total_images"]
    detected = analysis["has_pose_count"]

    console.print()
    console.print(Panel.fit(
        f"[bold green]Pose Analysis Complete[/bold green]\n"
        f"Images analyzed: {total}\n"
        f"Poses detected: {detected} ({detected/total*100:.1f}%)",
        border_style="green"
    ))
    console.print()

    # Pose types table
    if analysis["pose_types"]:
        pose_table = Table(title="Pose Type Distribution", box=box.ROUNDED, border_style="cyan")
        pose_table.add_column("Pose", style="yellow")
        pose_table.add_column("Count", justify="right", style="green")
        pose_table.add_column("Percentage", justify="right", style="dim")

        for pose_type, count in sorted(analysis["pose_types"].items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            pose_table.add_row(str(pose_type), str(count), f"{pct:.1f}%")

        console.print(pose_table)
        console.print()

    # Orientations table
    if analysis["orientations"]:
        orient_table = Table(title="Body Orientation", box=box.ROUNDED, border_style="cyan")
        orient_table.add_column("Orientation", style="yellow")
        orient_table.add_column("Count", justify="right", style="green")
        orient_table.add_column("Percentage", justify="right", style="dim")

        for orientation, count in sorted(analysis["orientations"].items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            orient_table.add_row(str(orientation), str(count), f"{pct:.1f}%")

        console.print(orient_table)
        console.print()

    # Hand positions table
    if analysis["hand_positions"]:
        hands_table = Table(title="Hand Positions", box=box.ROUNDED, border_style="cyan")
        hands_table.add_column("Position", style="yellow")
        hands_table.add_column("Count", justify="right", style="green")
        hands_table.add_column("Percentage", justify="right", style="dim")

        for hand_pos, count in sorted(analysis["hand_positions"].items(), key=lambda x: -x[1]):
            pct = (count / total * 100) if total > 0 else 0
            hands_table.add_row(str(hand_pos), str(count), f"{pct:.1f}%")

        console.print(hands_table)
        console.print()

    # Recommendations
    if analysis["recommendations"]:
        console.print("[bold cyan]Pose Recommendations:[/bold cyan]\n")
        for i, rec in enumerate(analysis["recommendations"], 1):
            console.print(f"  {i}. Generate [bold]{rec['count']}[/bold] images: {rec['target']}")
            console.print(f"     [dim]{rec['reason']}[/dim]")
        console.print()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze pose distribution in dataset")
    parser.add_argument("dataset", help="Directory containing images")
    parser.add_argument("--pose-model", help="Path to pose estimation model (.pt)")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    analyze_pose_distribution(
        dataset_dir=args.dataset,
        pose_model_path=args.pose_model,
        output_file=args.output,
        device=args.device
    )
