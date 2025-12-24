#!/usr/bin/env python3
"""
FaceVault - Beautiful face organization for your photo library.
Scan, cluster, label, and export faces with ease.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box
from rich.prompt import Prompt, Confirm
from PIL import Image

# Project paths
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "facevault"

console = Console()


# ============================================================================
# Core Face Detection & Clustering
# ============================================================================

def load_insightface(device: str = "cuda"):
    """Load InsightFace model with progress indicator."""
    from insightface.app import FaceAnalysis

    with console.status("[bold cyan]Loading InsightFace model (buffalo_l)..."):
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0 if device == "cuda" else -1)

    console.print("✓ Model loaded", style="bold green")
    return app


def extract_faces(app, image_path: Path, min_score: float = 0.5, min_size: int = 50) -> List[Dict]:
    """
    Extract face embeddings from an image with quality filtering.

    Args:
        app: InsightFace app instance
        image_path: Path to image file
        min_score: Minimum detection confidence (0-1)
        min_size: Minimum face size in pixels

    Returns:
        List of face dictionaries with embeddings and metadata
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return []

    faces = app.get(img)

    results = []
    for i, face in enumerate(faces):
        # Quality filtering
        if face.det_score < min_score:
            continue

        # Size filtering
        bbox = face.bbox
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width < min_size or height < min_size:
            continue

        results.append({
            "embedding": face.embedding.tolist(),
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score),
            "size": (int(width), int(height)),
            "age": int(face.age) if hasattr(face, 'age') else None,
            "gender": "F" if face.gender == 0 else "M" if hasattr(face, 'gender') else None,
        })

    return results


def cluster_embeddings(embeddings: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """
    Cluster face embeddings using DBSCAN with cosine distance.

    Args:
        embeddings: Array of face embeddings (N x embedding_dim)
        threshold: Cosine distance threshold (lower = stricter matching)

    Returns:
        Cluster labels (-1 for noise/unclustered)
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize

    embeddings_norm = normalize(embeddings)

    clustering = DBSCAN(
        eps=threshold,
        min_samples=2,
        metric="cosine"
    ).fit(embeddings_norm)

    return clustering.labels_


# ============================================================================
# Scan Command
# ============================================================================

@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default=str(OUTPUT_DIR), help='Output directory')
@click.option('--min-score', type=float, default=0.5, help='Minimum face detection confidence (0-1)')
@click.option('--min-size', type=int, default=50, help='Minimum face size in pixels')
@click.option('--no-resume', is_flag=True, help="Don't resume from existing cache")
def scan(input_dir: str, output: str, min_score: float, min_size: int, no_resume: bool):
    """Scan photos and extract face embeddings."""

    input_path = Path(input_dir)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_file = output_path / "face_cache.json"

    # Header
    console.print(Panel.fit(
        "[bold cyan]FaceVault Scanner[/bold cyan]\n"
        f"Input: {input_path}\n"
        f"Quality: min_score={min_score}, min_size={min_size}px",
        border_style="cyan"
    ))

    # Load cache if resuming
    cache = {}
    if not no_resume and cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        console.print(f"[yellow]Resuming:[/yellow] {len(cache)} cached entries loaded")

    # Find all images
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    images = []

    with console.status("[bold cyan]Discovering images...") as status:
        for ext in extensions:
            images.extend(input_path.rglob(f"*{ext}"))
            images.extend(input_path.rglob(f"*{ext.upper()}"))
        images = sorted(set(images))

    console.print(f"[green]✓[/green] Found {len(images)} images")

    # Filter already processed
    to_process = [img for img in images if str(img) not in cache]

    if not to_process:
        console.print("[yellow]All images already processed![/yellow]")
        return

    console.print(f"[cyan]Processing {len(to_process)} new images...[/cyan]\n")

    # Load model
    app = load_insightface()

    # Process with rich progress bar
    total_faces = 0
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Extracting faces...", total=len(to_process))

        for img_path in to_process:
            try:
                faces = extract_faces(app, img_path, min_score=min_score, min_size=min_size)
                cache[str(img_path)] = {
                    "faces": faces,
                    "count": len(faces),
                    "scanned_at": datetime.now().isoformat(),
                }
                total_faces += len(faces)

                # Save periodically
                if len(cache) % 100 == 0:
                    with open(cache_file, "w") as f:
                        json.dump(cache, f)

            except Exception as e:
                errors += 1
                cache[str(img_path)] = {
                    "faces": [],
                    "count": 0,
                    "error": str(e),
                    "scanned_at": datetime.now().isoformat(),
                }

            progress.update(task, advance=1)

    # Final save
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

    # Summary
    console.print()
    summary = Table(show_header=False, box=box.ROUNDED, border_style="green")
    summary.add_row("📸 Images scanned", str(len(to_process)))
    summary.add_row("👤 Faces detected", str(total_faces))
    summary.add_row("⚠️  Errors", str(errors))
    summary.add_row("💾 Cache saved to", str(cache_file))

    console.print(Panel(summary, title="[bold green]Scan Complete", border_style="green"))


# ============================================================================
# Cluster Command
# ============================================================================

@click.command()
@click.option('--output', '-o', type=click.Path(), default=str(OUTPUT_DIR), help='Output directory')
@click.option('--threshold', '-t', type=float, default=0.4, help='Clustering threshold (lower = stricter)')
@click.option('--min-cluster', type=int, default=3, help='Minimum faces per cluster')
@click.option('--preview', is_flag=True, help='Preview clusters before organizing')
def cluster(output: str, threshold: float, min_cluster: int, preview: bool):
    """Cluster detected faces into groups of people."""

    output_path = Path(output)
    cache_file = output_path / "face_cache.json"

    if not cache_file.exists():
        console.print("[red]Error:[/red] No face cache found. Run 'scan' first.")
        return

    # Header
    console.print(Panel.fit(
        "[bold magenta]FaceVault Clustering[/bold magenta]\n"
        f"Threshold: {threshold} | Min cluster size: {min_cluster}",
        border_style="magenta"
    ))

    # Load cache
    with console.status("[bold cyan]Loading face data..."):
        with open(cache_file) as f:
            cache = json.load(f)

    # Collect all embeddings
    all_faces = []
    for img_path, data in cache.items():
        for i, face in enumerate(data.get("faces", [])):
            if "embedding" in face:
                all_faces.append({
                    "image": img_path,
                    "face_idx": i,
                    "embedding": np.array(face["embedding"]),
                    "bbox": face.get("bbox"),
                    "score": face.get("det_score", 0),
                    "size": face.get("size", (0, 0)),
                })

    if not all_faces:
        console.print("[red]Error:[/red] No faces found in cache!")
        return

    console.print(f"[green]✓[/green] Loaded {len(all_faces)} faces from {len(cache)} images\n")

    # Cluster
    with console.status("[bold cyan]Clustering faces..."):
        embeddings = np.stack([f["embedding"] for f in all_faces])
        labels = cluster_embeddings(embeddings, threshold)

    # Analyze clusters
    cluster_counts = defaultdict(int)
    for label in labels:
        cluster_counts[label] += 1

    valid_clusters = {
        k: v for k, v in cluster_counts.items()
        if k != -1 and v >= min_cluster
    }

    console.print(f"[green]✓[/green] Found {len(valid_clusters)} clusters\n")

    # Build cluster assignments
    cluster_data = defaultdict(list)
    for face, label in zip(all_faces, labels):
        if label in valid_clusters:
            cluster_data[label].append({
                "image": face["image"],
                "score": face["score"],
                "size": face["size"],
            })

    # Preview
    if preview or True:  # Always show preview
        preview_table = Table(title="Cluster Preview", box=box.ROUNDED, border_style="magenta")
        preview_table.add_column("Cluster", style="cyan", width=12)
        preview_table.add_column("Faces", justify="right", style="yellow")
        preview_table.add_column("Sample Images", style="dim")

        for cluster_id in sorted(valid_clusters.keys(), key=lambda k: -cluster_counts[k]):
            cluster_name = f"person_{cluster_id:03d}"
            face_count = len(cluster_data[cluster_id])
            samples = [Path(f["image"]).name for f in cluster_data[cluster_id][:3]]
            preview_table.add_row(
                cluster_name,
                str(face_count),
                ", ".join(samples) + ("..." if face_count > 3 else "")
            )

        console.print(preview_table)
        console.print()

    # Show stats
    stats_panel = Table(show_header=False, box=None)
    stats_panel.add_row("👥 Valid clusters", f"[bold]{len(valid_clusters)}[/bold]")
    stats_panel.add_row("👤 Total faces", f"[bold]{sum(valid_clusters.values())}[/bold]")
    stats_panel.add_row("⭐ Largest cluster", f"[bold]{max(valid_clusters.values())} faces[/bold]")
    stats_panel.add_row("🗑️  Unclustered", f"[dim]{cluster_counts.get(-1, 0)} faces[/dim]")

    console.print(Panel(stats_panel, title="Statistics", border_style="cyan"))
    console.print()

    # Confirm if preview mode
    if preview:
        if not Confirm.ask("Proceed with saving cluster assignments?", default=True):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # Save cluster assignments
    assignments = {}
    for cluster_id, faces in cluster_data.items():
        cluster_name = f"person_{cluster_id:03d}"
        assignments[cluster_name] = {
            "images": list(set(f["image"] for f in faces)),
            "count": len(faces),
            "avg_score": float(np.mean([f["score"] for f in faces])),
        }

    assignments_file = output_path / "cluster_assignments.json"
    with open(assignments_file, "w") as f:
        json.dump(assignments, f, indent=2)

    console.print(f"[green]✓ Saved cluster assignments to {assignments_file}[/green]")


# ============================================================================
# Label Command
# ============================================================================

@click.command()
@click.option('--output', '-o', type=click.Path(), default=str(OUTPUT_DIR), help='Output directory')
@click.option('--show-paths', is_flag=True, help='Show sample file paths during labeling')
def label(output: str, show_paths: bool):
    """Interactively label face clusters with names."""

    output_path = Path(output)
    assignments_file = output_path / "cluster_assignments.json"
    labels_file = output_path / "cluster_labels.json"

    if not assignments_file.exists():
        console.print("[red]Error:[/red] No cluster assignments found. Run 'cluster' first.")
        return

    # Header
    console.print(Panel.fit(
        "[bold yellow]FaceVault Labeling[/bold yellow]\n"
        "Give your clusters meaningful names!",
        border_style="yellow"
    ))
    console.print()

    with open(assignments_file) as f:
        assignments = json.load(f)

    # Load existing labels
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)
    else:
        labels = {}

    # Sort by face count (largest first)
    sorted_clusters = sorted(
        assignments.items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )

    console.print(f"[cyan]Found {len(sorted_clusters)} clusters to label[/cyan]\n")

    labeled_count = 0

    for cluster_name, data in sorted_clusters:
        # Skip if already labeled
        if cluster_name in labels:
            console.print(f"[dim]{cluster_name}: already labeled as '{labels[cluster_name]}'[/dim]")
            continue

        # Show cluster info
        info_table = Table(show_header=False, box=box.SIMPLE, border_style="yellow")
        info_table.add_row("Cluster", f"[bold cyan]{cluster_name}[/bold cyan]")
        info_table.add_row("Faces", f"[yellow]{data['count']}[/yellow]")
        info_table.add_row("Avg Score", f"{data['avg_score']:.2f}")

        if show_paths:
            samples = data["images"][:3]
            for i, img in enumerate(samples, 1):
                info_table.add_row(f"Sample {i}", f"[dim]{Path(img).name}[/dim]")

        console.print(Panel(info_table, border_style="yellow", expand=False))

        # Prompt for name
        name = Prompt.ask(
            "  [bold]Name this person[/bold] (or 'skip' to skip, 'done' to finish)",
            default=""
        )

        if name.lower() == 'done':
            break
        elif name.lower() == 'skip' or not name.strip():
            console.print("  [dim]Skipped[/dim]\n")
            continue
        else:
            labels[cluster_name] = name.strip()
            labeled_count += 1
            console.print(f"  [green]✓ Labeled as '{name.strip()}'[/green]\n")

        # Save incrementally
        with open(labels_file, "w") as f:
            json.dump(labels, f, indent=2)

    # Summary
    console.print()
    console.print(Panel.fit(
        f"[bold green]Labeling Complete[/bold green]\n"
        f"Labeled: {labeled_count} new clusters\n"
        f"Total labels: {len(labels)}\n"
        f"Saved to: {labels_file}",
        border_style="green"
    ))


# ============================================================================
# Export Command
# ============================================================================

@click.command()
@click.option('--output', '-o', type=click.Path(), default=str(OUTPUT_DIR), help='Output directory')
@click.option('--format', '-f', type=click.Choice(['organized', 'lora', 'json']), default='organized', help='Export format')
@click.option('--person', type=str, help='Export specific person/cluster only')
@click.option('--symlink', is_flag=True, help='Use symlinks instead of copying files')
@click.option('--all', 'export_all', is_flag=True, help='Export all clusters')
def export(output: str, format: str, person: Optional[str], symlink: bool, export_all: bool):
    """Export organized faces to various formats."""

    output_path = Path(output)
    assignments_file = output_path / "cluster_assignments.json"
    labels_file = output_path / "cluster_labels.json"

    if not assignments_file.exists():
        console.print("[red]Error:[/red] No cluster assignments found. Run 'cluster' first.")
        return

    with open(assignments_file) as f:
        assignments = json.load(f)

    # Load labels if available
    labels = {}
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)

    # Determine what to export
    if person:
        if person not in assignments:
            console.print(f"[red]Error:[/red] Cluster '{person}' not found.")
            return
        to_export = {person: assignments[person]}
    elif export_all:
        to_export = assignments
    else:
        console.print("[yellow]Specify --person <name> or --all to export[/yellow]")
        return

    # Header
    console.print(Panel.fit(
        f"[bold blue]FaceVault Export[/bold blue]\n"
        f"Format: {format} | Mode: {'symlink' if symlink else 'copy'}",
        border_style="blue"
    ))
    console.print()

    # Export based on format
    if format == "organized":
        export_organized(to_export, labels, output_path, symlink)
    elif format == "lora":
        export_lora(to_export, labels, output_path, symlink)
    elif format == "json":
        export_json(to_export, labels, output_path)


def export_organized(assignments: Dict, labels: Dict, output_path: Path, use_symlink: bool):
    """Export to organized folder structure."""
    export_dir = output_path / "organized"
    export_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Organizing files...", total=len(assignments))

        for cluster_name, data in assignments.items():
            # Use label if available, otherwise use cluster name
            folder_name = labels.get(cluster_name, cluster_name)
            cluster_dir = export_dir / folder_name
            cluster_dir.mkdir(exist_ok=True)

            for img_path in data["images"]:
                src = Path(img_path)
                dst = cluster_dir / src.name

                if not dst.exists():
                    if use_symlink:
                        dst.symlink_to(src.absolute())
                    else:
                        shutil.copy2(src, dst)
                    total_files += 1

            progress.update(task, advance=1)

    console.print(f"[green]✓ Exported {total_files} files to {export_dir}[/green]")


def export_lora(assignments: Dict, labels: Dict, output_path: Path, use_symlink: bool):
    """Export in LoRA training-ready format."""
    export_dir = output_path / "lora_ready"
    export_dir.mkdir(parents=True, exist_ok=True)

    for cluster_name, data in assignments.items():
        person_name = labels.get(cluster_name, cluster_name)
        trigger_word = person_name.lower().replace(" ", "_")

        # Create person directory
        person_dir = export_dir / person_name
        person_dir.mkdir(exist_ok=True)

        # Copy/link images with numbered names
        for i, img_path in enumerate(data["images"], 1):
            src = Path(img_path)
            dst = person_dir / f"{trigger_word}_{i:04d}{src.suffix}"

            if not dst.exists():
                if use_symlink:
                    dst.symlink_to(src.absolute())
                else:
                    shutil.copy2(src, dst)

        # Create metadata file
        metadata = {
            "person": person_name,
            "trigger_word": trigger_word,
            "image_count": len(data["images"]),
            "avg_detection_score": data.get("avg_score", 0),
            "created_at": datetime.now().isoformat(),
        }

        with open(person_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    console.print(f"[green]✓ Exported LoRA datasets to {export_dir}[/green]")


def export_json(assignments: Dict, labels: Dict, output_path: Path):
    """Export as consolidated JSON manifest."""
    manifest = {
        "exported_at": datetime.now().isoformat(),
        "clusters": []
    }

    for cluster_name, data in assignments.items():
        manifest["clusters"].append({
            "id": cluster_name,
            "name": labels.get(cluster_name, cluster_name),
            "face_count": data["count"],
            "avg_score": data["avg_score"],
            "images": data["images"],
        })

    manifest_file = output_path / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[green]✓ Exported manifest to {manifest_file}[/green]")


# ============================================================================
# Stats Command
# ============================================================================

@click.command()
@click.option('--output', '-o', type=click.Path(), default=str(OUTPUT_DIR), help='Output directory')
def stats(output: str):
    """Show statistics about scanned and clustered faces."""

    output_path = Path(output)
    cache_file = output_path / "face_cache.json"
    assignments_file = output_path / "cluster_assignments.json"
    labels_file = output_path / "cluster_labels.json"

    console.print(Panel.fit("[bold]FaceVault Statistics[/bold]", border_style="cyan"))
    console.print()

    # Scan stats
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)

        total_images = len(cache)
        total_faces = sum(data.get("count", 0) for data in cache.values())
        errors = sum(1 for data in cache.values() if "error" in data)

        scan_table = Table(title="Scan Results", box=box.ROUNDED, border_style="green")
        scan_table.add_column("Metric", style="cyan")
        scan_table.add_column("Value", justify="right", style="yellow")

        scan_table.add_row("📸 Images scanned", f"{total_images:,}")
        scan_table.add_row("👤 Faces detected", f"{total_faces:,}")
        scan_table.add_row("📊 Avg faces/image", f"{total_faces/total_images:.2f}" if total_images > 0 else "0")
        scan_table.add_row("⚠️  Errors", str(errors))

        console.print(scan_table)
        console.print()
    else:
        console.print("[yellow]No scan data found. Run 'scan' first.[/yellow]\n")

    # Cluster stats
    if assignments_file.exists():
        with open(assignments_file) as f:
            assignments = json.load(f)

        cluster_counts = [data["count"] for data in assignments.values()]

        cluster_table = Table(title="Cluster Results", box=box.ROUNDED, border_style="magenta")
        cluster_table.add_column("Metric", style="cyan")
        cluster_table.add_column("Value", justify="right", style="yellow")

        cluster_table.add_row("👥 Clusters found", str(len(assignments)))
        cluster_table.add_row("👤 Clustered faces", f"{sum(cluster_counts):,}")
        cluster_table.add_row("⭐ Largest cluster", f"{max(cluster_counts)} faces")
        cluster_table.add_row("📊 Avg cluster size", f"{np.mean(cluster_counts):.1f}")

        console.print(cluster_table)
        console.print()
    else:
        console.print("[yellow]No cluster data found. Run 'cluster' first.[/yellow]\n")

    # Label stats
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)

        console.print(f"[green]✓[/green] {len(labels)} clusters labeled")
        console.print()


# ============================================================================
# Main CLI
# ============================================================================

@click.group()
@click.version_option(version="1.0.0", prog_name="FaceVault")
def cli():
    """
    🗂️  FaceVault - Beautiful face organization for your photo library.

    Workflow:
      1. scan    - Extract faces from photos
      2. cluster - Group similar faces
      3. label   - Name your clusters
      4. export  - Organize files by person
    """
    pass


# Add commands
cli.add_command(scan)
cli.add_command(cluster)
cli.add_command(label)
cli.add_command(export)
cli.add_command(stats)


if __name__ == "__main__":
    cli()
