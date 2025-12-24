#!/usr/bin/env python3
"""
Organize photos by face recognition.
Uses InsightFace to extract embeddings, clusters similar faces together.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "faces"


def load_insightface(device: str = "cuda"):
    """Load InsightFace model."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0 if device == "cuda" else -1)
    return app


def extract_faces(app, image_path: Path) -> list[dict]:
    """Extract face embeddings from an image."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return []

    # InsightFace expects BGR
    faces = app.get(img)

    results = []
    for i, face in enumerate(faces):
        results.append({
            "embedding": face.embedding.tolist(),
            "bbox": face.bbox.tolist(),
            "det_score": float(face.det_score),
            "age": int(face.age) if hasattr(face, 'age') else None,
            "gender": "F" if face.gender == 0 else "M" if hasattr(face, 'gender') else None,
        })

    return results


def cluster_embeddings(embeddings: np.ndarray, threshold: float = 0.4):
    """
    Cluster face embeddings using DBSCAN.
    threshold: cosine distance threshold (lower = stricter matching)
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize

    # Normalize for cosine similarity
    embeddings_norm = normalize(embeddings)

    # DBSCAN with cosine distance
    clustering = DBSCAN(
        eps=threshold,
        min_samples=2,
        metric="cosine"
    ).fit(embeddings_norm)

    return clustering.labels_


def scan_photos(
    input_dir: Path,
    app,
    resume: bool = True,
) -> dict:
    """Scan directory for faces."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OUTPUT_DIR / "face_cache.json"

    # Load cache if resuming
    if resume and cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached entries")
    else:
        cache = {}

    # Find images
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    images = []
    for ext in extensions:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))

    images = sorted(set(images))
    print(f"Found {len(images)} images")

    # Filter already processed
    to_process = [img for img in images if str(img) not in cache]
    print(f"Processing {len(to_process)} new images")

    for img_path in tqdm(to_process, desc="Extracting faces"):
        try:
            faces = extract_faces(app, img_path)
            cache[str(img_path)] = {
                "faces": faces,
                "count": len(faces),
            }

            # Save periodically
            if len(cache) % 100 == 0:
                with open(cache_file, "w") as f:
                    json.dump(cache, f)

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            cache[str(img_path)] = {"faces": [], "count": 0, "error": str(e)}

    # Final save
    with open(cache_file, "w") as f:
        json.dump(cache, f)

    return cache


def organize_by_clusters(
    cache: dict,
    output_dir: Path,
    threshold: float = 0.4,
    min_cluster_size: int = 3,
    copy_files: bool = False,
):
    """Organize photos into folders by face cluster."""

    # Collect all embeddings with source info
    all_faces = []
    for img_path, data in cache.items():
        for i, face in enumerate(data.get("faces", [])):
            if "embedding" in face:
                all_faces.append({
                    "image": img_path,
                    "face_idx": i,
                    "embedding": np.array(face["embedding"]),
                    "bbox": face.get("bbox"),
                })

    if not all_faces:
        print("No faces found!")
        return

    print(f"Clustering {len(all_faces)} faces...")

    # Stack embeddings
    embeddings = np.stack([f["embedding"] for f in all_faces])

    # Cluster
    labels = cluster_embeddings(embeddings, threshold)

    # Count clusters
    cluster_counts = defaultdict(int)
    for label in labels:
        cluster_counts[label] += 1

    valid_clusters = {k for k, v in cluster_counts.items() if k != -1 and v >= min_cluster_size}
    print(f"Found {len(valid_clusters)} clusters with >= {min_cluster_size} faces")
    print(f"Unclustered faces: {cluster_counts[-1]}")

    # Organize
    clusters_dir = output_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    cluster_data = defaultdict(list)

    for face, label in zip(all_faces, labels):
        if label in valid_clusters:
            cluster_data[label].append(face["image"])

            # Optionally copy/link files
            if copy_files:
                cluster_dir = clusters_dir / f"person_{label:03d}"
                cluster_dir.mkdir(exist_ok=True)

                src = Path(face["image"])
                dst = cluster_dir / src.name

                if not dst.exists():
                    shutil.copy2(src, dst)

    # Save cluster assignments
    assignments = {
        f"person_{k:03d}": list(set(v))
        for k, v in cluster_data.items()
    }

    with open(output_dir / "cluster_assignments.json", "w") as f:
        json.dump(assignments, f, indent=2)

    # Summary
    print(f"\nCluster summary:")
    for cluster_name, images in sorted(assignments.items(), key=lambda x: -len(x[1])):
        print(f"  {cluster_name}: {len(images)} photos")

    print(f"\nSaved to {output_dir}")


def label_clusters(output_dir: Path):
    """Interactive cluster labeling."""

    assignments_file = output_dir / "cluster_assignments.json"
    labels_file = output_dir / "cluster_labels.json"

    if not assignments_file.exists():
        print("No cluster assignments found. Run organize first.")
        return

    with open(assignments_file) as f:
        assignments = json.load(f)

    # Load existing labels
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)
    else:
        labels = {}

    print("Label your clusters (enter name, or 'skip' to skip, 'quit' to exit):\n")

    for cluster_name, images in sorted(assignments.items(), key=lambda x: -len(x[1])):
        if cluster_name in labels:
            print(f"{cluster_name}: already labeled as '{labels[cluster_name]}'")
            continue

        print(f"\n{cluster_name}: {len(images)} photos")
        print(f"  Sample: {images[0]}")

        name = input("  Name: ").strip()

        if name.lower() == 'quit':
            break
        elif name.lower() == 'skip' or not name:
            continue
        else:
            labels[cluster_name] = name

    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"\nLabels saved to {labels_file}")


def main():
    parser = argparse.ArgumentParser(description="Organize photos by face")
    parser.add_argument(
        "command",
        choices=["scan", "cluster", "label", "all"],
        help="Command to run"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Input directory with photos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Clustering threshold (lower = stricter, default 0.4)"
    )
    parser.add_argument(
        "--min-cluster",
        type=int,
        default=3,
        help="Minimum faces per cluster (default 3)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files to cluster folders"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from cache"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.command in ["scan", "all"]:
        if not args.input:
            print("--input required for scan")
            return

        app = load_insightface()
        cache = scan_photos(
            Path(args.input),
            app,
            resume=not args.no_resume,
        )

    if args.command in ["cluster", "all"]:
        cache_file = output_dir / "face_cache.json"
        if not cache_file.exists():
            print("No face cache found. Run 'scan' first.")
            return

        with open(cache_file) as f:
            cache = json.load(f)

        organize_by_clusters(
            cache,
            output_dir,
            threshold=args.threshold,
            min_cluster_size=args.min_cluster,
            copy_files=args.copy,
        )

    if args.command == "label":
        label_clusters(output_dir)


if __name__ == "__main__":
    main()
