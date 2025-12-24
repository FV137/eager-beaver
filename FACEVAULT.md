# FaceVault 🗂️

**Beautiful face organization for your photo library.**

Perfect for organizing Christmas photos, family albums, or any photo collection with people. Fast, intuitive, and terminal-based.

## Features

✨ **Simple Workflow** - Four commands: scan → cluster → label → export
🎨 **Beautiful UI** - Rich terminal interface with progress bars and tables
🚀 **Resume Support** - Interrupted? Just run again, it picks up where you left off
🎯 **Quality Filtering** - Configurable face detection thresholds
📦 **Multiple Export Formats** - Organized folders, LoRA-ready datasets, or JSON manifests
🔗 **Symlink Support** - Save disk space with symlinks instead of copies

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python facevault.py --help
```

### Basic Usage

```bash
# 1. Scan your photos
python facevault.py scan /path/to/photos

# 2. Cluster faces into people
python facevault.py cluster --preview

# 3. Label your clusters
python facevault.py label

# 4. Export organized results
python facevault.py export --all --format organized
```

## Commands

### `scan` - Extract Faces

Scan a directory of photos and extract face embeddings using InsightFace.

```bash
python facevault.py scan /path/to/photos [OPTIONS]
```

**Options:**
- `--output, -o PATH` - Output directory (default: `outputs/facevault`)
- `--min-score FLOAT` - Minimum face detection confidence 0-1 (default: 0.5)
- `--min-size INT` - Minimum face size in pixels (default: 50)
- `--no-resume` - Start fresh, ignore existing cache

**Example:**
```bash
# Scan with strict quality filters
python facevault.py scan ~/Pictures/Christmas2024 \
  --min-score 0.7 \
  --min-size 80
```

**Output:**
- Creates `outputs/facevault/face_cache.json` with all detected faces
- Resumes automatically if interrupted

---

### `cluster` - Group Similar Faces

Cluster detected faces into groups of people using DBSCAN with cosine similarity.

```bash
python facevault.py cluster [OPTIONS]
```

**Options:**
- `--output, -o PATH` - Output directory (default: `outputs/facevault`)
- `--threshold, -t FLOAT` - Clustering threshold (default: 0.4)
  - **Lower = stricter** (same person only)
  - **Higher = looser** (may merge different people)
- `--min-cluster INT` - Minimum faces per cluster (default: 3)
- `--preview` - Show preview before saving

**Example:**
```bash
# Strict clustering, show preview
python facevault.py cluster --threshold 0.35 --preview

# Only keep people with 10+ photos
python facevault.py cluster --min-cluster 10
```

**Output:**
- Creates `outputs/facevault/cluster_assignments.json`
- Shows preview table with cluster sizes
- Statistics on clustered vs unclustered faces

---

### `label` - Name Your Clusters

Interactive labeling to give meaningful names to face clusters.

```bash
python facevault.py label [OPTIONS]
```

**Options:**
- `--output, -o PATH` - Output directory (default: `outputs/facevault`)
- `--show-paths` - Display sample file paths during labeling

**Example:**
```bash
python facevault.py label --show-paths
```

**Interactive prompts:**
```
┌─ person_001 ─────────────────┐
│ Faces: 47                    │
│ Avg Score: 0.89              │
│ Sample 1: IMG_1234.jpg       │
└──────────────────────────────┘

Name this person (or 'skip' to skip, 'done' to finish): Emma
✓ Labeled as 'Emma'
```

**Output:**
- Creates `outputs/facevault/cluster_labels.json`
- Saves incrementally (safe to quit anytime)

---

### `export` - Organize Files

Export organized faces in various formats.

```bash
python facevault.py export [OPTIONS]
```

**Options:**
- `--output, -o PATH` - Output directory (default: `outputs/facevault`)
- `--format, -f FORMAT` - Export format: `organized`, `lora`, or `json`
- `--person NAME` - Export specific cluster only
- `--symlink` - Create symlinks instead of copying files
- `--all` - Export all clusters

**Formats:**

#### 1. **organized** - Simple folders by person
```
outputs/facevault/organized/
├── Emma/
│   ├── IMG_1234.jpg
│   ├── IMG_1567.jpg
│   └── ...
├── Dad/
└── Mom/
```

#### 2. **lora** - LoRA training ready
```
outputs/facevault/lora_ready/
├── Emma/
│   ├── emma_0001.jpg
│   ├── emma_0002.jpg
│   └── metadata.json
└── Dad/
    ├── dad_0001.jpg
    └── metadata.json
```

Each `metadata.json` contains:
```json
{
  "person": "Emma",
  "trigger_word": "emma",
  "image_count": 47,
  "avg_detection_score": 0.89,
  "created_at": "2025-12-24T10:30:00"
}
```

#### 3. **json** - Full manifest
Creates `manifest.json` with all cluster data for custom processing.

**Examples:**
```bash
# Export all as organized folders with symlinks
python facevault.py export --all --format organized --symlink

# Export just Emma for LoRA training
python facevault.py export --person person_001 --format lora

# Export everything as JSON manifest
python facevault.py export --all --format json
```

---

### `stats` - View Statistics

Show statistics about your scanned and clustered faces.

```bash
python facevault.py stats
```

**Example output:**
```
╭─ Scan Results ──────────────────╮
│ 📸 Images scanned    2,847      │
│ 👤 Faces detected    3,124      │
│ 📊 Avg faces/image   1.10       │
│ ⚠️  Errors           3          │
╰─────────────────────────────────╯

╭─ Cluster Results ───────────────╮
│ 👥 Clusters found    12         │
│ 👤 Clustered faces   2,891      │
│ ⭐ Largest cluster   487 faces  │
│ 📊 Avg cluster size  240.9      │
╰─────────────────────────────────╯

✓ 8 clusters labeled
```

## Complete Workflow Example

```bash
# Scan Christmas photos
python facevault.py scan ~/Pictures/Christmas2024

# Cluster with preview
python facevault.py cluster --threshold 0.4 --preview

# Label the people
python facevault.py label

# Export for LoRA training (symlinks to save space)
python facevault.py export --all --format lora --symlink

# Check your stats
python facevault.py stats
```

## Tips & Best Practices

### Clustering Threshold

Finding the right threshold for your photos:

- **0.3-0.35**: Very strict (only very similar faces)
- **0.4**: Recommended default (good balance)
- **0.45-0.5**: Looser (may merge similar-looking people)

Start with 0.4 and adjust if you see:
- **People split across clusters**: Increase threshold (0.45)
- **Different people merged**: Decrease threshold (0.35)

### Quality Filtering

Adjust `--min-score` and `--min-size` during scanning:

- **High quality only**: `--min-score 0.7 --min-size 100`
- **Catch everything**: `--min-score 0.3 --min-size 30`
- **Balanced (default)**: `--min-score 0.5 --min-size 50`

### Disk Space

Use `--symlink` when exporting to save disk space:

```bash
# Copy files (uses 2x disk space)
python facevault.py export --all --format organized

# Symlink files (minimal disk usage)
python facevault.py export --all --format organized --symlink
```

### Resume Support

All commands support resuming:

- **scan**: Skips already processed images
- **cluster**: Can re-cluster with different thresholds
- **label**: Only shows unlabeled clusters

Safe to Ctrl+C and restart anytime!

## LoRA Training Integration

After exporting with `--format lora`, your datasets are ready for training:

```bash
# Export faces
python facevault.py export --person person_001 --format lora

# Now in outputs/facevault/lora_ready/Emma/
# - emma_0001.jpg, emma_0002.jpg, ...
# - metadata.json (includes trigger word)

# Use with your favorite LoRA trainer
# The trigger word from metadata.json: "emma"
```

## Architecture

### Face Detection
- **Model**: InsightFace (buffalo_l)
- **Embeddings**: 512-dimensional face vectors
- **Detection**: Bounding boxes, confidence scores, age/gender estimates

### Clustering
- **Algorithm**: DBSCAN with cosine distance
- **Similarity**: Normalized embedding vectors
- **Parameters**: Configurable epsilon (threshold) and min_samples

### Storage
- **face_cache.json**: All detected faces with embeddings
- **cluster_assignments.json**: Cluster-to-image mappings
- **cluster_labels.json**: User-defined cluster names

## Troubleshooting

### "No module named 'click'"
```bash
pip install -r requirements.txt
```

### "No face cache found"
Run `scan` before other commands:
```bash
python facevault.py scan /path/to/photos
```

### Faces not clustering well
Try adjusting the threshold:
```bash
# More strict
python facevault.py cluster --threshold 0.35

# More loose
python facevault.py cluster --threshold 0.45
```

### GPU errors
FaceVault will automatically fall back to CPU if CUDA is unavailable. For better performance, ensure:
- CUDA toolkit installed
- `onnxruntime-gpu` working
- InsightFace can access GPU

## File Formats

All output files use JSON for easy integration:

**face_cache.json:**
```json
{
  "/path/to/image.jpg": {
    "faces": [
      {
        "embedding": [...],
        "bbox": [x1, y1, x2, y2],
        "det_score": 0.89,
        "size": [120, 140],
        "age": 28,
        "gender": "F"
      }
    ],
    "count": 1,
    "scanned_at": "2025-12-24T10:00:00"
  }
}
```

**cluster_assignments.json:**
```json
{
  "person_001": {
    "images": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
    "count": 47,
    "avg_score": 0.89
  }
}
```

**cluster_labels.json:**
```json
{
  "person_001": "Emma",
  "person_002": "Dad"
}
```

## Future Features (Coming Soon)

- 🌐 Web UI for easier face review/labeling
- 📊 Advanced quality metrics (blur detection, occlusion)
- 🔍 Similarity search (find all photos of a specific person)
- 🎯 Manual merge/split clusters
- 🖼️ Thumbnail generation for faster previews
- 📤 Direct upload to cloud storage

---

**Created with ❤️ for organizing your photo memories**

Perfect for Christmas photos, family albums, event photography, or any collection with people!
