# LoRA Training Workflow 🎨

**Complete pipeline for preparing personalized LoRA training datasets** with intelligent shot classification and concept extraction.

---

## Overview

The **Eager Beaver LoRA workflow** combines three powerful tools:

1. **FaceVault** - Organize photos by person
2. **Caption Images** - Generate detailed descriptions
3. **LoRA Prep** - Classify shots, extract concepts, organize dataset

The result: High-quality, well-organized datasets perfect for LoRA training.

---

## Quick Start

### End-to-End Pipeline

```bash
# 1. Organize your photo library
python facevault.py scan ~/Pictures/Family
python facevault.py cluster --threshold 0.35
python facevault.py label

# 2. Export person
python facevault.py export --person person_001 --format organized

# 3. Caption the images
python scripts/caption_images.py --dataset custom \
  --input outputs/facevault/organized/Emma \
  --output-name emma

# 4. Prepare LoRA dataset with shot classification
python lora_prep.py prepare outputs/facevault/organized/Emma \
  --name "Emma" \
  --facevault-cache outputs/facevault/face_cache.json \
  --captions outputs/processed/emma/captions.json \
  --taxonomy configs/taxonomy.json

# 5. Train LoRA (with your favorite trainer)
# Trigger word: emma
# Dataset: outputs/lora_datasets/emma/
```

**Done!** Your dataset is organized by shot type with rich metadata.

---

## The Three Shot Types

LoRA Prep automatically classifies images into three categories based on face size:

### 📷 **Close** - Headshots & Portraits
- Face takes up **>25%** of image
- Great for: Face details, expressions, close-up features
- Example: Professional headshot, selfie

### 📷 **Mid** - Upper Body & Waist-Up
- Face is **8-25%** of image
- Great for: Poses, clothing, upper body composition
- Example: Waist-up portrait, casual photo

### 📷 **Far** - Full Body & Environmental
- Face is **<8%** of image
- Great for: Full outfits, environment, wide shots
- Example: Full-body shot, action photo, landscape with person

---

## Dataset Structure

After running `lora_prep`, you'll have:

```
outputs/lora_datasets/emma/
├── close/                      # Close-up shots
│   ├── emma_close_0001.jpg
│   ├── emma_close_0002.jpg
│   └── ...
├── mid/                        # Mid-range shots
│   ├── emma_mid_0001.jpg
│   └── ...
├── far/                        # Full-body/distant shots
│   ├── emma_far_0001.jpg
│   └── ...
├── captions/                   # Text captions for each image
│   ├── emma_close_0001.txt
│   ├── emma_mid_0001.txt
│   └── ...
├── metadata.json              # Dataset overview
├── shots.json                 # Shot classification details
├── concepts.json              # Extracted concepts per image
└── preview_manifest.json      # For future preview UI
```

---

## Metadata Files

### **metadata.json**

High-level dataset information:

```json
{
  "person": "Emma",
  "trigger_word": "emma",
  "total_images": 47,
  "shot_distribution": {
    "close": 15,
    "mid": 20,
    "far": 12
  },
  "top_concepts": [
    {"concept": "smiling", "count": 35, "frequency": 0.74},
    {"concept": "outdoor", "count": 28, "frequency": 0.60},
    {"concept": "casual_clothing", "count": 22, "frequency": 0.47}
  ],
  "created_at": "2025-12-24T10:30:00"
}
```

### **shots.json**

Detailed shot classification for each image:

```json
{
  "emma_close_0001.jpg": {
    "shot_type": "close",
    "face_ratio": 0.42,
    "confidence": 0.89,
    "face_bbox": [120, 80, 450, 520]
  },
  "emma_mid_0001.jpg": {
    "shot_type": "mid",
    "face_ratio": 0.15,
    "confidence": 0.92,
    "face_bbox": [180, 100, 320, 280]
  }
}
```

### **concepts.json**

Extracted concepts and tags from captions:

```json
{
  "emma_close_0001.jpg": {
    "raw_caption": "Portrait of a young woman with blonde hair, smiling...",
    "concepts": ["portrait", "smiling", "blonde_hair", "outdoor", "natural_light"],
    "pose": "frontal",
    "setting": "outdoor",
    "lighting": "natural",
    "clothing": ["casual_top"],
    "attributes": ["smiling", "blonde_hair"]
  }
}
```

### **preview_manifest.json**

Consolidated data for preview UIs (future feature):

```json
{
  "dataset": {
    "name": "Emma",
    "trigger": "emma",
    "total_images": 47
  },
  "images": [
    {
      "filename": "emma_close_0001.jpg",
      "shot_type": "close",
      "shot_confidence": 0.89,
      "caption": "Portrait of a young woman...",
      "concepts": ["portrait", "smiling", "blonde_hair"],
      "pose": "frontal",
      "setting": "outdoor"
    }
  ]
}
```

---

## Command Reference

### `lora_prep prepare`

Prepare a LoRA training dataset with shot classification and concept extraction.

```bash
python lora_prep.py prepare INPUT_DIR --name NAME [OPTIONS]
```

**Required Arguments:**
- `INPUT_DIR` - Directory with images (from FaceVault or any source)
- `--name, -n NAME` - Person/concept name

**Options:**
- `--output, -o PATH` - Output directory (default: auto-generated)
- `--trigger, -t WORD` - Custom trigger word (default: auto from name)
- `--facevault-cache PATH` - FaceVault face_cache.json for shot classification
- `--captions PATH` - captions.json from caption_images.py
- `--taxonomy PATH` - taxonomy.json for concept extraction
- `--organize-by-shot` - Organize into close/mid/far folders (default: true)
- `--min-per-shot INT` - Minimum images per shot type (default: 3)
- `--symlink` - Create symlinks instead of copying files

**Examples:**

```bash
# Basic - just organize images
python lora_prep.py prepare ~/Pictures/Emma --name "Emma"

# Full integration with FaceVault + captions
python lora_prep.py prepare outputs/facevault/organized/Emma \
  --name "Emma" \
  --facevault-cache outputs/facevault/face_cache.json \
  --captions outputs/processed/emma/captions.json \
  --taxonomy configs/taxonomy.json

# Custom trigger word
python lora_prep.py prepare /path/to/images \
  --name "Emma Watson" \
  --trigger "ewatson"

# Use symlinks to save disk space
python lora_prep.py prepare /path/to/images \
  --name "Emma" \
  --symlink
```

---

## Integration Workflows

### Workflow 1: From Scratch (Complete Pipeline)

```bash
# Scan photo library
python facevault.py scan ~/Pictures/Family \
  --min-score 0.6 \
  --min-size 80

# Cluster faces
python facevault.py cluster \
  --threshold 0.35 \
  --min-cluster 5 \
  --preview

# Label people
python facevault.py label --show-paths

# Export person (Emma = person_001)
python facevault.py export \
  --person person_001 \
  --format organized

# Caption images
python scripts/caption_images.py \
  --dataset custom \
  --input outputs/facevault/organized/Emma \
  --output-name emma

# Prepare LoRA dataset
python lora_prep.py prepare outputs/facevault/organized/Emma \
  --name "Emma" \
  --facevault-cache outputs/facevault/face_cache.json \
  --captions outputs/processed/emma/captions.json \
  --taxonomy configs/taxonomy.json

# Train LoRA
# Use outputs/lora_datasets/emma/ with trigger word "emma"
```

### Workflow 2: Quick Prep (Already Have Images)

```bash
# Just caption and prep
python scripts/caption_images.py \
  --dataset custom \
  --input /path/to/emma/photos \
  --output-name emma

python lora_prep.py prepare /path/to/emma/photos \
  --name "Emma" \
  --captions outputs/processed/emma/captions.json
```

### Workflow 3: Multiple People

```bash
# Export all from FaceVault
python facevault.py export --all --format organized

# Process each person
for person in outputs/facevault/organized/*/; do
  name=$(basename "$person")

  # Caption
  python scripts/caption_images.py \
    --dataset custom \
    --input "$person" \
    --output-name "$name"

  # Prep LoRA
  python lora_prep.py prepare "$person" \
    --name "$name" \
    --facevault-cache outputs/facevault/face_cache.json \
    --captions "outputs/processed/$name/captions.json" \
    --taxonomy configs/taxonomy.json
done
```

---

## Shot Distribution Guidelines

For best LoRA training results:

### Recommended Distribution

- **Close shots**: 30-40% (face details, expressions)
- **Mid shots**: 40-50% (most versatile, common use case)
- **Far shots**: 20-30% (full body, variety)

### Minimum Requirements

- **At least 3 images per shot type** (configurable with `--min-per-shot`)
- **Total 15+ images** recommended minimum
- **30+ images** for better results
- **50+ images** for excellent quality

### Common Issues

**Too many close shots (>60%)**
- LoRA may struggle with full-body generation
- Add more mid/far shots or accept limited versatility

**Too many far shots (>50%)**
- Face consistency may suffer
- Add more close/mid shots for better face training

**Missing shot types**
- LoRA Prep will warn you
- Consider adding more photos to balance dataset

---

## Concept Extraction

LoRA Prep automatically extracts structured concepts from captions:

### Extracted Categories

- **Pose**: standing, sitting, lying, kneeling, etc.
- **Setting**: indoor, outdoor, studio
- **Lighting**: natural, artificial, soft, dramatic
- **Clothing**: Extracted from taxonomy
- **Attributes**: Facial features, expressions, hair color, etc.

### Using Concepts in Training

The extracted concepts can help you:

1. **Understand your dataset** - See what's well-represented
2. **Plan training tags** - Know which concepts to emphasize
3. **Quality control** - Ensure diversity in poses/settings
4. **Future UI** - Filter/browse by concept

### Top Concepts

Check `metadata.json` for the most common concepts in your dataset:

```json
{
  "top_concepts": [
    {"concept": "smiling", "count": 35, "frequency": 0.74},
    {"concept": "outdoor", "count": 28, "frequency": 0.60},
    {"concept": "blonde_hair", "count": 25, "frequency": 0.53}
  ]
}
```

High frequency (>70%) means the concept is very common - LoRA will learn it strongly.
Low frequency (<20%) means the concept is rare - may need more examples.

---

## Custom Taxonomies

Create domain-specific taxonomies for better concept extraction:

### Example: Cosplay Dataset

```json
{
  "costume_type": [
    "anime character",
    "video game character",
    "movie character",
    "original design"
  ],
  "costume_elements": [
    "wig", "armor", "prop weapon",
    "cape", "mask", "accessories"
  ],
  "setting": [
    "convention", "photo studio",
    "outdoor location", "green screen"
  ]
}
```

Save as `configs/taxonomy_cosplay.json` and use:

```bash
python lora_prep.py prepare /path/to/cosplay/photos \
  --name "Character Name" \
  --taxonomy configs/taxonomy_cosplay.json
```

---

## Future Features (Preview UI)

The `preview_manifest.json` is designed for a future web UI where you can:

- 🖼️ **Browse images** by shot type
- 🏷️ **Filter by concepts** (show all "outdoor" or "smiling")
- ✏️ **Edit classifications** (manually adjust shot types)
- 🔀 **Rebalance dataset** (move images between shot types)
- 📊 **Visualize distribution** (charts, graphs)
- 🎯 **Tag for training** (mark images for specific training emphasis)

Stay tuned! 🚀

---

## Training Tips

### Trigger Word Usage

The trigger word (e.g., "emma") should be:
- **Unique** - Not a common word
- **Short** - 1-2 words max
- **Memorable** - Easy to type

**Good triggers:**
- `emma` (person name)
- `ewatson` (initials + last name)
- `jdoe` (initials)
- `char_name` (character)

**Bad triggers:**
- `woman` (too generic)
- `the_person_emma` (too long)
- `photo` (conflicts with common terms)

### Using Shot Types in Training

Some LoRA trainers support per-folder tags:

```
close/    → Tag: "emma headshot"
mid/      → Tag: "emma portrait"
far/      → Tag: "emma full body"
```

This helps the model learn shot-specific features.

### Caption Integration

Each image has a `.txt` caption in the `captions/` folder. Use these with your LoRA trainer for better results.

**Example structure for kohya_ss:**
```
dataset/
├── emma_close_0001.jpg
├── emma_close_0001.txt    ← "portrait of emma, smiling..."
├── emma_mid_0001.jpg
├── emma_mid_0001.txt      ← "emma standing outdoor..."
```

---

## Troubleshooting

### "No face cache found" warning

Shot classification requires FaceVault's face detection data. Either:

1. Use FaceVault first:
   ```bash
   python facevault.py scan /path/to/photos
   ```

2. Or provide the cache:
   ```bash
   python lora_prep.py prepare /path/to/photos \
     --facevault-cache outputs/facevault/face_cache.json
   ```

3. Or skip shot classification (all images → `mid/`)

### High "unknown" classification

If many images are classified as "unknown":
- Ensure `--facevault-cache` points to correct cache file
- Check that cache includes the images you're processing
- Verify face detection worked (run FaceVault scan first)

### Unbalanced shot distribution

If you see warnings about too few images per shot type:
- **Add more photos** to underrepresented categories
- **Lower `--min-per-shot`** if you have a small dataset
- **Accept the imbalance** (LoRA will just be less versatile)

### Missing captions

If no captions are generated:
- Ensure `--captions` points to the right `captions.json`
- Check that image filenames match between caption file and images
- Run `caption_images.py` first if needed

---

## Advanced Usage

### Batch Processing Multiple People

```bash
#!/bin/bash
# prep_all_loras.sh

for person_dir in outputs/facevault/organized/*/; do
  person=$(basename "$person_dir")

  echo "Processing $person..."

  python lora_prep.py prepare "$person_dir" \
    --name "$person" \
    --facevault-cache outputs/facevault/face_cache.json \
    --captions "outputs/processed/$person/captions.json" \
    --taxonomy configs/taxonomy.json \
    --symlink
done

echo "All LoRA datasets prepared!"
```

### Custom Output Organization

```bash
# Organize by person in a specific directory
python lora_prep.py prepare /path/to/photos \
  --name "Emma" \
  --output /training/datasets/emma \
  --trigger "emw"
```

### Quality Control

After preparing, check the metadata:

```bash
# View shot distribution
jq '.shot_distribution' outputs/lora_datasets/emma/metadata.json

# View top concepts
jq '.top_concepts' outputs/lora_datasets/emma/metadata.json

# Find low-confidence shots
jq '[.[] | select(.confidence < 0.5)]' outputs/lora_datasets/emma/shots.json
```

---

## Integration with Training Tools

### Kohya_ss

```bash
# Dataset structure for kohya
# Use the organized folders directly
# Enable caption files in kohya config
```

### ComfyUI

```bash
# Point to outputs/lora_datasets/emma/
# Use metadata.json for trigger word
```

### Auto1111

```bash
# Copy images + captions
# Use trigger word from metadata.json
```

---

## Complete Example: Family Christmas Photos

```bash
# 1. Scan your Christmas photos
python facevault.py scan ~/Pictures/Christmas2024 \
  --min-score 0.6

# 2. Cluster faces
python facevault.py cluster --threshold 0.35 --preview

# 3. Label family members
python facevault.py label
# → person_001 = "Dad"
# → person_002 = "Mom"
# → person_003 = "Emma"
# → person_004 = "Jake"

# 4. Export everyone
python facevault.py export --all --format organized

# 5. Caption all photos
for person in outputs/facevault/organized/*/; do
  name=$(basename "$person")
  python scripts/caption_images.py \
    --dataset custom \
    --input "$person" \
    --output-name "$name"
done

# 6. Prepare LoRA datasets for everyone
for person in outputs/facevault/organized/*/; do
  name=$(basename "$person")
  python lora_prep.py prepare "$person" \
    --name "$name" \
    --facevault-cache outputs/facevault/face_cache.json \
    --captions "outputs/processed/$name/captions.json"
done

# 7. Train LoRAs!
# outputs/lora_datasets/Dad/     (trigger: "dad")
# outputs/lora_datasets/Mom/     (trigger: "mom")
# outputs/lora_datasets/Emma/    (trigger: "emma")
# outputs/lora_datasets/Jake/    (trigger: "jake")

# 8. Generate family portrait with AI
# Prompt: "family photo of dad, mom, emma, and jake at christmas"
```

---

## FAQ

**Q: Do I need FaceVault to use LoRA Prep?**

No! You can use any image directory. However, FaceVault provides better shot classification.

**Q: Can I use this for non-person LoRAs (styles, objects)?**

Yes! The shot classification won't apply, but concept extraction still works.

**Q: How many images do I need?**

Minimum 15, recommended 30+, ideal 50+ for high quality results.

**Q: Can I manually adjust shot classifications?**

Not yet - this is a future feature with the preview UI. For now, you can manually move files between folders.

**Q: Do I need captions?**

No, but they significantly improve training quality and enable concept extraction.

**Q: Can I train on specific shot types only?**

Yes! Just use one folder (e.g., `close/` only) for your training.

---

**Built with ❤️ for creating personalized AI**

Perfect for training LoRAs of yourself, family, friends, characters, or anything else! 🎄🎨
