# Cantankerous Kumquat

NSFW Vision Training Pipeline - Train Ministral 3 14B for improved image captioning with precise vocabulary.

## Overview

```
Download Datasets → Caption with Qwen3-VL → Prepare Data → Train Ministral 3
```

## Quick Start

### 1. Environment Setup

```bash
cd /home/lab/eager_beaver
source venv/bin/activate
huggingface-cli login
```

### 2. Download Datasets

```bash
python scripts/download_datasets.py
```

Downloads:
- `wallstoneai/civitai-top-nsfw-images-with-metadata` (~6K images)
- `zxbsmk/NSFW-T2I` (~38K images)

### 3. Caption Images

Run on GPU machine (3090 recommended):

```bash
# Caption all datasets
python scripts/caption_images.py --dataset all

# Or one at a time
python scripts/caption_images.py --dataset civitai
python scripts/caption_images.py --dataset nsfw_t2i
```

Supports resume - will skip already captioned images.

### 4. Prepare Training Data

```bash
python scripts/prepare_training_data.py --dataset all --merge
```

Creates conversation-format training data at `data/training_ready/`.

### 5. Upload to HuggingFace

```bash
python scripts/upload_dataset.py --dataset combined --repo-id YOUR_USERNAME/nsfw-captions
```

### 6. Train with HF Jobs

Use the HuggingFace Skills in Claude Code:

```
Fine-tune mistralai/Ministral-3-14B-Base-2512 on YOUR_USERNAME/nsfw-captions
for vision captioning using full fine-tuning on an A100.
```

## Customization

### Vocabulary Taxonomy

Edit `configs/taxonomy.json` to customize the vocabulary used during captioning.
The captioner will be prompted to use these specific terms.

Categories include:
- `clothing_lower` - underwear, shorts, skirts, pants
- `clothing_upper` - bras, tops, etc.
- `swimwear` - bikinis, swimsuits
- `lingerie` - stockings, garters, etc.
- `poses` - standing, sitting, etc.
- `expressions` - facial expressions
- `lighting` - lighting descriptions
- `settings` - environments

### Caption Quality

Adjust filtering in `scripts/prepare_training_data.py`:
- `min_caption_length` - minimum characters (default: 50)
- `max_caption_length` - maximum characters (default: 2000)

## Project Structure

```
eager_beaver/
├── configs/
│   └── taxonomy.json       # Vocabulary definitions
├── data/
│   ├── raw/               # Downloaded datasets
│   ├── processed/         # Captioned data
│   └── training_ready/    # Final training format
├── scripts/
│   ├── download_datasets.py
│   ├── caption_images.py
│   ├── prepare_training_data.py
│   └── upload_dataset.py
├── models/                # Local model cache
├── outputs/               # Training outputs
└── venv/                  # Python environment
```

## Hardware Requirements

| Task | GPU | VRAM | Notes |
|------|-----|------|-------|
| Captioning | RTX 3090 | 24GB | bf16, ~2 img/sec |
| Captioning | RTX 3060 | 12GB | 4-bit quantized |
| Training (QLoRA) | RTX 3090 | 24GB | Local option |
| Training (Full) | A100 80GB | 80GB | HF Jobs recommended |

## Next Steps

1. Review and refine taxonomy for your specific needs
2. Run captioning on a sample first to verify quality
3. Manually review some captions before full training
4. Consider iterative training - start small, evaluate, expand
