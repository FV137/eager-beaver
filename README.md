# Eager Beaver 🦫

**Complete toolkit for vision AI workflows** - from photo organization to custom model training.

## ✨ **NEW: Beautiful TUI Wizard**

**The easiest way to train LoRAs!** Guided workflow with model presets (SDXL, Flux.1, ZIT) and smart defaults.

```bash
python beaver.py
```

Features:
- 🎨 Beautiful terminal interface with progress tracking
- ⚙️ Model-specific presets (SDXL, Flux.1, ZIT, SD 1.5, SD 2.1)
- ⚠️ Quality warnings (shot distribution, dataset size)
- 💾 Session save/resume
- 🚀 Complete guided workflow in one command

**[Try it now: `python beaver.py`](#guided-wizard-recommended)**

---

## What's Inside

This monorepo contains four powerful tools that work great together or standalone:

### 🗂️ **FaceVault** - Photo Organization
Beautiful face organization for your photo library. Perfect for Christmas photos, family albums, or any collection with people.

```bash
python facevault.py scan ~/Pictures/Christmas2024
python facevault.py cluster --preview
python facevault.py label
python facevault.py export --all --format organized
```

**[📖 Full FaceVault Documentation →](FACEVAULT.md)**

### 🎨 **LoRA Prep** - Intelligent Dataset Preparation
Prepare LoRA training datasets with automatic shot classification (close/mid/far) and concept extraction.

```bash
python lora_prep.py prepare /path/to/photos \
  --name "Emma" \
  --facevault-cache outputs/facevault/face_cache.json \
  --captions outputs/processed/emma/captions.json
```

**[📖 Full LoRA Workflow Guide →](LORA_WORKFLOW.md)**

### 🚀 **LoRA Train** - Simplified Training Integration
Configure LoRA training with smart defaults. Generates configs for kohya_ss, diffusers, and more.

```bash
python lora_train.py train outputs/lora_datasets/emma --interactive
# Auto-calculates epochs, batch size, learning rate
# Generates ready-to-run training scripts
```

**[📖 Full LoRA Workflow Guide →](LORA_WORKFLOW.md)**

### 🏗️ **Vision Training Pipeline**
End-to-end pipeline for training vision models with custom datasets and precise vocabulary control.

```bash
# Pull datasets (HuggingFace, local, or organized with FaceVault)
python scripts/download_datasets.py

# Caption with vision models
python scripts/caption_images.py --dataset all

# Train your own vision model
# (Supports general datasets, not just NSFW)
```

---

## Quick Start

### Installation

```bash
# Clone the repo
git clone <repo-url> eager-beaver
cd eager-beaver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (for model downloads)
huggingface-cli login
```

### Guided Wizard (Recommended)

**The easiest way to get started!**

```bash
python beaver.py
```

Interactive menu:
1. **LoRA Training Pipeline** - Complete workflow with model presets
2. **Photo Organization** - Just organize photos (FaceVault only)
3. **Individual Tools** - Run tools manually

**Model Presets Included:**
- 🎨 **SDXL** - Standard Stable Diffusion XL (1024px, lr=1e-4, dim=32)
- ⚡ **Flux.1** - Black Forest Labs Flux (1024px, lr=4e-4, dim=64)
- 🎯 **ZIT (ZTSNR)** - Zero Terminal SNR (1024px, lr=8e-5, special SNR settings)
- 📸 **SD 1.5** - Classic Stable Diffusion (512px, lr=1e-4, dim=32)
- 🖼️ **SD 2.1** - SD 2.1 with v-param (768px, lr=1e-4, dim=32)
- ⚙️ **Custom** - Start with SDXL defaults, customize as needed

**Quality Warnings:**
The wizard shows yellow bubble warnings for:
- Low image count (<15 images)
- Unbalanced shot distribution (too many close-ups, not enough variety)
- Missing face classifications

---

### Manual Workflows (Advanced)

If you prefer running tools individually:

#### **Workflow A: Photo Organization Only**

Use FaceVault to organize your personal photo library:

```bash
# Scan and organize
python facevault.py scan ~/Pictures
python facevault.py cluster
python facevault.py label
python facevault.py export --all --format organized

# See stats
python facevault.py stats
```

#### **Workflow B: Vision Model Training**

Train a custom vision model on any dataset:

```bash
# Download datasets from HuggingFace
python scripts/download_datasets.py

# Caption images with Qwen3-VL
python scripts/caption_images.py --dataset all

# Prepare training data
python scripts/prepare_training_data.py --dataset all --merge

# Upload to HuggingFace
python scripts/upload_dataset.py --dataset combined --repo-id your-username/dataset-name

# Train via HuggingFace (or local)
# Fine-tune on your captioned dataset
```

#### **Workflow C: Personal LoRA Training (The "Both" Option)** ⭐

Organize your photos with FaceVault, then train personalized LoRA models with intelligent shot classification:

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

# 5. Configure training with smart defaults
python lora_train.py train outputs/lora_datasets/emma --interactive

# 6. Train LoRA (multiple options)
# - Edit and run: outputs/lora_models/emma/.../train.sh
# - Use kohya_ss: python train_network.py --config kohya_config.json
# - Use diffusers: See generated train.sh for examples
```

**What you get:**
- Images organized by shot type (close-up, mid-range, full-body)
- Extracted concepts from captions (poses, settings, clothing)
- Auto-calculated training parameters (epochs, batch size, learning rate)
- Ready-to-run configs for kohya_ss, diffusers, and more
- Complete pipeline: photos → trained LoRA

---

## Project Structure

```
eager-beaver/
├── beaver.py                   # ✨ TUI wizard (RECOMMENDED!)
├── facevault.py                # 🗂️  Face organization CLI
├── FACEVAULT.md               # Documentation for FaceVault
├── lora_prep.py               # 🎨 LoRA dataset preparation
├── lora_train.py              # 🚀 LoRA training integration
├── LORA_WORKFLOW.md           # LoRA training workflow guide
│
├── configs/
│   ├── taxonomy.json          # Vocabulary for captioning
│   └── model_presets.json     # Model-specific training presets
│
├── scripts/                    # 🏗️  Vision training pipeline
│   ├── download_datasets.py   # Pull datasets from HuggingFace
│   ├── caption_images.py      # Caption with Qwen3-VL (or custom model)
│   ├── prepare_training_data.py   # Convert to training format
│   ├── upload_dataset.py      # Upload to HuggingFace Hub
│   ├── test_captioner.py      # Test captioning models
│   └── organize_faces.py      # (Legacy - use facevault.py)
│
├── configs/
│   └── taxonomy.json          # Vocabulary for captioning (customizable)
│
├── outputs/
│   ├── facevault/             # FaceVault outputs
│   │   ├── organized/         # Organized by person
│   │   └── face_cache.json    # Face detection cache
│   ├── lora_datasets/         # LoRA training datasets
│   │   └── <person>/
│   │       ├── close/         # Close-up shots
│   │       ├── mid/           # Mid-range shots
│   │       ├── far/           # Full-body shots
│   │       ├── captions/      # Text captions
│   │       └── metadata.json  # Dataset info
│   ├── lora_models/           # LoRA training configs
│   │   └── <person>/
│   │       └── <timestamp>/
│   │           ├── kohya_config.json
│   │           ├── training_config.json
│   │           └── train.sh
│   └── faces/                 # Legacy organize_faces.py output
│
├── data/
│   ├── raw/                   # Downloaded datasets
│   ├── processed/             # Captioned datasets
│   └── training_ready/        # Final training format
│
├── requirements.txt           # All dependencies
└── README.md                  # This file
```

---

## Tools Deep Dive

### 🗂️ FaceVault

**Beautiful terminal-based face organization.**

#### Features:
- ✨ Simple 4-step workflow: scan → cluster → label → export
- 🎨 Rich terminal UI with progress bars and previews
- 🚀 Resume support (safe to interrupt anytime)
- 🎯 Quality filtering (min confidence, min face size)
- 📦 Multiple export formats (organized folders, LoRA-ready, JSON)
- 🔗 Symlink support (save disk space)

#### Commands:
```bash
facevault.py scan      # Extract faces from photos
facevault.py cluster   # Group similar faces
facevault.py label     # Name your clusters interactively
facevault.py export    # Export in various formats
facevault.py stats     # View statistics
```

**[📖 Full Documentation](FACEVAULT.md)** | **[Examples](FACEVAULT.md#complete-workflow-example)**

---

### 🎨 Vision Training Pipeline

**Complete pipeline for vision model fine-tuning.**

#### Dataset Support

**Not just NSFW!** This pipeline works with any image dataset:
- Personal photos (via FaceVault)
- Art collections
- Product photography
- Scientific imagery
- HuggingFace datasets
- Local directories

The included `taxonomy.json` is just one example - create your own for any domain.

#### Pipeline Steps

**1. Download Datasets**

```bash
# Download from HuggingFace
python scripts/download_datasets.py

# Datasets saved to data/raw/
```

Currently configured for:
- `wallstoneai/civitai-top-nsfw-images-with-metadata` (~6K images)
- `zxbsmk/NSFW-T2I` (~38K images)

*Edit the script to add your own HuggingFace datasets.*

**2. Caption Images**

```bash
# Caption all datasets
python scripts/caption_images.py --dataset all

# Caption specific dataset
python scripts/caption_images.py --dataset civitai

# Caption custom directory
python scripts/caption_images.py --dataset custom \
  --input /path/to/images \
  --output-name my_dataset

# Use different model
python scripts/caption_images.py --model "another/vision-model"
```

**Default model:** `Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5`
- Automatic VRAM detection (bf16 on 24GB+, 4-bit on smaller GPUs)
- Resume support (skips already captioned)
- Uses taxonomy from `configs/taxonomy.json`

**3. Prepare Training Data**

```bash
# Prepare all datasets
python scripts/prepare_training_data.py --dataset all --merge

# Prepare specific dataset
python scripts/prepare_training_data.py --dataset civitai
```

Converts captions to conversation format for vision model training.
Output: `data/training_ready/`

**4. Upload to HuggingFace**

```bash
python scripts/upload_dataset.py \
  --dataset combined \
  --repo-id your-username/dataset-name
```

**5. Train**

Use HuggingFace AutoTrain, local training scripts, or cloud platforms.

**Example for Ministral 3 14B:**
```python
# Fine-tune mistralai/Ministral-3-14B-Base-2512
# on your-username/dataset-name
# for vision captioning using full fine-tuning on A100
```

---

## Customization

### Vocabulary Taxonomy

Edit `configs/taxonomy.json` to customize captioning vocabulary for your domain.

**Current categories (customizable):**
- `clothing_lower`, `clothing_upper`, `clothing_full`
- `swimwear`, `lingerie`, `accessories`
- `body_descriptors`, `poses`, `expressions`
- `lighting`, `settings`

**Example custom taxonomy for art:**
```json
{
  "art_style": ["impressionist", "abstract", "realistic", ...],
  "medium": ["oil painting", "watercolor", "digital", ...],
  "composition": ["rule of thirds", "centered", "asymmetric", ...],
  "color_palette": ["warm", "cool", "monochrome", ...]
}
```

### Caption Quality Filters

Adjust in `scripts/prepare_training_data.py`:
- `min_caption_length` - Minimum characters (default: 50)
- `max_caption_length` - Maximum characters (default: 2000)

### Face Detection Quality

Adjust in FaceVault scan:
```bash
# High quality only
python facevault.py scan /path --min-score 0.7 --min-size 100

# Catch everything
python facevault.py scan /path --min-score 0.3 --min-size 30
```

---

## Hardware Requirements

| Task | GPU | VRAM | Notes |
|------|-----|------|-------|
| FaceVault (scan) | RTX 3060+ | 8GB+ | Or CPU (slower) |
| Captioning (Qwen3-VL) | RTX 3090 | 24GB | bf16, ~2 img/sec |
| Captioning (4-bit) | RTX 3060 | 12GB | Quantized |
| Vision Training (QLoRA) | RTX 3090 | 24GB | Local option |
| Vision Training (Full) | A100 | 80GB | HF AutoTrain recommended |

**Note:** FaceVault and captioning both support CPU fallback, but GPU is much faster.

---

## Use Cases

### Personal Photo Library
1. Use FaceVault to organize family photos
2. Export faces for each person
3. Train personalized LoRA models
4. Generate custom artwork of family members

### Content Creation
1. Organize stock photos by subject
2. Caption with custom taxonomy
3. Train specialized vision model
4. Generate captions for new content

### Research & Training
1. Collect domain-specific images
2. Caption with controlled vocabulary
3. Fine-tune vision models
4. Create specialized captioning tools

### Dataset Curation
1. Download raw datasets from HuggingFace
2. Filter and organize with FaceVault
3. Caption with quality control
4. Re-upload curated dataset

---

## Roadmap

### FaceVault
- [ ] Web UI for face review/labeling
- [ ] Advanced quality metrics (blur, occlusion)
- [ ] Similarity search
- [ ] Manual cluster merge/split
- [ ] Direct cloud uploads

### Vision Pipeline
- [ ] Universal dataset loader (HF, Kaggle, local)
- [ ] Multiple captioning model support
- [ ] Interactive caption editing/refinement
- [ ] Integrated LoRA training
- [ ] Quality scoring and filtering
- [ ] Multi-GPU support

### Integration
- [ ] One-command end-to-end workflow
- [ ] Preset configs for common use cases
- [ ] Automatic taxonomy generation
- [ ] Model evaluation tools

---

## Contributing

This is a personal toolkit, but contributions are welcome!

**Areas for improvement:**
- Additional export formats
- New captioning models
- Training integrations
- Documentation improvements
- Bug fixes

---

## License

MIT License - Free for personal and commercial use.

---

## Credits

**Tools & Libraries:**
- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
- [Qwen3-VL](https://huggingface.co/Qwen) - Vision-language model
- [Rich](https://github.com/Textualize/rich) - Terminal UI
- [Click](https://click.palletsprojects.com/) - CLI framework
- [HuggingFace](https://huggingface.co/) - Model hub & training

**Inspiration:**
- [p-e-w/heretic](https://github.com/p-e-w/heretic) - Simple, powerful CLI UX

---

**Built with ❤️ for the joy of creating personalized AI**

Perfect for organizing Christmas photos, training custom models, or building the next generation of vision AI! 🎄🦫
