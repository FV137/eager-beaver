# 🔄 Bootstrap Loop - Self-Empowering Training

Complete automation of the synthetic data generation and training improvement cycle.

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                  SELF-EMPOWERING LOOP                       │
└─────────────────────────────────────────────────────────────┘

  Real Photos (50-100 images)
         ↓
  1. FaceVault Scan/Cluster/Label
         ↓
  2. Train Initial LoRA → [LoRA v1]
         ↓
  ┌──────────────────────── BOOTSTRAP LOOP ─────────────────────┐
  │                                                              │
  │  3. Gap Analysis                                             │
  │     └─→ "Need 15 profile shots, 10 hands visible"           │
  │                                                              │
  │  4. Generate Synthetic Images                                │
  │     └─→ Use [LoRA v1] + base model                          │
  │         Create targeted images from gaps                     │
  │                                                              │
  │  5. Validate Synthetic                                       │
  │     └─→ YOLO checks: correct angle? good quality?           │
  │         Filter: approved (80%) vs rejected (20%)             │
  │                                                              │
  │  6. Merge Approved into Dataset                              │
  │     └─→ Real (100) + Synthetic (40) = 140 images            │
  │                                                              │
  │  7. Retrain LoRA → [LoRA v2]                                │
  │     └─→ Better balanced, more diverse                        │
  │                                                              │
  │  8. Check Convergence                                        │
  │     └─→ If improvement < 5%: DONE                            │
  │     └─→ Else: REPEAT (use LoRA v2 for generation)          │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
         ↓
  Final LoRA: Perfect digital twin for
  - D-ID avatars
  - MetaHuman
  - VTuber rigs
  - Any character LoRA use case
```

## Why This Matters

**The Problem**: Training a good character LoRA needs 50-100 diverse, high-quality images. Most people don't have:
- Enough photos
- Balanced angles (front/profile/three-quarter)
- Variety in expressions, poses, settings
- Consistent quality

**The Solution**: Start with what you have, generate what you need, improve iteratively.

**The Magic**: Each iteration improves the LoRA, which improves generation quality, which improves the next LoRA. Positive feedback loop until convergence.

## Components

### 1. Gap Analysis (`gap_analysis.py`)

**What it does**: Analyzes your dataset using specialized YOLO models to find weaknesses.

**YOLO Models Used**:
- `yolov5l-face.pt` - Face detection and angle estimation
- `hand_yolov8n.pt` - Hand visibility detection
- `eyes_yolov.pt` - Eyes visibility detection

**Output**:
```json
{
  "recommendations": [
    {
      "type": "face_angle",
      "target": "profile",
      "count": 15,
      "reason": "Profile shots are 8.2% (target: 20%)"
    },
    {
      "type": "composition",
      "target": "hands_visible",
      "count": 10,
      "reason": "Hands visible in 12.0% (target: 30%)"
    }
  ]
}
```

**Usage**:
```bash
python gap_analysis.py outputs/facevault/organized/person \
  --models-dir models \
  --output gaps.json
```

### 2. Synthetic Generation (`generate_synthetic.py`)

**What it does**: Creates targeted synthetic images using diffusers pipeline (no ComfyUI needed!).

**How it works**:
1. Loads base model (SDXL/Flux/ZIT)
2. Loads your trained LoRA
3. Builds structured prompts from gap recommendations
4. Generates batch with variation
5. Saves with complete metadata

**Prompt Templates**:
- **Face Angles**: front, profile, three-quarter specific prompts
- **Expressions**: neutral, smile, serious, laughing
- **Settings**: indoor, outdoor, studio, casual
- **Quality**: Professional photography keywords

**Usage**:
```bash
python generate_synthetic.py gaps.json \
  --base-model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora outputs/lora_models/person/lora.safetensors \
  --person "ohwx person" \
  --output outputs/synthetic/person \
  --num-per-gap 5 \
  --steps 30 \
  --guidance 7.5
```

**For ZIT (better character LoRAs)**:
```bash
python generate_synthetic.py gaps.json \
  --base-model cagliostrolab/animagine-xl-3.1 \
  --lora lora.safetensors \
  --person "1girl, ohwx" \
  --output outputs/synthetic/person
```

### 3. Validation (`validate_synthetic.py`)

**What it does**: Quality-checks generated images using same YOLO models as gap analysis.

**Validation Checks**:
- **Target Achievement**: Did we get the requested angle/feature?
- **Blur Score**: Laplacian variance >100
- **Brightness**: 30-225 range (not too dark/bright)
- **Face Confidence**: >0.5 detection score

**Auto-Filtering**:
- Approved → Ready for training
- Rejected → Failed quality checks

**Usage**:
```bash
python validate_synthetic.py outputs/synthetic/person \
  --models-dir models \
  --approved outputs/curated/person \
  --rejected outputs/rejected/person
```

**Output**:
```
Validation Complete
─────────────────────
Approved: 42 (84.0%)
Rejected: 8
Avg Quality: 0.82
```

### 4. Bootstrap Loop (`bootstrap_loop.py`)

**What it does**: Orchestrates the complete cycle automatically.

**Configuration** (`bootstrap_config.json`):
```json
{
  "person_name": "alice",
  "person_trigger": "ohwx alice",
  "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
  "dataset_dir": "outputs/facevault/organized/alice",
  "models_dir": "models",
  "output_dir": "outputs/bootstrap",
  "lora_output_dir": "outputs/lora_models/alice",
  "max_iterations": 5,
  "convergence_threshold": 0.05,
  "generation": {
    "num_per_gap": 5,
    "num_inference_steps": 30,
    "guidance_scale": 7.5
  },
  "validation": {
    "auto_filter": true
  },
  "training": {
    "preset": "sdxl",
    "epochs": 10
  }
}
```

**Usage**:
```bash
python bootstrap_loop.py --config bootstrap_config.json
```

**What happens**:
```
Iteration 1:
  ✓ Gap Analysis: 3 gaps found
  ✓ Generated: 15 images
  ✓ Validation: 12 approved (80%)
  ✓ Merged: 12 images into dataset
  ✓ Training: LoRA v2 complete

Iteration 2:
  ✓ Gap Analysis: 2 gaps found (improving!)
  ✓ Generated: 10 images
  ✓ Validation: 9 approved (90%)  ← Better quality
  ✓ Merged: 9 images
  ✓ Training: LoRA v3 complete

Iteration 3:
  ✓ Gap Analysis: 1 gap found
  ✓ Generated: 5 images
  ✓ Validation: 5 approved (100%)  ← Excellent!
  ✓ Improvement: 4.2% (below 5% threshold)

✓ Convergence Achieved!
```

## Complete Workflow

### Initial Setup

1. **Organize Photos with FaceVault**:
```bash
python beaver.py  # Choose workflow #1: LoRA Training Pipeline
# This runs: scan → cluster → dedup → label → caption → prep → train
```

2. **Install YOLO Models**:
```bash
mkdir models
# Place your .pt files in models/
#   - yolov5l-face.pt
#   - hand_yolov8n.pt
#   - eyes_yolov.pt (optional)
```

3. **Check PKL Safety** (if using .pkl YOLO models):
```bash
python inspect_pkl.py models/*.pkl --convert
# Converts safe .pkl files to .safetensors
```

### Running the Loop

**Option A: Fully Automated**
```bash
python bootstrap_loop.py --config my_config.json
# Sit back and watch the magic!
```

**Option B: Manual Control** (step-by-step)
```bash
# Iteration 1
python gap_analysis.py outputs/facevault/organized/alice \
  --output gaps_1.json

python generate_synthetic.py gaps_1.json \
  --base-model stabilityai/stable-diffusion-xl-base-1.0 \
  --lora lora_v1.safetensors \
  --person "ohwx alice" \
  --output synthetic_1/

python validate_synthetic.py synthetic_1/ \
  --approved approved_1/ \
  --rejected rejected_1/

# Manually merge approved_1/ into dataset
# Retrain LoRA → lora_v2.safetensors

# Iteration 2
python gap_analysis.py outputs/facevault/organized/alice \
  --output gaps_2.json

python generate_synthetic.py gaps_2.json \
  --lora lora_v2.safetensors \
  ...
# Repeat until satisfied
```

## Metrics to Watch

**Per Iteration**:
- **Gap Count**: Should decrease (dataset getting more balanced)
- **Approval Rate**: Should increase (LoRA quality improving)
- **Quality Score**: Should increase (better generations)

**Convergence Indicators**:
- Approval rate plateaus at 85-95%
- Gap count stabilizes at 0-2 gaps
- Quality score >0.85 consistently
- Improvement <5% between iterations

## Tips & Best Practices

### Starting Dataset
- **Minimum**: 50 images
- **Ideal**: 100+ images
- **Already deduplicated**: Run FaceVault dedup first

### Generation Settings
- **SDXL**: Standard, reliable
  - Steps: 30-40
  - Guidance: 7.0-8.0
- **Flux.1**: High quality, slower
  - Steps: 20-30 (fewer needed)
  - Guidance: 3.5-5.0
- **ZIT**: Best for character LoRAs
  - Steps: 25-35
  - Guidance: 6.0-7.5

### Iteration Count
- **3-5 iterations**: Usually sufficient
- **More than 7**: Diminishing returns likely
- **Early convergence**: Your initial dataset was already good!

### When to Stop
1. Approval rate >90% for 2 consecutive iterations
2. Gap count = 0 (perfect balance)
3. Visual quality meets your needs
4. Improvement <2% (stricter than default 5%)

## Troubleshooting

### Low Approval Rates (<60%)
- LoRA might be undertrained
- Base model mismatch (check trigger words)
- Generation parameters too aggressive
- **Fix**: Lower guidance scale, increase steps

### No Gaps Detected
- YOLO models not loaded properly
- Dataset too small for meaningful analysis
- **Fix**: Check models/ directory, verify .pt files

### OOM Errors
- GPU memory insufficient
- **Fix**:
  - Reduce batch size
  - Use `torch.float16`
  - Enable `vae_slicing`
  - Lower resolution

### Poor Generation Quality
- LoRA weight too high/low
- Negative prompt insufficient
- **Fix**: Adjust LoRA scale, enhance negative prompt

## Advanced: Custom Gap Types

You can extend gap_analysis.py for custom detection:

```python
# Add to gap_analysis.py

def analyze_lighting(image_path: str, model) -> Dict:
    """Detect lighting conditions."""
    # Your custom YOLO model here
    ...

# Add to recommendations
if dark_pct < 20:
    recommendations.append({
        "type": "lighting",
        "target": "low_key",
        "count": needed,
        "reason": "Only 8% low-key lighting shots"
    })
```

Then in generate_synthetic.py:
```python
# Add to prompt templates
LIGHTING_PROMPTS = {
    "low_key": "dramatic lighting, low key, moody atmosphere",
    "high_key": "bright lighting, high key, airy atmosphere",
    ...
}
```

## Output Structure

```
outputs/bootstrap/
├── iteration_history.json       # Complete metrics log
├── gaps_iter1.json              # Gap analysis iteration 1
├── gaps_iter2.json
├── synthetic_iter1/
│   ├── synthetic_face_angle_profile_001_20241226_143022.png
│   ├── synthetic_face_angle_profile_002_20241226_143035.png
│   ├── ...
│   └── generation_metadata.json
├── approved_iter1/              # Passed validation
├── rejected_iter1/              # Failed validation
├── synthetic_iter2/
├── approved_iter2/
└── ...
```

## Integration with External Tools

### D-ID
```bash
# After bootstrap convergence
python export_for_did.py \
  --dataset outputs/facevault/organized/alice \
  --output alice_did_package/
```

### MetaHuman
```bash
# Export face sheets
python export_facesheet.py \
  --dataset outputs/facevault/organized/alice \
  --angles front,profile_left,profile_right,three_quarter \
  --output alice_metahuman.jpg
```

### VTuber (Live2D)
```bash
# Export expression set
python export_expressions.py \
  --dataset outputs/facevault/organized/alice \
  --expressions neutral,smile,surprised,angry \
  --output alice_live2d/
```

## Citation & Credits

This bootstrap loop concept is inspired by:
- Self-training techniques in semi-supervised learning
- Curriculum learning (start simple, add complexity)
- Active learning (identify and fill gaps)
- LoRA training best practices from the community

YOLO models from:
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv8](https://github.com/ultralytics/ultralytics)

Diffusion models:
- [Stability AI (SDXL)](https://stability.ai/)
- [Black Forest Labs (Flux)](https://blackforestlabs.ai/)
- [Cagliostro Lab (ZIT/Animagine)](https://huggingface.co/cagliostrolab)

---

**Built with ❤️ during 5 days of double tokens!**

For questions, issues, or improvements: Create an issue on the repo.
