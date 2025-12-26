#!/usr/bin/env python3
"""
Synthetic Image Generator - Create images from gap analysis recommendations.

Takes gap analysis output and generates targeted synthetic images using:
- Base model (SDXL, Flux.1, ZIT)
- Trained character LoRA
- Optional pose/style LoRAs
- Structured prompts from gap recommendations

No ComfyUI needed - pure diffusers pipeline for automation.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Prompt, Confirm

console = Console()


# ============================================================================
# Prompt Templates
# ============================================================================

ANGLE_PROMPTS = {
    "front": [
        "facing camera directly, front view, looking at viewer",
        "straight on view, centered composition, direct gaze",
        "frontal portrait, eye contact, symmetrical face",
    ],
    "profile": [
        "side profile view, looking to the side, profile shot",
        "90 degree angle, side view portrait, profile composition",
        "lateral view, side facing, classical profile",
    ],
    "three_quarter": [
        "three-quarter view, slightly turned, angled portrait",
        "3/4 view, partial side angle, dynamic composition",
        "angled face view, turned slightly, dimensional portrait",
    ],
}

EXPRESSION_PROMPTS = {
    "neutral": "neutral expression, calm demeanor, relaxed face",
    "smile": "gentle smile, warm expression, happy demeanor",
    "serious": "serious expression, focused look, contemplative",
    "laughing": "laughing, joyful expression, big smile",
}

SETTING_PROMPTS = {
    "indoor": "indoor setting, soft lighting, interior background",
    "outdoor": "outdoor setting, natural lighting, exterior background",
    "studio": "studio lighting, professional setup, clean background",
    "casual": "casual environment, relaxed setting, natural backdrop",
}

QUALITY_SUFFIX = "high quality, detailed, sharp focus, professional photography, 8k uhd, dslr"


# ============================================================================
# Pipeline Setup
# ============================================================================

def load_pipeline(
    base_model: str,
    lora_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> StableDiffusionXLPipeline:
    """
    Load diffusers pipeline with optional LoRA.

    Args:
        base_model: HuggingFace model ID or path
        lora_path: Path to LoRA safetensors file
        device: Device to load on
        dtype: Data type for inference

    Returns:
        Configured pipeline
    """
    console.print(f"[cyan]Loading base model: {base_model}[/cyan]")

    # Load base pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
    )

    # Optimize scheduler for quality/speed balance
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )

    # Load LoRA if provided
    if lora_path and Path(lora_path).exists():
        console.print(f"[cyan]Loading LoRA: {lora_path}[/cyan]")
        pipe.load_lora_weights(lora_path)

    # Move to device
    pipe = pipe.to(device)

    # Enable optimizations
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            console.print("[dim]  ✓ xformers enabled[/dim]")
        except:
            pass

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    console.print("[green]✓ Pipeline ready[/green]\n")
    return pipe


# ============================================================================
# Prompt Generation
# ============================================================================

def build_prompt_from_gap(
    gap_recommendation: Dict,
    person_trigger: str,
    base_quality: str = QUALITY_SUFFIX,
    additional_context: str = ""
) -> str:
    """
    Build a structured prompt from gap recommendation.

    Args:
        gap_recommendation: Dict with type, target, count, reason
        person_trigger: Trigger word for person (e.g., "ohwx person")
        base_quality: Quality enhancement suffix
        additional_context: Additional prompt context

    Returns:
        Complete prompt string
    """

    gap_type = gap_recommendation.get("type", "")
    target = gap_recommendation.get("target", "")

    # Start with person trigger
    prompt_parts = [person_trigger]

    # Add target-specific prompts
    if gap_type == "face_angle":
        angle_templates = ANGLE_PROMPTS.get(target, ["portrait"])
        prompt_parts.append(random.choice(angle_templates))

    elif gap_type == "composition":
        if "hands" in target:
            prompt_parts.append("hands visible, showing hands, gesturing")
        elif "eyes" in target:
            prompt_parts.append("eyes clearly visible, expressive eyes")

    # Add random variation
    expression = random.choice(list(EXPRESSION_PROMPTS.values()))
    setting = random.choice(list(SETTING_PROMPTS.values()))

    prompt_parts.extend([expression, setting])

    # Add context and quality
    if additional_context:
        prompt_parts.append(additional_context)

    prompt_parts.append(base_quality)

    return ", ".join(prompt_parts)


def build_negative_prompt() -> str:
    """Standard negative prompt for quality."""
    return (
        "blurry, low quality, jpeg artifacts, ugly, duplicate, morbid, "
        "mutilated, extra fingers, mutated hands, poorly drawn hands, "
        "poorly drawn face, mutation, deformed, bad anatomy, bad proportions, "
        "extra limbs, cloned face, disfigured, gross proportions, "
        "malformed limbs, missing arms, missing legs, extra arms, extra legs, "
        "fused fingers, too many fingers, long neck, lowres, worst quality"
    )


# ============================================================================
# Batch Generation
# ============================================================================

def generate_from_gaps(
    gap_report_path: str,
    output_dir: str,
    base_model: str,
    lora_path: Optional[str],
    person_trigger: str,
    num_images_per_gap: int = 5,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    device: str = "cuda"
):
    """
    Generate synthetic images based on gap analysis.

    Args:
        gap_report_path: Path to gap analysis JSON
        output_dir: Directory to save generated images
        base_model: HuggingFace model ID
        lora_path: Path to trained LoRA
        person_trigger: Trigger word for person
        num_images_per_gap: How many images to generate per gap
        num_inference_steps: Diffusion steps
        guidance_scale: CFG scale
        seed: Random seed (None for random)
        device: Device to run on
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load gap report
    with open(gap_report_path) as f:
        gap_report = json.load(f)

    recommendations = gap_report.get("recommendations", [])

    if not recommendations:
        console.print("[yellow]No gap recommendations found in report[/yellow]")
        return

    # Display generation plan
    console.print(Panel.fit(
        "[bold cyan]Synthetic Generation Plan[/bold cyan]\n"
        f"Base Model: {base_model}\n"
        f"LoRA: {lora_path}\n"
        f"Person: {person_trigger}\n"
        f"Output: {output_path}",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Show recommendations
    plan_table = Table(title="Generation Queue", box=box.ROUNDED, border_style="cyan")
    plan_table.add_column("Gap Type", style="yellow")
    plan_table.add_column("Target", style="cyan")
    plan_table.add_column("Images", justify="right", style="green")

    total_images = 0
    for rec in recommendations:
        # Limit to requested count or num_images_per_gap, whichever is smaller
        count = min(rec.get("count", num_images_per_gap), num_images_per_gap * 3)
        total_images += count
        plan_table.add_row(
            rec.get("type", "unknown"),
            rec.get("target", "unknown"),
            str(count)
        )

    console.print(plan_table)
    console.print(f"\n[bold]Total images to generate: {total_images}[/bold]\n")

    # Confirm before proceeding
    if not Confirm.ask("Proceed with generation?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Load pipeline
    pipe = load_pipeline(base_model, lora_path, device=device)

    # Set seed if provided
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    else:
        generator = None

    # Generate images
    negative_prompt = build_negative_prompt()
    metadata_log = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Generating images...", total=total_images)

        image_counter = 0

        for rec in recommendations:
            gap_type = rec.get("type", "unknown")
            target = rec.get("target", "unknown")
            count = min(rec.get("count", num_images_per_gap), num_images_per_gap * 3)

            for i in range(count):
                # Build prompt
                prompt = build_prompt_from_gap(rec, person_trigger)

                # Generate
                try:
                    # Use different seed for each image if base seed provided
                    if seed is not None:
                        current_seed = seed + image_counter
                        gen = torch.Generator(device=device).manual_seed(current_seed)
                    else:
                        current_seed = random.randint(0, 2**32 - 1)
                        gen = torch.Generator(device=device).manual_seed(current_seed)

                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=gen,
                    )

                    image = result.images[0]

                    # Save with structured naming
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"synthetic_{gap_type}_{target}_{i:03d}_{timestamp}.png"
                    image_path = output_path / filename

                    image.save(image_path)

                    # Log metadata
                    metadata = {
                        "file": str(filename),
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "gap_type": gap_type,
                        "gap_target": target,
                        "seed": current_seed,
                        "steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                        "model": base_model,
                        "lora": str(lora_path) if lora_path else None,
                        "person": person_trigger,
                        "synthetic": True,
                        "generated_at": datetime.now().isoformat(),
                    }
                    metadata_log.append(metadata)

                    image_counter += 1
                    progress.update(task, advance=1)

                except Exception as e:
                    console.print(f"\n[red]Error generating image {image_counter}: {e}[/red]")
                    progress.update(task, advance=1)
                    continue

    # Save metadata
    metadata_file = output_path / "generation_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata_log, f, indent=2)

    console.print()
    console.print(Panel.fit(
        f"[bold green]✓ Generation Complete[/bold green]\n"
        f"Generated: {image_counter} images\n"
        f"Output: {output_path}\n"
        f"Metadata: {metadata_file}",
        border_style="green"
    ))


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic images from gap analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from gap report
  python generate_synthetic.py gaps.json \\
    --base-model stabilityai/stable-diffusion-xl-base-1.0 \\
    --lora outputs/lora_models/person/lora.safetensors \\
    --person "ohwx person" \\
    --output outputs/synthetic/person

  # With ZIT model
  python generate_synthetic.py gaps.json \\
    --base-model cagliostrolab/animagine-xl-3.1 \\
    --lora lora.safetensors \\
    --person "1girl, ohwx" \\
    --num-per-gap 10

  # With seed for reproducibility
  python generate_synthetic.py gaps.json \\
    --base-model stabilityai/stable-diffusion-xl-base-1.0 \\
    --lora lora.safetensors \\
    --person "ohwx person" \\
    --seed 42
        """
    )

    parser.add_argument("gap_report", help="Path to gap analysis JSON")
    parser.add_argument("--base-model", "-m", required=True,
                       help="HuggingFace model ID or path")
    parser.add_argument("--lora", "-l", help="Path to trained LoRA .safetensors")
    parser.add_argument("--person", "-p", required=True,
                       help="Person trigger word (e.g., 'ohwx person')")
    parser.add_argument("--output", "-o", default="outputs/synthetic",
                       help="Output directory")
    parser.add_argument("--num-per-gap", type=int, default=5,
                       help="Images to generate per gap (default: 5)")
    parser.add_argument("--steps", type=int, default=30,
                       help="Inference steps (default: 30)")
    parser.add_argument("--guidance", type=float, default=7.5,
                       help="Guidance scale (default: 7.5)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    generate_from_gaps(
        gap_report_path=args.gap_report,
        output_dir=args.output,
        base_model=args.base_model,
        lora_path=args.lora,
        person_trigger=args.person,
        num_images_per_gap=args.num_per_gap,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        device=args.device
    )
