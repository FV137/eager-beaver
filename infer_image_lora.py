#!/usr/bin/env python3
"""
Image LoRA Inference - Test character LoRAs

Load and test trained character LoRAs for image generation.
Quick test generations to see the learned visual identity.
"""

import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich import box

console = Console()

# Try to import diffusers
try:
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


# ============================================================================
# Pipeline Loading
# ============================================================================

def load_pipeline_with_lora(
    base_model: str,
    lora_path: str,
    lora_scale: float = 0.8,
    device: str = "cuda"
):
    """Load diffusion pipeline with LoRA."""

    console.print(f"[cyan]Loading {base_model}...[/cyan]")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16"
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True
    )

    # Load LoRA weights
    console.print(f"[cyan]Loading LoRA from {lora_path}...[/cyan]")
    pipe.load_lora_weights(lora_path)

    # Set LoRA scale
    pipe.set_adapters(["default"], adapter_weights=[lora_scale])

    pipe = pipe.to(device)

    console.print(f"[green]✓ Pipeline loaded (LoRA scale: {lora_scale})[/green]\n")

    return pipe


# ============================================================================
# Generation
# ============================================================================

def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = None,
    output_path: str = None
):
    """Generate single image."""

    # Set seed for reproducibility
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    console.print(f"[cyan]Generating image...[/cyan]")
    console.print(f"[dim]Prompt: {prompt}[/dim]")
    if negative_prompt:
        console.print(f"[dim]Negative: {negative_prompt}[/dim]")
    console.print()

    # Generate
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # Save
    if output_path:
        image.save(output_path)
        console.print(f"[green]✓ Saved to {output_path}[/green]")
    else:
        # Auto-name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_generation_{timestamp}.png"
        image.save(filename)
        console.print(f"[green]✓ Saved to {filename}[/green]")

    return image


# ============================================================================
# Interactive Mode
# ============================================================================

def interactive_generation(
    pipe,
    trigger_word: str = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5
):
    """Interactive image generation."""

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Interactive Generation Mode[/bold cyan]\n\n"
        f"{'[yellow]Trigger Word:[/yellow] ' + trigger_word if trigger_word else ''}\n"
        "[dim]Enter prompts to generate test images\n"
        "Type 'quit' or 'exit' to stop[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    count = 0

    while True:
        # Get prompt
        try:
            prompt = Prompt.ask("[bold yellow]Prompt[/bold yellow]")
        except (KeyboardInterrupt, EOFError):
            break

        if prompt.lower() in ['quit', 'exit', 'q']:
            break

        if not prompt.strip():
            continue

        # Add trigger word if provided
        if trigger_word and trigger_word not in prompt.lower():
            prompt = f"{trigger_word}, {prompt}"

        # Optional negative prompt
        negative = Prompt.ask(
            "[dim]Negative prompt (optional)[/dim]",
            default=""
        )

        # Generate
        count += 1
        output_file = f"test_gen_{count:03d}.png"

        generate_image(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_path=output_file
        )

        console.print()

    console.print(f"\n[dim]Generated {count} test images[/dim]\n")


# ============================================================================
# Quick Test Suite
# ============================================================================

def run_test_suite(
    pipe,
    trigger_word: str,
    output_dir: str = "./test_generations"
):
    """Run quick test suite with common scenarios."""

    Path(output_dir).mkdir(exist_ok=True)

    test_prompts = [
        f"{trigger_word}, portrait, front view, neutral expression",
        f"{trigger_word}, smiling, happy expression, close-up",
        f"{trigger_word}, profile view, side angle",
        f"{trigger_word}, three-quarter view, confident pose",
        f"{trigger_word}, full body, standing pose",
        f"{trigger_word}, upper body, hands visible",
    ]

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Running Test Suite[/bold cyan]\n\n"
        f"Generating {len(test_prompts)} test images",
        border_style="cyan"
    ))
    console.print()

    for i, prompt in enumerate(test_prompts, 1):
        output_file = Path(output_dir) / f"test_{i:02d}.png"

        console.print(f"[yellow]{i}/{len(test_prompts)}:[/yellow] {prompt}")

        generate_image(
            pipe=pipe,
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=42 + i,  # Reproducible
            output_path=str(output_file)
        )

        console.print()

    console.print(Panel.fit(
        f"[bold green]Test Suite Complete![/bold green]\n\n"
        f"Generated {len(test_prompts)} images in {output_dir}",
        border_style="green"
    ))
    console.print()


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test character LoRA with image generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Load a trained character LoRA and generate test images to explore
the learned visual identity.

Examples:
  # Interactive mode
  python infer_image_lora.py stabilityai/stable-diffusion-xl-base-1.0 \\
    --lora ./lora_output/lora.safetensors \\
    --trigger "alexis"

  # Quick test suite
  python infer_image_lora.py stabilityai/stable-diffusion-xl-base-1.0 \\
    --lora ./lora_output/lora.safetensors \\
    --trigger "alexis" \\
    --test-suite

  # Single prompt
  python infer_image_lora.py stabilityai/stable-diffusion-xl-base-1.0 \\
    --lora ./lora_output/lora.safetensors \\
    --prompt "portrait, smiling" \\
    --trigger "alexis"
        """
    )

    parser.add_argument("base_model", help="Base diffusion model")
    parser.add_argument("--lora", "-l", required=True,
                       help="Path to LoRA weights (.safetensors)")
    parser.add_argument("--trigger", "-t",
                       help="Trigger word for LoRA")
    parser.add_argument("--prompt", "-p",
                       help="Single prompt (non-interactive)")
    parser.add_argument("--negative", default="",
                       help="Negative prompt")
    parser.add_argument("--steps", type=int, default=30,
                       help="Inference steps (default: 30)")
    parser.add_argument("--guidance", type=float, default=7.5,
                       help="Guidance scale (default: 7.5)")
    parser.add_argument("--lora-scale", type=float, default=0.8,
                       help="LoRA strength (default: 0.8)")
    parser.add_argument("--test-suite", action="store_true",
                       help="Run quick test suite")
    parser.add_argument("--output", "-o",
                       help="Output file/directory")

    args = parser.parse_args()

    # Check dependencies
    if not DIFFUSERS_AVAILABLE:
        console.print("[red]Error: diffusers not installed[/red]")
        console.print("[dim]Install: pip install diffusers[/dim]")
        sys.exit(1)

    # Load pipeline
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Loading Character LoRA[/bold cyan]\n\n"
        f"Base Model: {args.base_model}\n"
        f"LoRA: {args.lora}\n"
        f"Trigger: {args.trigger or 'None'}",
        border_style="cyan"
    ))
    console.print()

    pipe = load_pipeline_with_lora(
        base_model=args.base_model,
        lora_path=args.lora,
        lora_scale=args.lora_scale
    )

    # Run mode
    if args.test_suite:
        if not args.trigger:
            console.print("[red]Error: --trigger required for test suite[/red]")
            sys.exit(1)

        output_dir = args.output or "./test_generations"
        run_test_suite(pipe, args.trigger, output_dir)

    elif args.prompt:
        prompt = args.prompt
        if args.trigger and args.trigger not in prompt.lower():
            prompt = f"{args.trigger}, {prompt}"

        generate_image(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            output_path=args.output
        )

    else:
        interactive_generation(
            pipe=pipe,
            trigger_word=args.trigger,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance
        )


if __name__ == "__main__":
    main()
