#!/usr/bin/env python3
"""
LoRA Train - Simplified LoRA training integration.

Automatically configures and trains LoRA models from prepared datasets.
Supports multiple training backends with smart defaults.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Confirm, Prompt, IntPrompt, FloatPrompt

console = Console()

# Project paths
PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "lora_models"


# ============================================================================
# Config Generation
# ============================================================================

def generate_kohya_config(
    dataset_path: Path,
    output_path: Path,
    metadata: Dict,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    resolution: int = 1024,
    batch_size: int = 1,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    network_dim: int = 32,
    network_alpha: int = 16,
) -> Dict:
    """
    Generate training config for kohya_ss format.

    Args:
        dataset_path: Path to prepared dataset
        output_path: Where to save trained LoRA
        metadata: Dataset metadata from lora_prep
        base_model: Base model to train from
        resolution: Training resolution
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        network_dim: LoRA network dimension (rank)
        network_alpha: LoRA network alpha

    Returns:
        Config dictionary
    """

    trigger_word = metadata.get("trigger_word", "person")
    total_images = metadata.get("total_images", 0)

    # Calculate steps
    steps_per_epoch = total_images // batch_size
    max_train_steps = steps_per_epoch * epochs

    # Save frequency (every epoch)
    save_every_n_epochs = 1

    config = {
        "model_arguments": {
            "pretrained_model_name_or_path": base_model,
            "v2": False,
            "v_parameterization": False,
        },
        "dataset_arguments": {
            "dataset_dir": str(dataset_path),
            "resolution": resolution,
            "batch_size": batch_size,
            "enable_bucket": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": 2048,
            "bucket_reso_steps": 64,
            "caption_extension": ".txt",
            "shuffle_caption": True,
            "keep_tokens": 1,  # Keep trigger word
        },
        "training_arguments": {
            "output_dir": str(output_path),
            "output_name": trigger_word,
            "max_train_epochs": epochs,
            "max_train_steps": max_train_steps,
            "learning_rate": learning_rate,
            "lr_scheduler": "cosine_with_restarts",
            "lr_warmup_steps": int(max_train_steps * 0.1),
            "train_batch_size": batch_size,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "fp16",
            "save_precision": "fp16",
            "save_every_n_epochs": save_every_n_epochs,
            "save_model_as": "safetensors",
            "clip_skip": 2,
            "seed": 42,
            "prior_loss_weight": 1.0,
        },
        "network_arguments": {
            "network_module": "networks.lora",
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "network_train_unet_only": False,
            "network_train_text_encoder_only": False,
        },
        "optimizer_arguments": {
            "optimizer_type": "AdamW8bit",
            "learning_rate": learning_rate,
            "lr_scheduler": "cosine_with_restarts",
            "lr_scheduler_num_cycles": 3,
        },
        "logging_arguments": {
            "logging_dir": str(output_path / "logs"),
            "log_with": "tensorboard",
            "log_prefix": trigger_word,
        },
    }

    return config


def generate_simple_config(
    dataset_path: Path,
    metadata: Dict,
    output_path: Path,
) -> Dict:
    """
    Generate simplified training config for custom trainers or reference.

    Returns:
        Simple config dictionary
    """

    shot_dist = metadata.get("shot_distribution", {})
    total = metadata.get("total_images", 0)

    config = {
        "dataset": {
            "path": str(dataset_path),
            "name": metadata.get("person", "unknown"),
            "trigger_word": metadata.get("trigger_word", "person"),
            "total_images": total,
            "shot_distribution": shot_dist,
        },
        "training": {
            "output_path": str(output_path),
            "recommended_epochs": calculate_recommended_epochs(total),
            "recommended_batch_size": calculate_recommended_batch_size(total),
            "recommended_learning_rate": 1e-4,
            "recommended_network_dim": 32,
        },
        "metadata": metadata.get("top_concepts", []),
    }

    return config


def calculate_recommended_epochs(total_images: int) -> int:
    """Calculate recommended epochs based on dataset size."""
    if total_images < 20:
        return 20
    elif total_images < 50:
        return 15
    elif total_images < 100:
        return 10
    else:
        return 8


def calculate_recommended_batch_size(total_images: int) -> int:
    """Calculate recommended batch size."""
    if total_images < 30:
        return 1
    elif total_images < 100:
        return 2
    else:
        return 4


# ============================================================================
# Training Command
# ============================================================================

@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output directory for trained LoRA')
@click.option('--base-model', '-m', default='stabilityai/stable-diffusion-xl-base-1.0', help='Base model to train from')
@click.option('--backend', type=click.Choice(['kohya', 'simple', 'config-only']), default='config-only', help='Training backend')
@click.option('--resolution', type=int, default=1024, help='Training resolution')
@click.option('--epochs', type=int, help='Number of epochs (auto-calculated if not specified)')
@click.option('--batch-size', type=int, help='Batch size (auto-calculated if not specified)')
@click.option('--learning-rate', '-lr', type=float, default=1e-4, help='Learning rate')
@click.option('--network-dim', type=int, default=32, help='LoRA network dimension (rank)')
@click.option('--network-alpha', type=int, default=16, help='LoRA network alpha')
@click.option('--interactive', '-i', is_flag=True, help='Interactive configuration')
def train(
    dataset_path: str,
    output: Optional[str],
    base_model: str,
    backend: str,
    resolution: int,
    epochs: Optional[int],
    batch_size: Optional[int],
    learning_rate: float,
    network_dim: int,
    network_alpha: int,
    interactive: bool,
):
    """
    Train a LoRA model from a prepared dataset.

    Examples:

        # Generate config only (default - no external dependencies)
        lora_train outputs/lora_datasets/emma

        # Interactive configuration
        lora_train outputs/lora_datasets/emma --interactive

        # Custom settings
        lora_train outputs/lora_datasets/emma \\
          --epochs 15 \\
          --batch-size 2 \\
          --learning-rate 1e-4
    """

    dataset_path = Path(dataset_path)

    # Load metadata
    metadata_file = dataset_path / "metadata.json"
    if not metadata_file.exists():
        console.print(f"[red]Error:[/red] No metadata.json found in {dataset_path}")
        console.print("[yellow]Tip:[/yellow] Run lora_prep first to prepare the dataset")
        return

    with open(metadata_file) as f:
        metadata = json.load(f)

    trigger_word = metadata.get("trigger_word", "person")
    person_name = metadata.get("person", "unknown")
    total_images = metadata.get("total_images", 0)
    shot_dist = metadata.get("shot_distribution", {})

    # Auto-generate output path if not provided
    if not output:
        output_path = OUTPUT_DIR / trigger_word / datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        output_path = Path(output)

    output_path.mkdir(parents=True, exist_ok=True)

    # Header
    console.print(Panel.fit(
        f"[bold cyan]LoRA Training Setup[/bold cyan]\n"
        f"Person: {person_name}\n"
        f"Trigger: {trigger_word}\n"
        f"Dataset: {dataset_path}\n"
        f"Images: {total_images}",
        border_style="cyan"
    ))
    console.print()

    # Show shot distribution
    dist_table = Table(title="Dataset Overview", box=box.ROUNDED, border_style="cyan")
    dist_table.add_column("Shot Type", style="cyan")
    dist_table.add_column("Count", justify="right", style="yellow")
    dist_table.add_column("Percentage", justify="right", style="green")

    for shot_type in ["close", "mid", "far"]:
        count = shot_dist.get(shot_type, 0)
        pct = (count / total_images * 100) if total_images > 0 else 0
        dist_table.add_row(shot_type.capitalize(), str(count), f"{pct:.1f}%")

    console.print(dist_table)
    console.print()

    # Calculate recommendations
    if not epochs:
        epochs = calculate_recommended_epochs(total_images)

    if not batch_size:
        batch_size = calculate_recommended_batch_size(total_images)

    # Show recommendations
    rec_table = Table(title="Recommended Settings", box=box.ROUNDED, border_style="yellow")
    rec_table.add_column("Parameter", style="cyan")
    rec_table.add_column("Value", style="yellow")
    rec_table.add_column("Reasoning", style="dim")

    rec_table.add_row("Epochs", str(epochs), f"{total_images} images → {epochs} epochs optimal")
    rec_table.add_row("Batch Size", str(batch_size), "Based on dataset size")
    rec_table.add_row("Learning Rate", f"{learning_rate:.0e}", "Standard for LoRA")
    rec_table.add_row("Network Dim", str(network_dim), "Good balance: detail vs size")
    rec_table.add_row("Network Alpha", str(network_alpha), "Typically dim/2")

    console.print(rec_table)
    console.print()

    # Interactive mode
    if interactive:
        console.print("[bold]Interactive Configuration[/bold]\n")

        epochs = IntPrompt.ask("Epochs", default=epochs)
        batch_size = IntPrompt.ask("Batch size", default=batch_size)
        learning_rate = FloatPrompt.ask("Learning rate", default=learning_rate)
        network_dim = IntPrompt.ask("Network dimension", default=network_dim)
        network_alpha = IntPrompt.ask("Network alpha", default=network_alpha)

        console.print()

    # Generate configs
    console.print("[cyan]Generating training configuration...[/cyan]\n")

    # Kohya-style config
    kohya_config = generate_kohya_config(
        dataset_path=dataset_path,
        output_path=output_path,
        metadata=metadata,
        base_model=base_model,
        resolution=resolution,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        network_dim=network_dim,
        network_alpha=network_alpha,
    )

    kohya_config_file = output_path / "kohya_config.json"
    with open(kohya_config_file, "w") as f:
        json.dump(kohya_config, f, indent=2)

    console.print(f"[green]✓[/green] Saved kohya_ss config: {kohya_config_file}")

    # Simple config
    simple_config = generate_simple_config(
        dataset_path=dataset_path,
        metadata=metadata,
        output_path=output_path,
    )

    simple_config_file = output_path / "training_config.json"
    with open(simple_config_file, "w") as f:
        json.dump(simple_config, f, indent=2)

    console.print(f"[green]✓[/green] Saved simple config: {simple_config_file}")

    # Training info file
    training_info = {
        "dataset": str(dataset_path),
        "person": person_name,
        "trigger_word": trigger_word,
        "total_images": total_images,
        "shot_distribution": shot_dist,
        "training_params": {
            "base_model": base_model,
            "resolution": resolution,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
        },
        "output_path": str(output_path),
        "created_at": datetime.now().isoformat(),
    }

    info_file = output_path / "training_info.json"
    with open(info_file, "w") as f:
        json.dump(training_info, f, indent=2)

    console.print(f"[green]✓[/green] Saved training info: {info_file}")

    # Create training script template
    training_script = f"""#!/bin/bash
# LoRA Training Script for {person_name}
# Generated by Eager Beaver - LoRA Train

# Dataset: {dataset_path}
# Trigger word: {trigger_word}
# Total images: {total_images}

# Training parameters:
# - Epochs: {epochs}
# - Batch size: {batch_size}
# - Learning rate: {learning_rate}
# - Network dim: {network_dim}

echo "Training LoRA for {person_name} (trigger: {trigger_word})"
echo "Dataset: {dataset_path}"
echo ""

# Method 1: Using kohya_ss (if installed)
# Uncomment and adjust paths:
# python /path/to/kohya_ss/train_network.py \\
#   --config {kohya_config_file}

# Method 2: Using diffusers (example)
# accelerate launch train_lora.py \\
#   --pretrained_model_name_or_path="{base_model}" \\
#   --train_data_dir="{dataset_path}" \\
#   --output_dir="{output_path}" \\
#   --resolution={resolution} \\
#   --train_batch_size={batch_size} \\
#   --num_train_epochs={epochs} \\
#   --learning_rate={learning_rate} \\
#   --rank={network_dim}

# Method 3: Using ComfyUI (manual setup)
# 1. Load images from: {dataset_path}
# 2. Use captions from: {dataset_path}/captions/
# 3. Trigger word: {trigger_word}
# 4. Settings: {epochs} epochs, lr={learning_rate}

echo ""
echo "Config files ready!"
echo "Edit this script with your preferred training method."
"""

    script_file = output_path / "train.sh"
    with open(script_file, "w") as f:
        f.write(training_script)

    script_file.chmod(0o755)
    console.print(f"[green]✓[/green] Created training script: {script_file}")

    console.print()

    # Summary
    summary = Table(show_header=False, box=box.ROUNDED, border_style="green")
    summary.add_row("🎯 Trigger word", f"[bold]{trigger_word}[/bold]")
    summary.add_row("📦 Dataset", str(dataset_path))
    summary.add_row("📸 Total images", str(total_images))
    summary.add_row("🔧 Config files", str(output_path))
    summary.add_row("📄 Kohya config", str(kohya_config_file.name))
    summary.add_row("📄 Simple config", str(simple_config_file.name))
    summary.add_row("🚀 Training script", str(script_file.name))

    console.print(Panel(summary, title="[bold green]Setup Complete!", border_style="green"))

    console.print()
    console.print("[dim]Next steps:[/dim]")
    console.print(f"  1. Review configs in: {output_path}")
    console.print(f"  2. Edit train.sh with your training method")
    console.print(f"  3. Run: [bold]cd {output_path} && ./train.sh[/bold]")
    console.print(f"  4. Or use kohya_ss: [bold]python train_network.py --config {kohya_config_file}[/bold]")

    if backend == "kohya":
        console.print()
        console.print("[yellow]Note:[/yellow] kohya_ss must be installed separately")
        console.print("  Install: https://github.com/kohya-ss/sd-scripts")


# ============================================================================
# List Command
# ============================================================================

@click.command()
def list_models():
    """List all prepared training configs."""

    if not OUTPUT_DIR.exists():
        console.print("[yellow]No training configs found.[/yellow]")
        return

    configs = list(OUTPUT_DIR.rglob("training_info.json"))

    if not configs:
        console.print("[yellow]No training configs found.[/yellow]")
        return

    table = Table(title="Training Configs", box=box.ROUNDED, border_style="cyan")
    table.add_column("Person", style="cyan")
    table.add_column("Trigger", style="yellow")
    table.add_column("Images", justify="right", style="green")
    table.add_column("Epochs", justify="right", style="blue")
    table.add_column("Path", style="dim")

    for config_file in sorted(configs, key=lambda p: p.stat().st_mtime, reverse=True):
        with open(config_file) as f:
            info = json.load(f)

        person = info.get("person", "unknown")
        trigger = info.get("trigger_word", "?")
        images = info.get("total_images", 0)
        epochs = info.get("training_params", {}).get("epochs", "?")
        path = config_file.parent

        table.add_row(person, trigger, str(images), str(epochs), str(path.relative_to(OUTPUT_DIR)))

    console.print(table)


# ============================================================================
# Main CLI
# ============================================================================

@click.group()
@click.version_option(version="1.0.0", prog_name="LoRA Train")
def cli():
    """
    🎨 LoRA Train - Simplified LoRA training integration.

    Automatically configures training from prepared datasets.
    Generates configs for multiple training backends.
    """
    pass


cli.add_command(train)
cli.add_command(list_models)


if __name__ == "__main__":
    cli()
