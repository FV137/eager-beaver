#!/usr/bin/env python3
"""
Bootstrap Loop - Self-empowering training cycle automation.

Automates the complete cycle:
1. Analyze dataset for gaps
2. Generate synthetic images for gaps
3. Validate synthetic images
4. Merge approved images into dataset
5. Retrain LoRA with improved dataset
6. Repeat until convergence

This is the heart of the self-empowering loop.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import box
from rich.prompt import Confirm, IntPrompt

console = Console()


# ============================================================================
# Bootstrap Configuration
# ============================================================================

class BootstrapConfig:
    """Configuration for bootstrap loop."""

    def __init__(self, config_file: Optional[str] = None):
        self.config = {
            "person_name": "person",
            "person_trigger": "ohwx person",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "dataset_dir": "outputs/facevault/organized/person",
            "models_dir": "models",
            "output_dir": "outputs/bootstrap",
            "lora_output_dir": "outputs/lora_models/person",
            "max_iterations": 5,
            "convergence_threshold": 0.05,  # Stop if <5% improvement
            "generation": {
                "num_per_gap": 5,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
            },
            "validation": {
                "auto_filter": True,
            },
            "training": {
                "preset": "sdxl",
                "epochs": 10,
            }
        }

        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                user_config = json.load(f)
                self.config.update(user_config)

    def save(self, output_file: str):
        """Save config to file."""
        with open(output_file, "w") as f:
            json.dump(self.config, f, indent=2)


# ============================================================================
# Iteration Tracking
# ============================================================================

class IterationTracker:
    """Track metrics across iterations."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.output_dir / "iteration_history.json"
        self.history = []

        if self.history_file.exists():
            with open(self.history_file) as f:
                self.history = json.load(f)

    def log_iteration(self, iteration: int, metrics: Dict):
        """Log metrics for an iteration."""
        entry = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }

        self.history.append(entry)

        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_improvement(self, metric_name: str) -> Optional[float]:
        """Calculate improvement in metric from previous iteration."""
        if len(self.history) < 2:
            return None

        current = self.history[-1].get(metric_name, 0)
        previous = self.history[-2].get(metric_name, 0)

        if previous == 0:
            return None

        improvement = (current - previous) / previous
        return improvement

    def check_convergence(self, threshold: float) -> bool:
        """Check if loop has converged (diminishing returns)."""
        if len(self.history) < 2:
            return False

        # Check approval rate improvement
        approval_improvement = self.get_improvement("synthetic_approval_rate")

        if approval_improvement is None:
            return False

        # Converged if improvement is below threshold
        return abs(approval_improvement) < threshold


# ============================================================================
# Bootstrap Loop Steps
# ============================================================================

def run_gap_analysis(config: BootstrapConfig, iteration: int) -> str:
    """Run gap analysis on current dataset."""
    console.print(Panel.fit(
        f"[bold cyan]Step 1: Gap Analysis[/bold cyan]\n"
        f"Iteration: {iteration}",
        border_style="cyan"
    ))

    dataset_dir = config.config["dataset_dir"]
    models_dir = config.config["models_dir"]
    output_file = Path(config.config["output_dir"]) / f"gaps_iter{iteration}.json"

    cmd = [
        sys.executable, "gap_analysis.py",
        dataset_dir,
        "--models-dir", models_dir,
        "--output", str(output_file)
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        console.print("[red]✗ Gap analysis failed[/red]")
        return None

    console.print(f"[green]✓ Gap analysis complete: {output_file}[/green]\n")
    return str(output_file)


def run_generation(config: BootstrapConfig, gap_file: str, iteration: int) -> str:
    """Generate synthetic images from gaps."""
    console.print(Panel.fit(
        f"[bold cyan]Step 2: Synthetic Generation[/bold cyan]\n"
        f"Iteration: {iteration}",
        border_style="cyan"
    ))

    output_dir = Path(config.config["output_dir"]) / f"synthetic_iter{iteration}"
    lora_path = Path(config.config["lora_output_dir"]) / "lora.safetensors"

    gen_config = config.config["generation"]

    cmd = [
        sys.executable, "generate_synthetic.py",
        gap_file,
        "--base-model", config.config["base_model"],
        "--person", config.config["person_trigger"],
        "--output", str(output_dir),
        "--num-per-gap", str(gen_config["num_per_gap"]),
        "--steps", str(gen_config["num_inference_steps"]),
        "--guidance", str(gen_config["guidance_scale"]),
    ]

    # Add LoRA if exists
    if lora_path.exists():
        cmd.extend(["--lora", str(lora_path)])
    else:
        console.print("[yellow]⚠ No LoRA found - using base model only[/yellow]")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        console.print("[red]✗ Generation failed[/red]")
        return None

    console.print(f"[green]✓ Generation complete: {output_dir}[/green]\n")
    return str(output_dir)


def run_validation(config: BootstrapConfig, synthetic_dir: str, iteration: int) -> Dict:
    """Validate synthetic images."""
    console.print(Panel.fit(
        f"[bold cyan]Step 3: Validation[/bold cyan]\n"
        f"Iteration: {iteration}",
        border_style="cyan"
    ))

    approved_dir = Path(config.config["output_dir"]) / f"approved_iter{iteration}"
    rejected_dir = Path(config.config["output_dir"]) / f"rejected_iter{iteration}"
    models_dir = config.config["models_dir"]

    cmd = [
        sys.executable, "validate_synthetic.py",
        synthetic_dir,
        "--models-dir", models_dir,
        "--approved", str(approved_dir),
        "--rejected", str(rejected_dir),
    ]

    if not config.config["validation"]["auto_filter"]:
        cmd.append("--no-auto-filter")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        console.print("[red]✗ Validation failed[/red]")
        return None

    # Read validation report
    report_file = Path(synthetic_dir) / "validation_report.json"
    if report_file.exists():
        with open(report_file) as f:
            report = json.load(f)

        console.print(f"[green]✓ Validation complete[/green]\n")
        return {
            "approved_dir": str(approved_dir),
            "rejected_dir": str(rejected_dir),
            "report": report
        }

    return None


def merge_into_dataset(config: BootstrapConfig, approved_dir: str, iteration: int):
    """Merge approved synthetic images into training dataset."""
    console.print(Panel.fit(
        f"[bold cyan]Step 4: Dataset Merge[/bold cyan]\n"
        f"Iteration: {iteration}",
        border_style="cyan"
    ))

    dataset_dir = Path(config.config["dataset_dir"])
    approved_path = Path(approved_dir)

    if not approved_path.exists():
        console.print("[yellow]No approved images to merge[/yellow]\n")
        return 0

    # Count approved images
    approved_images = list(approved_path.glob("*.png"))
    approved_images.extend(approved_path.glob("*.jpg"))

    if not approved_images:
        console.print("[yellow]No approved images found[/yellow]\n")
        return 0

    console.print(f"[cyan]Merging {len(approved_images)} approved images...[/cyan]")

    # Copy to dataset
    merged_count = 0
    for img_path in approved_images:
        # Rename to avoid conflicts
        new_name = f"synthetic_iter{iteration}_{img_path.name}"
        dst_path = dataset_dir / new_name

        shutil.copy2(img_path, dst_path)
        merged_count += 1

    console.print(f"[green]✓ Merged {merged_count} images into dataset[/green]\n")
    return merged_count


def run_training(config: BootstrapConfig, iteration: int) -> bool:
    """Run LoRA training on updated dataset."""
    console.print(Panel.fit(
        f"[bold cyan]Step 5: LoRA Training[/bold cyan]\n"
        f"Iteration: {iteration}",
        border_style="cyan"
    ))

    console.print("[yellow]Training integration pending - placeholder[/yellow]")
    console.print("[dim]This would call lora_train.py with updated dataset[/dim]\n")

    # TODO: Integrate with lora_train.py
    # For now, assume training succeeds
    return True


# ============================================================================
# Main Bootstrap Loop
# ============================================================================

def run_bootstrap_loop(config_file: Optional[str] = None):
    """Run complete bootstrap loop."""

    # Load configuration
    config = BootstrapConfig(config_file)

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔄 Bootstrap Loop - Self-Empowering Training[/bold cyan]\n\n"
        f"Person: {config.config['person_name']}\n"
        f"Base Model: {config.config['base_model']}\n"
        f"Max Iterations: {config.config['max_iterations']}\n"
        f"Convergence Threshold: {config.config['convergence_threshold']}",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Initialize tracker
    tracker = IterationTracker(config.config["output_dir"])

    # Check if resuming
    current_iteration = len(tracker.history) + 1

    if current_iteration > 1:
        console.print(f"[yellow]Resuming from iteration {current_iteration}[/yellow]\n")

    # Confirm before starting
    if not Confirm.ask("Start bootstrap loop?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        return

    # Main loop
    for iteration in range(current_iteration, config.config["max_iterations"] + 1):
        console.print()
        console.print("=" * 80)
        console.print(f"[bold yellow]ITERATION {iteration}/{config.config['max_iterations']}[/bold yellow]")
        console.print("=" * 80)
        console.print()

        metrics = {"iteration": iteration}

        # Step 1: Gap Analysis
        gap_file = run_gap_analysis(config, iteration)
        if not gap_file:
            break

        # Read gap recommendations
        with open(gap_file) as f:
            gap_data = json.load(f)

        metrics["gap_count"] = len(gap_data.get("recommendations", []))

        # Step 2: Generate Synthetic
        synthetic_dir = run_generation(config, gap_file, iteration)
        if not synthetic_dir:
            break

        # Step 3: Validate
        validation_result = run_validation(config, synthetic_dir, iteration)
        if not validation_result:
            break

        report = validation_result["report"]
        stats = report.get("statistics", {})

        metrics["synthetic_generated"] = report.get("total", 0)
        metrics["synthetic_approved"] = stats.get("total_approved", 0)
        metrics["synthetic_rejected"] = stats.get("total_rejected", 0)
        metrics["synthetic_approval_rate"] = stats.get("approval_rate", 0)
        metrics["synthetic_avg_quality"] = stats.get("avg_quality_score", 0)

        # Step 4: Merge approved images
        merged_count = merge_into_dataset(config, validation_result["approved_dir"], iteration)
        metrics["merged_count"] = merged_count

        # Step 5: Train (if we have new data)
        if merged_count > 0:
            train_success = run_training(config, iteration)
            metrics["training_success"] = train_success
        else:
            console.print("[yellow]No new images - skipping training[/yellow]\n")
            metrics["training_success"] = False

        # Log iteration
        tracker.log_iteration(iteration, metrics)

        # Display iteration summary
        display_iteration_summary(metrics)

        # Check convergence
        if tracker.check_convergence(config.config["convergence_threshold"]):
            console.print()
            console.print(Panel.fit(
                "[bold green]✓ Convergence Achieved[/bold green]\n"
                "Improvement below threshold - loop complete!",
                border_style="green"
            ))
            break

        # Continue prompt
        if iteration < config.config["max_iterations"]:
            console.print()
            if not Confirm.ask("Continue to next iteration?", default=True):
                console.print("[yellow]Loop stopped by user[/yellow]")
                break

    # Final summary
    display_final_summary(tracker)


def display_iteration_summary(metrics: Dict):
    """Display summary for iteration."""
    console.print()
    console.print(Panel.fit(
        f"[bold green]Iteration {metrics['iteration']} Complete[/bold green]\n\n"
        f"Gaps Found: {metrics.get('gap_count', 0)}\n"
        f"Generated: {metrics.get('synthetic_generated', 0)}\n"
        f"Approved: {metrics.get('synthetic_approved', 0)} "
        f"({metrics.get('synthetic_approval_rate', 0):.1f}%)\n"
        f"Rejected: {metrics.get('synthetic_rejected', 0)}\n"
        f"Merged: {metrics.get('merged_count', 0)}",
        border_style="green"
    ))


def display_final_summary(tracker: IterationTracker):
    """Display final summary of all iterations."""
    console.print()
    console.print("=" * 80)
    console.print("[bold cyan]BOOTSTRAP LOOP COMPLETE[/bold cyan]")
    console.print("=" * 80)
    console.print()

    if not tracker.history:
        console.print("[yellow]No iterations completed[/yellow]")
        return

    # Build summary table
    summary_table = Table(title="Iteration History", box=box.ROUNDED, border_style="cyan")
    summary_table.add_column("Iter", justify="right", style="cyan")
    summary_table.add_column("Generated", justify="right", style="yellow")
    summary_table.add_column("Approved", justify="right", style="green")
    summary_table.add_column("Approval %", justify="right", style="dim")
    summary_table.add_column("Merged", justify="right", style="magenta")

    for entry in tracker.history:
        summary_table.add_row(
            str(entry.get("iteration", "?")),
            str(entry.get("synthetic_generated", 0)),
            str(entry.get("synthetic_approved", 0)),
            f"{entry.get('synthetic_approval_rate', 0):.1f}%",
            str(entry.get("merged_count", 0))
        )

    console.print(summary_table)
    console.print()

    # Overall stats
    total_generated = sum(e.get("synthetic_generated", 0) for e in tracker.history)
    total_approved = sum(e.get("synthetic_approved", 0) for e in tracker.history)
    total_merged = sum(e.get("merged_count", 0) for e in tracker.history)

    console.print(Panel.fit(
        f"[bold green]Overall Statistics[/bold green]\n\n"
        f"Total Iterations: {len(tracker.history)}\n"
        f"Total Generated: {total_generated}\n"
        f"Total Approved: {total_approved}\n"
        f"Total Merged: {total_merged}\n"
        f"Overall Approval Rate: {(total_approved/total_generated*100) if total_generated > 0 else 0:.1f}%",
        border_style="green"
    ))


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-empowering bootstrap training loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The Bootstrap Loop automates:
1. Gap analysis on current dataset
2. Generate synthetic images targeting gaps
3. Validate synthetic quality with YOLO
4. Merge approved images into dataset
5. Retrain LoRA with improved data
6. Repeat until convergence

Example:
  python bootstrap_loop.py --config bootstrap_config.json

Config file format:
  {
    "person_name": "alice",
    "person_trigger": "ohwx alice",
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "dataset_dir": "outputs/facevault/organized/alice",
    "max_iterations": 5,
    "convergence_threshold": 0.05
  }
        """
    )

    parser.add_argument("--config", "-c", help="Bootstrap configuration JSON")

    args = parser.parse_args()

    try:
        run_bootstrap_loop(args.config)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
