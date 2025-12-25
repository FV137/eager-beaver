#!/usr/bin/env python3
"""
Eager Beaver - Beautiful TUI wizard for LoRA training pipeline.

Guides you through the complete workflow with smart defaults and
model-specific presets.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()

# Project paths
PROJECT_DIR = Path(__file__).parent
CONFIG_DIR = PROJECT_DIR / "configs"
SESSION_DIR = PROJECT_DIR / ".sessions"


# ============================================================================
# Session Management
# ============================================================================

def save_session(session_data: Dict):
    """Save current session state."""
    SESSION_DIR.mkdir(exist_ok=True)
    session_id = session_data.get("id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    session_file = SESSION_DIR / f"{session_id}.json"

    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

    return session_file


def load_session(session_id: str) -> Optional[Dict]:
    """Load saved session."""
    session_file = SESSION_DIR / f"{session_id}.json"
    if session_file.exists():
        with open(session_file) as f:
            return json.load(f)
    return None


def list_sessions() -> List[Dict]:
    """List all saved sessions."""
    if not SESSION_DIR.exists():
        return []

    sessions = []
    for session_file in SESSION_DIR.glob("*.json"):
        with open(session_file) as f:
            session = json.load(f)
            session["file"] = session_file.name
            sessions.append(session)

    return sorted(sessions, key=lambda s: s.get("last_updated", ""), reverse=True)


# ============================================================================
# Model Presets
# ============================================================================

def load_presets() -> Dict:
    """Load model presets configuration."""
    presets_file = CONFIG_DIR / "model_presets.json"
    with open(presets_file) as f:
        return json.load(f)


def show_preset_selector() -> str:
    """Show model preset selection screen."""
    presets = load_presets()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Select Base Model[/bold cyan]\n"
        "Choose a preset with optimized settings for your base model",
        border_style="cyan"
    ))
    console.print()

    # Build selection table
    table = Table(box=box.ROUNDED, border_style="cyan", show_header=True)
    table.add_column("Option", style="cyan", width=8)
    table.add_column("Model", style="yellow")
    table.add_column("Resolution", justify="right", style="green")
    table.add_column("Notes", style="dim")

    options = ["sdxl", "flux1", "zit", "sd15", "sd21", "custom"]

    for i, key in enumerate(options, 1):
        if key.startswith("_"):
            continue
        preset = presets[key]
        table.add_row(
            f"[{i}]",
            preset["name"],
            f"{preset['resolution']}px",
            preset["notes"][:50] + "..." if len(preset["notes"]) > 50 else preset["notes"]
        )

    console.print(table)
    console.print()

    choice = IntPrompt.ask(
        "Select model preset",
        choices=[str(i) for i in range(1, len(options) + 1)],
        default="1"
    )

    return options[choice - 1]


# ============================================================================
# Quality Warnings (Yellow Bubbles!)
# ============================================================================

def check_dataset_quality(metadata: Dict) -> List[str]:
    """Check dataset quality and return warnings."""
    warnings = []

    total_images = metadata.get("total_images", 0)
    shot_dist = metadata.get("shot_distribution", {})

    # Check total count
    if total_images < 15:
        warnings.append(f"Only {total_images} images (recommended: 15+)")
    elif total_images < 30:
        warnings.append(f"{total_images} images - acceptable but more is better (30+)")

    # Check shot distribution
    if total_images > 0:
        close_pct = (shot_dist.get("close", 0) / total_images) * 100
        mid_pct = (shot_dist.get("mid", 0) / total_images) * 100
        far_pct = (shot_dist.get("far", 0) / total_images) * 100

        if close_pct < 20:
            warnings.append(f"Low close-up shots ({close_pct:.0f}% - recommend 30-40%)")
        elif close_pct > 60:
            warnings.append(f"Too many close-ups ({close_pct:.0f}% - may limit versatility)")

        if mid_pct < 30:
            warnings.append(f"Low mid-range shots ({mid_pct:.0f}% - recommend 40-50%)")

        if far_pct < 15:
            warnings.append(f"Low full-body shots ({far_pct:.0f}% - recommend 20-30%)")

    # Check for unknown shots
    unknown = shot_dist.get("unknown", 0)
    if unknown > total_images * 0.3:
        warnings.append(f"{unknown} images couldn't be classified - provide FaceVault cache")

    return warnings


def show_quality_panel(metadata: Dict):
    """Show quality assessment panel with yellow bubbles."""
    warnings = check_dataset_quality(metadata)

    if not warnings:
        console.print(Panel(
            "[bold green]✓ Dataset Quality: Excellent[/bold green]\n"
            "All quality checks passed!",
            border_style="green",
            box=box.ROUNDED
        ))
        return

    # Build warning panel
    warning_text = "[bold yellow]⚠️  Dataset Quality Warnings[/bold yellow]\n\n"
    for warning in warnings:
        warning_text += f"  • {warning}\n"

    warning_text += "\n[dim]💡 Tip: Add more photos or adjust shot distribution for better results[/dim]"

    console.print(Panel(
        warning_text,
        border_style="yellow",
        box=box.ROUNDED
    ))


# ============================================================================
# Workflow Steps
# ============================================================================

def run_facevault_scan(photo_dir: str, session: Dict) -> bool:
    """Run FaceVault scan step."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 1/6: Scanning Photos[/bold cyan]\n"
        f"Directory: {photo_dir}",
        border_style="cyan"
    ))
    console.print()

    cmd = [
        sys.executable, "facevault.py", "scan", photo_dir,
        "--min-score", "0.6",
        "--min-size", "80"
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        session["steps"]["scan"] = {"status": "completed", "photo_dir": photo_dir}
        save_session(session)
        return True

    console.print("[red]✗ Scan failed[/red]")
    return False


def run_facevault_cluster(session: Dict) -> bool:
    """Run FaceVault cluster step."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 2/6: Clustering Faces[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    threshold = Prompt.ask("Clustering threshold", default="0.35")

    cmd = [
        sys.executable, "facevault.py", "cluster",
        "--threshold", threshold,
        "--preview"
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        session["steps"]["cluster"] = {"status": "completed", "threshold": threshold}
        save_session(session)
        return True

    console.print("[red]✗ Clustering failed[/red]")
    return False


def run_facevault_dedup(session: Dict) -> bool:
    """Run FaceVault dedup step."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 3/7: Removing Duplicates[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    cmd = [sys.executable, "facevault.py", "dedup"]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        session["steps"]["dedup"] = {"status": "completed"}
        save_session(session)
        return True

    console.print("[red]✗ Deduplication failed[/red]")
    return False


def run_facevault_label(session: Dict) -> bool:
    """Run FaceVault label step."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 4/7: Labeling People[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    cmd = [sys.executable, "facevault.py", "label", "--preview", "--preview-count", "5"]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        session["steps"]["label"] = {"status": "completed"}
        save_session(session)

        # Ask which person to process
        person_id = Prompt.ask("\nWhich person do you want to train a LoRA for?", default="person_001")
        person_name = Prompt.ask("What name did you give them?", default="Person")

        session["person_id"] = person_id
        session["person_name"] = person_name
        save_session(session)

        return True

    console.print("[red]✗ Labeling failed[/red]")
    return False


def run_caption_images(session: Dict) -> bool:
    """Run caption images step."""
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Step 5/7: Captioning Images[/bold cyan]\n"
        f"Person: {session.get('person_name', 'Unknown')}",
        border_style="cyan"
    ))
    console.print()

    person_id = session.get("person_id", "person_001")
    person_name = session.get("person_name", "person")

    input_dir = f"outputs/facevault/organized/{person_name}"

    console.print("[yellow]This may take a while depending on GPU and dataset size...[/yellow]\n")

    cmd = [
        sys.executable, "scripts/caption_images.py",
        "--dataset", "custom",
        "--input", input_dir,
        "--output-name", person_name.lower()
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        session["steps"]["caption"] = {
            "status": "completed",
            "person_name": person_name
        }
        save_session(session)
        return True

    console.print("[red]✗ Captioning failed[/red]")
    return False


def run_lora_prep(session: Dict) -> bool:
    """Run LoRA prep step."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 6/7: Preparing LoRA Dataset[/bold cyan]\n"
        "Shot classification + concept extraction",
        border_style="cyan"
    ))
    console.print()

    person_name = session.get("person_name", "person")
    input_dir = f"outputs/facevault/organized/{person_name}"

    cmd = [
        sys.executable, "lora_prep.py", "prepare", input_dir,
        "--name", person_name,
        "--facevault-cache", "outputs/facevault/face_cache.json",
        "--captions", f"outputs/processed/{person_name.lower()}/captions.json",
        "--taxonomy", "configs/taxonomy.json"
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        session["steps"]["prep"] = {"status": "completed"}
        save_session(session)

        # Load metadata and show quality assessment
        console.print()
        metadata_file = Path(f"outputs/lora_datasets/{person_name.lower()}/metadata.json")
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

            show_quality_panel(metadata)
            console.print()

            if not Confirm.ask("Continue to training configuration?", default=True):
                return False

        return True

    console.print("[red]✗ Dataset preparation failed[/red]")
    return False


def run_lora_train(session: Dict, preset_key: str) -> bool:
    """Run LoRA train configuration step."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Step 7/7: Configuring Training[/bold cyan]\n"
        f"Model preset: {preset_key.upper()}",
        border_style="cyan"
    ))
    console.print()

    person_name = session.get("person_name", "person")
    dataset_path = f"outputs/lora_datasets/{person_name.lower()}"

    # Load preset
    presets = load_presets()
    preset = presets[preset_key]

    # Show preset details
    preset_table = Table(show_header=False, box=box.SIMPLE)
    preset_table.add_row("Base Model", preset["base_model"])
    preset_table.add_row("Resolution", f"{preset['resolution']}px")
    preset_table.add_row("Learning Rate", f"{preset['learning_rate']:.0e}")
    preset_table.add_row("Network Dim", str(preset["network_dim"]))
    preset_table.add_row("Network Alpha", str(preset["network_alpha"]))

    console.print(Panel(preset_table, title=f"[bold]{preset['name']} Settings[/bold]", border_style="cyan"))
    console.print()

    if not Confirm.ask("Use these settings?", default=True):
        console.print("[yellow]You can customize in interactive mode...[/yellow]")

    # Build command
    cmd = [
        sys.executable, "lora_train.py", "train", dataset_path,
        "--base-model", preset["base_model"],
        "--resolution", str(preset["resolution"]),
        "--learning-rate", str(preset["learning_rate"]),
        "--network-dim", str(preset["network_dim"]),
        "--network-alpha", str(preset["network_alpha"])
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        session["steps"]["train"] = {
            "status": "completed",
            "preset": preset_key
        }
        session["completed"] = True
        save_session(session)
        return True

    console.print("[red]✗ Training configuration failed[/red]")
    return False


# ============================================================================
# Main Workflows
# ============================================================================

def workflow_lora_training():
    """Complete LoRA training pipeline."""

    # Welcome screen
    console.clear()
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🦫 Eager Beaver - LoRA Training Pipeline[/bold cyan]\n\n"
        "Complete workflow from photos to trained LoRA model",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Initialize variables
    session = None
    photo_dir = None
    preset_key = None

    # Check for existing sessions
    sessions = list_sessions()
    if sessions:
        console.print(f"[yellow]Found {len(sessions)} saved session(s)[/yellow]")
        if Confirm.ask("Resume previous session?", default=False):
            # Show sessions
            table = Table(box=box.ROUNDED)
            table.add_column("#", style="cyan")
            table.add_column("Person", style="yellow")
            table.add_column("Last Step", style="green")
            table.add_column("Date", style="dim")

            for i, sess in enumerate(sessions[:5], 1):
                last_step = "Unknown"
                for step in ["train", "prep", "caption", "label", "cluster", "scan"]:
                    if sess.get("steps", {}).get(step, {}).get("status") == "completed":
                        last_step = step.capitalize()
                        break

                table.add_row(
                    str(i),
                    sess.get("person_name", "Unknown"),
                    last_step,
                    sess.get("last_updated", "")[:10]
                )

            console.print(table)
            console.print()

            choice = IntPrompt.ask(
                "Select session (0 for new)",
                choices=[str(i) for i in range(0, len(sessions) + 1)],
                default="0"
            )

            if choice > 0 and choice <= len(sessions):
                session = sessions[choice - 1]
                console.print(f"[green]✓ Resuming session: {session.get('person_name', 'Unknown')}[/green]")

                # Load and validate session data
                photo_dir = session.get("steps", {}).get("scan", {}).get("photo_dir", "")
                preset_key = session.get("preset", "sdxl")

                # Validate saved directory still exists
                if photo_dir and not Path(photo_dir).exists():
                    console.print(f"[yellow]⚠️  Saved directory not found: {photo_dir}[/yellow]")
                    console.print("[dim]Please provide a new directory[/dim]")
                    photo_dir = None

                # Prompt for directory if needed
                if not photo_dir:
                    photo_dir = Prompt.ask(
                        "\n📂 Photo directory to process",
                        default="./photos"
                    )
                    if not Path(photo_dir).exists():
                        console.print(f"[red]✗ Directory not found: {photo_dir}[/red]")
                        console.print("[dim]Hint: Use absolute path like /home/user/photos or relative like ./my_photos[/dim]")
                        return
                    # Update session with new directory
                    if "steps" not in session:
                        session["steps"] = {}
                    if "scan" not in session["steps"]:
                        session["steps"]["scan"] = {}
                    session["steps"]["scan"]["photo_dir"] = photo_dir
                    save_session(session)
                else:
                    console.print(f"[dim]Photo dir: {photo_dir}[/dim]")

                console.print(f"[dim]Preset: {preset_key.upper()}[/dim]")
                console.print()

                # Continue to workflow execution (skip completed steps)

    # If no session was loaded, start new
    if session is None:
        session = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "workflow": "lora_training",
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "steps": {}
        }

        # Get photo directory with helpful prompt
        console.print("\n[cyan]Photo Directory[/cyan]")
        console.print("[dim]Examples: ./photos, /home/user/my_photos, ~/Pictures/portraits[/dim]")
        photo_dir = Prompt.ask("📂 Directory path", default="./photos")

        if not Path(photo_dir).exists():
            console.print(f"[red]✗ Directory not found: {photo_dir}[/red]")
            console.print("[dim]Hint: Use 'ls' to see available directories, or provide absolute path[/dim]")
            return

        # Select model preset
        preset_key = show_preset_selector()
        session["preset"] = preset_key
        save_session(session)

    console.print()
    console.print(f"[green]✓ Using {preset_key.upper()} preset[/green]")
    console.print()

    # Execute workflow steps
    steps = [
        ("scan", lambda: run_facevault_scan(photo_dir, session)),
        ("cluster", lambda: run_facevault_cluster(session)),
        ("dedup", lambda: run_facevault_dedup(session)),
        ("label", lambda: run_facevault_label(session)),
        ("caption", lambda: run_caption_images(session)),
        ("prep", lambda: run_lora_prep(session)),
        ("train", lambda: run_lora_train(session, preset_key)),
    ]

    for step_name, step_func in steps:
        # Skip already completed steps when resuming
        step_status = session.get("steps", {}).get(step_name, {}).get("status")
        if step_status == "completed":
            console.print(f"[dim]⏭️  Skipping {step_name} (already completed)[/dim]")
            continue

        if not step_func():
            console.print(f"\n[red]Workflow stopped at: {step_name}[/red]")
            console.print(f"[yellow]Session saved. Resume later with: beaver.py[/yellow]")
            return

    # Success!
    console.print()
    console.print(Panel.fit(
        "[bold green]🎉 LoRA Training Pipeline Complete![/bold green]\n\n"
        f"Person: {session.get('person_name', 'Unknown')}\n"
        f"Preset: {preset_key.upper()}\n"
        f"Dataset: outputs/lora_datasets/{session.get('person_name', 'person').lower()}/\n"
        f"Configs: outputs/lora_models/{session.get('person_name', 'person').lower()}/\n\n"
        "[dim]Next: Edit train.sh and run training![/dim]",
        border_style="green",
        box=box.DOUBLE
    ))


def workflow_photo_organization():
    """Photo organization only (FaceVault)."""
    console.clear()
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🗂️  FaceVault - Photo Organization[/bold cyan]\n\n"
        "Organize photos by person with face detection",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    console.print("[cyan]Photo Directory[/cyan]")
    console.print("[dim]Examples: ./photos, /home/user/my_photos, ~/Pictures/christmas_2024[/dim]")
    photo_dir = Prompt.ask("📂 Directory path", default="./photos")

    if not Path(photo_dir).exists():
        console.print(f"[red]✗ Directory not found: {photo_dir}[/red]")
        console.print("[dim]Hint: Use 'ls' to see available directories, or provide absolute path[/dim]")
        return

    # Simple 3-step flow
    steps = [
        ("Scan", f"{sys.executable} facevault.py scan {photo_dir}"),
        ("Cluster", f"{sys.executable} facevault.py cluster --preview"),
        ("Label", f"{sys.executable} facevault.py label"),
    ]

    for step_name, cmd in steps:
        console.print()
        console.print(Panel.fit(f"[bold cyan]{step_name}[/bold cyan]", border_style="cyan"))
        console.print()
        subprocess.run(cmd, shell=True)

    console.print()
    console.print("[green]✓ Photo organization complete![/green]")
    console.print("[dim]Export with: python facevault.py export --all --format organized[/dim]")


# ============================================================================
# Main CLI
# ============================================================================

@click.command()
@click.option('--workflow', type=click.Choice(['lora', 'photos', 'menu']), default='menu', help='Workflow to run')
def main(workflow: str):
    """
    🦫 Eager Beaver - Beautiful TUI wizard for LoRA training.

    Guided workflows with smart defaults and model presets.
    """

    if workflow == "menu" or workflow is None:
        # Show main menu
        console.clear()
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🦫 Eager Beaver[/bold cyan]\n\n"
            "Complete toolkit for vision AI workflows",
            border_style="cyan",
            box=box.DOUBLE
        ))
        console.print()

        table = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
        table.add_column("Option", style="cyan", width=6)
        table.add_column("Workflow", style="yellow")
        table.add_column("Description", style="dim")

        table.add_row("[1]", "LoRA Training", "Complete pipeline: photos → trained LoRA")
        table.add_row("[2]", "Photo Organization", "FaceVault only (organize by person)")
        table.add_row("[3]", "Individual Tools", "Run tools separately")

        console.print(table)
        console.print()

        choice = IntPrompt.ask("Select workflow", choices=["1", "2", "3"], default="1")

        if choice == 1:
            workflow = "lora"
        elif choice == 2:
            workflow = "photos"
        else:
            console.print("\n[cyan]Individual tools:[/cyan]")
            console.print("  python facevault.py --help")
            console.print("  python lora_prep.py --help")
            console.print("  python lora_train.py --help")
            return

    # Execute selected workflow
    if workflow == "lora":
        workflow_lora_training()
    elif workflow == "photos":
        workflow_photo_organization()


if __name__ == "__main__":
    main()
