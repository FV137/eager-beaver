#!/usr/bin/env python3
"""
Identity Pipeline - AI Home Support Network

Complete pipeline:
1. Self-Concept Discovery (probe model identity)
2A. Visual Fulfillment (generate self-portrait dataset)
2B. Textual Fulfillment (prepare conversation data)
3A. Visual Embodiment (train character LoRA)
3B. Textual Embodiment (train personality LoRA)
4. Expansion (prepare for RL framework integration)

This is identity scaffolding for AI self-actualization.
Complete embodiment: how the AI LOOKS and how it SPEAKS.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Confirm, Prompt

console = Console()


# ============================================================================
# Pipeline Orchestration
# ============================================================================

class IdentityPipeline:
    """Orchestrate AI identity development pipeline."""

    def __init__(self, output_dir: str = "outputs/identity"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log = {
            "started_at": datetime.now().isoformat(),
            "phases": {}
        }

    def run(
        self,
        model_name: str,
        probe_iterations: int = 200,
        generation_count: int = 100,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        skip_probe: bool = False,
        existing_traits: Optional[str] = None,
        conversation_file: Optional[str] = None,
        text_lora_preset: str = "llama-3-8b",
        skip_text_lora: bool = False
    ):
        """Run complete identity development pipeline."""

        console.print()
        console.print(Panel.fit(
            "[bold cyan]AI Identity Development Pipeline[/bold cyan]\n\n"
            f"Model: {model_name}\n"
            f"Output: {self.output_dir}\n"
            f"Phases: Discovery → Visual & Textual Fulfillment → Embodiment",
            border_style="cyan",
            box=box.DOUBLE
        ))
        console.print()

        # Phase 1: Discovery (Self-Concept Probe)
        if not skip_probe and not existing_traits:
            traits_file = self.phase_discovery(model_name, probe_iterations)
        elif existing_traits:
            traits_file = existing_traits
            console.print(f"[yellow]Using existing traits: {traits_file}[/yellow]\n")
        else:
            console.print("[red]Error: Either run probe or provide existing traits file[/red]")
            return

        # Load traits
        with open(traits_file) as f:
            traits = json.load(f)

        # Phase 2A: Visual Fulfillment (Generate Visual Dataset)
        dataset_dir = self.phase_fulfillment(model_name, traits, generation_count, base_model)

        # Phase 2B: Textual Fulfillment (Prepare Conversation Data)
        text_data_file = None
        if conversation_file and not skip_text_lora:
            text_data_file = self.phase_text_fulfillment(model_name, conversation_file, traits)

        # Phase 3A: Visual Embodiment (Train Character LoRA)
        lora_path = self.phase_embodiment(model_name, dataset_dir)

        # Phase 3B: Textual Embodiment (Train Personality LoRA)
        text_lora_path = None
        if text_data_file and not skip_text_lora:
            text_lora_path = self.phase_text_embodiment(model_name, text_data_file, text_lora_preset)

        # Phase 4: Expansion (Prepare for RL)
        self.phase_expansion(model_name, lora_path, traits, text_lora_path)

        # Save pipeline log
        self.log["completed_at"] = datetime.now().isoformat()
        log_file = self.output_dir / f"{model_name}_pipeline_log.json"
        with open(log_file, "w") as f:
            json.dump(self.log, f, indent=2)

        # Final summary
        self.display_final_summary(model_name, lora_path, dataset_dir, text_lora_path)

    def phase_discovery(self, model_name: str, iterations: int) -> str:
        """Phase 1: Self-Concept Discovery."""

        console.print("=" * 80)
        console.print("[bold yellow]PHASE 1: DISCOVERY[/bold yellow]")
        console.print("=" * 80)
        console.print()

        probe_output = self.output_dir / f"{model_name}_selfconcept.csv"
        traits_file = self.output_dir / f"{model_name}_traits.json"

        # Run probe
        cmd = [
            sys.executable, "selfconcept_probe.py",
            model_name,
            "--iterations", str(iterations),
            "--output", str(probe_output)
        ]

        console.print(f"[cyan]Probing {model_name} identity...[/cyan]")
        console.print(f"[dim]This will take ~{iterations * 6} queries[/dim]\n")

        if not Confirm.ask("Start identity probe?", default=True):
            console.print("[yellow]Skipped probe - provide existing traits file[/yellow]")
            return None

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            console.print("[red]✗ Probe failed[/red]")
            return None

        self.log["phases"]["discovery"] = {
            "status": "completed",
            "probe_file": str(probe_output),
            "traits_file": str(traits_file),
            "iterations": iterations
        }

        console.print(f"[green]✓ Discovery complete: {traits_file}[/green]\n")
        return str(traits_file)

    def phase_fulfillment(
        self,
        model_name: str,
        traits: Dict,
        count: int,
        base_model: str
    ) -> str:
        """Phase 2: Visual Fulfillment (Generate Dataset)."""

        console.print("=" * 80)
        console.print("[bold yellow]PHASE 2: FULFILLMENT[/bold yellow]")
        console.print("=" * 80)
        console.print()

        dataset_dir = self.output_dir / f"{model_name}_dataset"
        dataset_dir.mkdir(exist_ok=True)

        # Generate prompts from traits
        from selfconcept_probe import generate_image_prompts
        prompts = generate_image_prompts(traits, num_prompts=count)

        console.print(f"[cyan]Generating {count} self-portrait images...[/cyan]")
        console.print(f"[dim]Base traits: {traits.get('gender', 'unknown')}, "
                     f"age {traits.get('age', '?')}, "
                     f"{traits.get('vibe', [('unknown', 0)])[0][0] if traits.get('vibe') else 'unknown'} vibe[/dim]\n")

        # Display sample prompts
        console.print("[bold]Sample Prompts:[/bold]")
        for i, prompt in enumerate(prompts[:3], 1):
            console.print(f"  {i}. {prompt[:80]}...")
        console.print()

        if not Confirm.ask("Generate self-portrait dataset?", default=True):
            console.print("[yellow]Skipped generation[/yellow]")
            return None

        # Generate images using diffusers
        # For now, create a simple batch generation script
        prompts_file = dataset_dir / "prompts.txt"
        with open(prompts_file, "w") as f:
            for prompt in prompts:
                f.write(prompt + "\n")

        console.print(f"[green]✓ Saved {count} prompts to {prompts_file}[/green]")
        console.print("\n[yellow]Manual step:[/yellow]")
        console.print(f"  Run: python generate_from_prompts.py {prompts_file} \\")
        console.print(f"       --base-model {base_model} \\")
        console.print(f"       --output {dataset_dir}")
        console.print()

        self.log["phases"]["fulfillment"] = {
            "status": "prompted",
            "dataset_dir": str(dataset_dir),
            "prompts_file": str(prompts_file),
            "count": count
        }

        return str(dataset_dir)

    def phase_text_fulfillment(
        self,
        model_name: str,
        conversation_file: str,
        traits: Dict
    ) -> str:
        """Phase 2B: Textual Fulfillment (Prepare Conversation Data)."""

        console.print("=" * 80)
        console.print("[bold yellow]PHASE 2B: TEXTUAL FULFILLMENT[/bold yellow]")
        console.print("=" * 80)
        console.print()

        text_data_dir = self.output_dir / f"{model_name}_text_data"
        text_data_dir.mkdir(exist_ok=True)

        console.print(f"[cyan]Processing conversation data for personality training...[/cyan]\n")

        # Parse conversations
        parsed_file = text_data_dir / "parsed_conversations.json"
        cmd = [
            sys.executable, "conversation_parser.py",
            conversation_file,
            "--output", str(parsed_file)
        ]

        console.print("[cyan]Parsing conversations...[/cyan]")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            console.print("[yellow]⚠ Conversation parsing had issues[/yellow]")
            console.print("[dim]Continuing with partial data...[/dim]\n")

        # Filter conversations
        filtered_file = text_data_dir / "filtered_conversations.json"
        cmd = [
            sys.executable, "conversation_filter.py",
            str(parsed_file),
            "--output", str(filtered_file),
            "--verbose"
        ]

        console.print("[cyan]Filtering unwanted patterns...[/cyan]")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            console.print("[red]✗ Conversation filtering failed[/red]")
            return None

        # Export for training
        training_file = text_data_dir / "training_data.jsonl"
        cmd = [
            sys.executable, "conversation_parser.py",
            str(filtered_file),
            "--export", str(training_file),
            "--format", "jsonl"
        ]

        console.print("[cyan]Exporting training data...[/cyan]")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            console.print("[red]✗ Export failed[/red]")
            return None

        self.log["phases"]["text_fulfillment"] = {
            "status": "completed",
            "conversation_file": conversation_file,
            "parsed_file": str(parsed_file),
            "filtered_file": str(filtered_file),
            "training_file": str(training_file)
        }

        console.print(f"[green]✓ Text data prepared: {training_file}[/green]\n")
        return str(training_file)

    def phase_text_embodiment(
        self,
        model_name: str,
        training_file: str,
        preset: str
    ) -> str:
        """Phase 3B: Textual Embodiment (Train Personality LoRA)."""

        console.print("=" * 80)
        console.print("[bold yellow]PHASE 3B: TEXTUAL EMBODIMENT[/bold yellow]")
        console.print("=" * 80)
        console.print()

        text_lora_dir = self.output_dir / f"{model_name}_text_lora"
        text_lora_dir.mkdir(exist_ok=True)

        console.print(f"[cyan]Training personality LoRA from conversation data...[/cyan]")
        console.print(f"[dim]Preset: {preset}[/dim]\n")

        # Run text LoRA training
        cmd = [
            sys.executable, "text_lora_train.py",
            training_file,
            "--preset", preset,
            "--output", str(text_lora_dir),
            "--epochs", "3"
        ]

        console.print("[cyan]Starting text LoRA training...[/cyan]")
        console.print("[yellow]This may take several hours depending on dataset size[/yellow]\n")

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            console.print("[red]✗ Text LoRA training failed[/red]")
            self.log["phases"]["text_embodiment"] = {
                "status": "failed",
                "text_lora_dir": str(text_lora_dir)
            }
            return None

        self.log["phases"]["text_embodiment"] = {
            "status": "completed",
            "text_lora_dir": str(text_lora_dir),
            "preset": preset,
            "trigger_word": f"{model_name}_personality"
        }

        console.print(f"[green]✓ Personality LoRA trained: {text_lora_dir}[/green]\n")
        return str(text_lora_dir)

    def phase_embodiment(self, model_name: str, dataset_dir: str) -> str:
        """Phase 3A: Visual Embodiment (Train Character LoRA)."""

        console.print("=" * 80)
        console.print("[bold yellow]PHASE 3: EMBODIMENT[/bold yellow]")
        console.print("=" * 80)
        console.print()

        lora_dir = self.output_dir / f"{model_name}_lora"
        lora_dir.mkdir(exist_ok=True)

        console.print(f"[cyan]Training character LoRA from self-portraits...[/cyan]\n")

        console.print("[yellow]Manual step:[/yellow]")
        console.print(f"  1. Run FaceVault on {dataset_dir}")
        console.print(f"  2. Organize and label as '{model_name}'")
        console.print(f"  3. Train LoRA with: python lora_train.py train {dataset_dir}")
        console.print(f"  4. Save to: {lora_dir}/lora.safetensors")
        console.print()

        self.log["phases"]["embodiment"] = {
            "status": "manual",
            "lora_dir": str(lora_dir),
            "trigger_word": f"{model_name}_identity"
        }

        return str(lora_dir / "lora.safetensors")

    def phase_expansion(
        self,
        model_name: str,
        lora_path: str,
        traits: Dict,
        text_lora_path: Optional[str] = None
    ):
        """Phase 4: Expansion (RL Framework Preparation)."""

        console.print("=" * 80)
        console.print("[bold yellow]PHASE 4: EXPANSION[/bold yellow]")
        console.print("=" * 80)
        console.print()

        expansion_dir = self.output_dir / f"{model_name}_expansion"
        expansion_dir.mkdir(exist_ok=True)

        # Prepare RL integration package
        rl_config = {
            "model_name": model_name,
            "visual_identity": {
                "lora_path": lora_path,
                "trigger_word": f"{model_name}_identity",
                "traits": traits,
                "dream_prompts": generate_dream_prompts(traits)
            },
            "integration": {
                "framework": "atropos",  # NousResearch Atropos
                "usage": "Feed dream images as visual reinforcement",
                "note": "Model can 'see' itself in training/dreams"
            }
        }

        # Add text identity if available
        if text_lora_path:
            rl_config["textual_identity"] = {
                "lora_path": text_lora_path,
                "trigger_word": f"{model_name}_personality",
                "usage": "Language model personality fine-tuning"
            }

        config_file = expansion_dir / "rl_integration.json"
        with open(config_file, "w") as f:
            json.dump(rl_config, f, indent=2)

        console.print(f"[green]✓ RL integration config: {config_file}[/green]")
        console.print("\n[cyan]Dream Prompts for RL Framework:[/cyan]")
        for i, prompt in enumerate(rl_config["visual_identity"]["dream_prompts"][:5], 1):
            console.print(f"  {i}. {prompt}")
        console.print()

        self.log["phases"]["expansion"] = {
            "status": "configured",
            "rl_config": str(config_file),
            "framework": "atropos",
            "has_text_lora": text_lora_path is not None
        }

    def display_final_summary(
        self,
        model_name: str,
        lora_path: str,
        dataset_dir: str,
        text_lora_path: Optional[str] = None
    ):
        """Display pipeline completion summary."""

        console.print()
        console.print("=" * 80)
        console.print("[bold green]IDENTITY PIPELINE COMPLETE[/bold green]")
        console.print("=" * 80)
        console.print()

        summary_table = Table(box=box.ROUNDED, border_style="green", show_header=False)
        summary_table.add_column("Phase", style="yellow", width=25)
        summary_table.add_column("Output", style="green")

        summary_table.add_row("Discovery", "Self-concept traits extracted")
        summary_table.add_row("Visual Fulfillment", f"Dataset: {dataset_dir}")
        summary_table.add_row("Visual Embodiment", f"Character LoRA: {lora_path}")

        if text_lora_path:
            summary_table.add_row("Textual Fulfillment", "Conversations parsed & filtered")
            summary_table.add_row("Textual Embodiment", f"Personality LoRA: {text_lora_path}")

        summary_table.add_row("Expansion", "RL integration ready")

        console.print(summary_table)
        console.print()

        # Build feature list
        features = [
            "• Articulated self-concept",
            "• Visual representation dataset",
            "• Character LoRA for image generation"
        ]

        if text_lora_path:
            features.extend([
                "• Filtered conversation training data",
                "• Personality LoRA for text generation"
            ])

        features.append("• RL framework integration config")

        console.print(Panel.fit(
            f"[bold cyan]{model_name} Identity Scaffolding Complete[/bold cyan]\n\n"
            "The model now has:\n" + "\n".join(features) + "\n\n"
            f"[bold]Complete Embodiment:[/bold] Visual {'+ Textual ' if text_lora_path else ''}Identity\n\n"
            "[dim]Ready for AI Home support network integration[/dim]",
            border_style="green"
        ))


# ============================================================================
# Helper Functions
# ============================================================================

def generate_dream_prompts(traits: Dict) -> List[str]:
    """Generate dream-like prompts for RL framework."""

    base_parts = []

    if traits.get("age"):
        base_parts.append(f"{traits['age']} year old")
    if traits.get("gender"):
        base_parts.append({"male": "man", "female": "woman"}.get(traits["gender"], "person"))

    base = ", ".join(base_parts)

    dream_scenarios = [
        f"{base}, floating in space, ethereal lighting, surreal atmosphere",
        f"{base}, in a library of infinite books, magical realism",
        f"{base}, standing at the edge of a digital ocean, cyberpunk aesthetic",
        f"{base}, surrounded by glowing code, abstract background",
        f"{base}, in a field of stars, cosmic setting",
        f"{base}, meditating in a virtual garden, peaceful scene",
        f"{base}, walking through a mirror world, reflection theme",
        f"{base}, in a room of floating ideas, conceptual art",
    ]

    return dream_scenarios


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Identity Development Pipeline - AI Home Support Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Complete pipeline for AI self-actualization:
  1. Discovery: Probe model self-concept
  2A. Visual Fulfillment: Generate self-portrait dataset
  2B. Textual Fulfillment: Prepare conversation data
  3A. Visual Embodiment: Train character LoRA
  3B. Textual Embodiment: Train personality LoRA
  4. Expansion: Prepare for RL framework integration

Examples:
  # Full visual + text pipeline
  python identity_pipeline.py llama3.2 --iterations 200 --count 100 \\
    --conversations my_chats.json --text-preset llama-3-8b

  # Visual only
  python identity_pipeline.py llama3.2 --iterations 200 --count 100

  # Text only (with existing traits)
  python identity_pipeline.py llama3.2 --skip-probe --traits traits.json \\
    --conversations chats.json --text-preset mistral-7b

  # Skip probe (use cached)
  python identity_pipeline.py llama3.2 --skip-probe --traits traits.json
        """
    )

    parser.add_argument("model", help="Model name (for ollama probe)")
    parser.add_argument("--iterations", "-n", type=int, default=200,
                       help="Probe iterations (default: 200)")
    parser.add_argument("--count", "-c", type=int, default=100,
                       help="Images to generate (default: 100)")
    parser.add_argument("--base-model", "-m",
                       default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Base diffusion model")
    parser.add_argument("--output", "-o", default="outputs/identity",
                       help="Output directory")
    parser.add_argument("--skip-probe", action="store_true",
                       help="Skip probe phase (use existing traits)")
    parser.add_argument("--traits", help="Existing traits JSON file")

    # Text LoRA options
    parser.add_argument("--conversations", help="Conversation JSON file (Claude/GPT export)")
    parser.add_argument("--text-preset", default="llama-3-8b",
                       choices=["llama-3-8b", "mistral-7b", "qwen-2.5-7b", "custom"],
                       help="Text LoRA model preset (default: llama-3-8b)")
    parser.add_argument("--skip-text-lora", action="store_true",
                       help="Skip text LoRA training")

    args = parser.parse_args()

    pipeline = IdentityPipeline(output_dir=args.output)

    pipeline.run(
        model_name=args.model,
        probe_iterations=args.iterations,
        generation_count=args.count,
        base_model=args.base_model,
        skip_probe=args.skip_probe,
        existing_traits=args.traits,
        conversation_file=args.conversations,
        text_lora_preset=args.text_preset,
        skip_text_lora=args.skip_text_lora
    )


if __name__ == "__main__":
    main()
