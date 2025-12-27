#!/usr/bin/env python3
"""
Text LoRA Training - Personality fine-tuning for language models.

Trains LoRA adapters from filtered conversation data to capture:
- Speaking style and personality
- Domain expertise patterns
- Conversational patterns
- Identity-aligned responses

Integrates with self-concept probe and conversation filter for
complete textual identity scaffolding.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.prompt import Confirm, Prompt, IntPrompt
from rich import box

console = Console()

# Try to import training libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    console.print("[dim]Unsloth not available - will use PEFT fallback[/dim]")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ============================================================================
# Training Configuration Presets
# ============================================================================

TEXT_LORA_PRESETS = {
    "llama-3-8b": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "learning_rate": 2e-4,
        "batch_size": 4,
        "max_seq_length": 2048,
    },
    "mistral-7b": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "learning_rate": 2e-4,
        "batch_size": 4,
        "max_seq_length": 2048,
    },
    "qwen-2.5-7b": {
        "base_model": "Qwen/Qwen2.5-7B",
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "learning_rate": 2e-4,
        "batch_size": 4,
        "max_seq_length": 2048,
    },
    "custom": {
        "base_model": "",
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "learning_rate": 2e-4,
        "batch_size": 4,
        "max_seq_length": 2048,
    }
}


# ============================================================================
# Data Loading
# ============================================================================

def load_training_data(jsonl_path: str) -> List[Dict]:
    """Load training data from JSONL export."""
    data = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data.append(entry)

    return data


def format_conversation_pair(entry: Dict, format_template: str = "alpaca") -> Dict:
    """
    Format conversation entry for training.

    Templates:
    - alpaca: instruction/input/output format
    - chat: messages format
    - plain: simple prompt/response
    """

    if format_template == "alpaca":
        return {
            "instruction": entry.get("instruction", ""),
            "input": "",
            "output": entry.get("response", "")
        }

    elif format_template == "chat":
        return {
            "messages": [
                {"role": "user", "content": entry.get("instruction", "")},
                {"role": "assistant", "content": entry.get("response", "")}
            ]
        }

    else:  # plain
        return {
            "prompt": entry.get("instruction", ""),
            "response": entry.get("response", "")
        }


def prepare_dataset(
    jsonl_path: str,
    format_template: str = "alpaca",
    max_samples: Optional[int] = None
) -> List[Dict]:
    """Prepare dataset for training."""

    console.print(f"[cyan]Loading training data from {jsonl_path}[/cyan]")

    raw_data = load_training_data(jsonl_path)
    console.print(f"[green]✓ Loaded {len(raw_data)} conversation pairs[/green]")

    # Format for training
    formatted_data = []
    for entry in raw_data[:max_samples] if max_samples else raw_data:
        formatted = format_conversation_pair(entry, format_template)
        formatted_data.append(formatted)

    console.print(f"[green]✓ Formatted {len(formatted_data)} training samples[/green]\\n")

    return formatted_data


# ============================================================================
# Unsloth Training Path
# ============================================================================

def train_with_unsloth(
    dataset: List[Dict],
    config: Dict,
    output_dir: str,
    num_epochs: int = 3
):
    """Train using Unsloth (optimized for speed)."""

    if not UNSLOTH_AVAILABLE:
        console.print("[red]Error: Unsloth not installed[/red]")
        console.print("[dim]Install with: pip install unsloth[/dim]")
        return False

    console.print(Panel.fit(
        "[bold cyan]Training with Unsloth[/bold cyan]\\n"
        f"Base Model: {config['base_model']}\\n"
        f"LoRA Rank: {config['lora_r']}\\n"
        f"Samples: {len(dataset)}",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    try:
        # Load model with Unsloth
        console.print("[cyan]Loading model...[/cyan]")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config['base_model'],
            max_seq_length=config['max_seq_length'],
            dtype=None,
            load_in_4bit=True,
        )

        # Add LoRA adapters
        console.print("[cyan]Adding LoRA adapters...[/cyan]")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=config['target_modules'],
            bias="none",
            use_gradient_checkpointing=True,
        )

        # Format dataset for Unsloth
        def formatting_func(example):
            instruction = example.get("instruction", "")
            output = example.get("output", "")

            text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
            return text

        # Training arguments
        from transformers import TrainingArguments
        from trl import SFTTrainer

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=config['batch_size'],
            gradient_accumulation_steps=4,
            learning_rate=config['learning_rate'],
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            optim="adamw_8bit",
            warmup_steps=50,
            report_to="none",
        )

        # Create trainer
        console.print("[cyan]Initializing trainer...[/cyan]")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config['max_seq_length'],
            formatting_func=formatting_func,
            args=training_args,
        )

        # Train
        console.print()
        console.print("[bold green]Starting training...[/bold green]")
        console.print()

        trainer.train()

        # Save LoRA weights
        console.print()
        console.print("[cyan]Saving LoRA weights...[/cyan]")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        console.print(f"[green]✓ Training complete! Saved to {output_dir}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]✗ Training failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PEFT Training Path (Fallback)
# ============================================================================

def train_with_peft(
    dataset: List[Dict],
    config: Dict,
    output_dir: str,
    num_epochs: int = 3
):
    """Train using PEFT/Transformers (fallback if no Unsloth)."""

    if not PEFT_AVAILABLE:
        console.print("[red]Error: transformers/peft not installed[/red]")
        console.print("[dim]Install with: pip install transformers peft trl[/dim]")
        return False

    console.print(Panel.fit(
        "[bold cyan]Training with PEFT[/bold cyan]\\n"
        f"Base Model: {config['base_model']}\\n"
        f"LoRA Rank: {config['lora_r']}\\n"
        f"Samples: {len(dataset)}",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    try:
        # Load model
        console.print("[cyan]Loading model...[/cyan]")
        model = AutoModelForCausalLM.from_pretrained(
            config['base_model'],
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
        tokenizer.pad_token = tokenizer.eos_token

        # Prepare for training
        model = prepare_model_for_kbit_training(model)

        # LoRA config
        console.print("[cyan]Configuring LoRA...[/cyan]")
        peft_config = LoraConfig(
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=config['target_modules'],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=config['batch_size'],
            gradient_accumulation_steps=4,
            learning_rate=config['learning_rate'],
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            optim="paged_adamw_8bit",
            warmup_steps=50,
            report_to="none",
        )

        # Format dataset
        def formatting_func(example):
            instruction = example.get("instruction", "")
            output = example.get("output", "")

            text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
            return text

        # Create trainer
        console.print("[cyan]Initializing trainer...[/cyan]")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config['max_seq_length'],
            formatting_func=formatting_func,
            args=training_args,
        )

        # Train
        console.print()
        console.print("[bold green]Starting training...[/bold green]")
        console.print()

        trainer.train()

        # Save
        console.print()
        console.print("[cyan]Saving LoRA weights...[/cyan]")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        console.print(f"[green]✓ Training complete! Saved to {output_dir}[/green]")
        return True

    except Exception as e:
        console.print(f"[red]✗ Training failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_text_lora(
    data_file: str,
    output_dir: str,
    preset: str = "llama-3-8b",
    custom_config: Optional[Dict] = None,
    num_epochs: int = 3,
    use_unsloth: bool = True
):
    """
    Complete text LoRA training pipeline.

    Args:
        data_file: Path to filtered JSONL conversation data
        output_dir: Where to save the trained LoRA
        preset: Model preset name or "custom"
        custom_config: Custom config dict (if preset="custom")
        num_epochs: Training epochs
        use_unsloth: Use Unsloth if available (faster)
    """

    # Check dependencies
    if not TORCH_AVAILABLE:
        console.print("[red]Error: PyTorch not installed[/red]")
        console.print("[dim]Install with: pip install torch[/dim]")
        return False

    # Load config
    if preset in TEXT_LORA_PRESETS:
        config = TEXT_LORA_PRESETS[preset].copy()
    elif custom_config:
        config = custom_config
    else:
        console.print(f"[red]Error: Unknown preset '{preset}' and no custom config provided[/red]")
        return False

    # Prepare dataset
    dataset = prepare_dataset(data_file, format_template="alpaca")

    if len(dataset) == 0:
        console.print("[red]Error: No training data loaded[/red]")
        return False

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save training config
    config_file = output_path / "training_config.json"
    with open(config_file, "w") as f:
        json.dump({
            "preset": preset,
            "config": config,
            "num_epochs": num_epochs,
            "dataset_size": len(dataset),
            "data_file": str(data_file),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Choose training method
    if use_unsloth and UNSLOTH_AVAILABLE:
        success = train_with_unsloth(dataset, config, str(output_path), num_epochs)
    elif PEFT_AVAILABLE:
        success = train_with_peft(dataset, config, str(output_path), num_epochs)
    else:
        console.print("[red]Error: No training libraries available[/red]")
        console.print("[dim]Install either unsloth or transformers+peft+trl[/dim]")
        return False

    return success


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train text LoRA from filtered conversation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Llama-3 personality LoRA
  python text_lora_train.py data.jsonl --preset llama-3-8b --output ./lora_output

  # Train Mistral with custom epochs
  python text_lora_train.py data.jsonl --preset mistral-7b --epochs 5

  # Use PEFT instead of Unsloth
  python text_lora_train.py data.jsonl --preset llama-3-8b --no-unsloth

  # List available presets
  python text_lora_train.py --list-presets

Presets: llama-3-8b, mistral-7b, qwen-2.5-7b, custom
        """
    )

    parser.add_argument("data_file", nargs="?", help="JSONL training data file")
    parser.add_argument("--output", "-o", default="./text_lora_output",
                       help="Output directory for LoRA weights")
    parser.add_argument("--preset", "-p", default="llama-3-8b",
                       choices=list(TEXT_LORA_PRESETS.keys()),
                       help="Model preset")
    parser.add_argument("--epochs", "-e", type=int, default=3,
                       help="Training epochs (default: 3)")
    parser.add_argument("--no-unsloth", action="store_true",
                       help="Force PEFT instead of Unsloth")
    parser.add_argument("--list-presets", action="store_true",
                       help="List available model presets")

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]Available Text LoRA Presets[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        table = Table(box=box.ROUNDED, border_style="cyan")
        table.add_column("Preset", style="yellow")
        table.add_column("Base Model", style="green")
        table.add_column("LoRA Rank", justify="right")
        table.add_column("Learning Rate", justify="right")

        for name, config in TEXT_LORA_PRESETS.items():
            if name != "custom":
                table.add_row(
                    name,
                    config['base_model'],
                    str(config['lora_r']),
                    f"{config['learning_rate']:.0e}"
                )

        console.print(table)
        console.print()
        return

    # Require data file
    if not args.data_file:
        parser.print_help()
        return

    # Run training
    success = train_text_lora(
        data_file=args.data_file,
        output_dir=args.output,
        preset=args.preset,
        num_epochs=args.epochs,
        use_unsloth=not args.no_unsloth
    )

    if success:
        console.print()
        console.print(Panel.fit(
            f"[bold green]Text LoRA Training Complete![/bold green]\\n\\n"
            f"LoRA saved to: {args.output}\\n"
            f"Epochs: {args.epochs}\\n"
            f"Preset: {args.preset}\\n\\n"
            "[dim]Use this LoRA to fine-tune the model's personality and speaking style[/dim]",
            border_style="green"
        ))
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
