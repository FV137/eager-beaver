#!/usr/bin/env python3
"""
Self-Concept Probe to Training Data Converter

Converts self-concept probe responses into text LoRA training data.

This is the HUGE missed opportunity - we probe the model 200+ times
to discover its identity, generating perfect personality data...
and then don't use it for training!

This script converts probe responses into training format for
personality LoRA fine-tuning.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict

from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()


def load_probe_responses(csv_file: str) -> List[Dict]:
    """Load responses from self-concept probe CSV."""

    responses = []

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            responses.append({
                'iteration': int(row['iteration']),
                'question': row['question'],
                'response': row['response']
            })

    return responses


def convert_to_training_format(responses: List[Dict]) -> List[Dict]:
    """
    Convert probe Q&A pairs to training format.

    Each question-response pair becomes a training sample.
    This teaches the model to respond in its own discovered voice.
    """

    training_data = []

    for item in responses:
        training_data.append({
            'instruction': item['question'],
            'response': item['response'],
            'source': 'self_concept_probe',
            'iteration': item['iteration']
        })

    return training_data


def export_jsonl(training_data: List[Dict], output_file: str):
    """Export to JSONL format for text LoRA training."""

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')


def convert_probe_to_training(
    probe_csv: str,
    output_jsonl: str,
    min_response_length: int = 10
):
    """
    Complete conversion pipeline.

    Args:
        probe_csv: Path to probe responses CSV
        output_jsonl: Output JSONL file for training
        min_response_length: Filter out very short responses
    """

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Self-Concept Probe → Training Data Converter[/bold cyan]\n\n"
        "[dim]Converting identity discovery into personality training data[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Load probe responses
    console.print(f"[cyan]Loading probe responses from {probe_csv}...[/cyan]")
    responses = load_probe_responses(probe_csv)
    console.print(f"[green]✓ Loaded {len(responses)} probe responses[/green]\n")

    # Filter short responses
    filtered = [r for r in responses if len(r['response']) >= min_response_length]
    if len(filtered) < len(responses):
        console.print(f"[yellow]Filtered out {len(responses) - len(filtered)} short responses[/yellow]")
        responses = filtered

    # Convert to training format
    console.print("[cyan]Converting to training format...[/cyan]")
    training_data = convert_to_training_format(responses)
    console.print(f"[green]✓ Created {len(training_data)} training samples[/green]\n")

    # Export
    console.print(f"[cyan]Exporting to {output_jsonl}...[/cyan]")
    export_jsonl(training_data, output_jsonl)
    console.print(f"[green]✓ Exported to {output_jsonl}[/green]\n")

    # Stats
    avg_response_length = sum(len(d['response']) for d in training_data) / len(training_data)

    console.print(Panel.fit(
        f"[bold green]Conversion Complete![/bold green]\n\n"
        f"Training Samples: {len(training_data)}\n"
        f"Avg Response Length: {avg_response_length:.0f} chars\n\n"
        f"[cyan]Next Steps:[/cyan]\n"
        f"  1. Preview patterns: python text_pattern_preview.py {output_jsonl}\n"
        f"  2. Train LoRA: python text_lora_train.py {output_jsonl} --preset llama-3-8b\n\n"
        f"[dim]This LoRA will capture the model's self-discovered personality[/dim]",
        border_style="green"
    ))
    console.print()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert self-concept probe responses to text LoRA training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The probe generates 200+ iterations of the model articulating its identity.
This is PERFECT training data for personality LoRA fine-tuning!

This script converts probe CSV responses into JSONL training format.

Examples:
  # Convert probe responses
  python probe_to_training.py llama3.2_probe_responses.csv \\
    --output llama3.2_personality_training.jsonl

  # Filter very short responses
  python probe_to_training.py responses.csv \\
    --output training.jsonl --min-length 20

Workflow:
  1. Run self-concept probe: python selfconcept_probe.py
  2. Convert to training data: python probe_to_training.py responses.csv
  3. Preview patterns: python text_pattern_preview.py training.jsonl
  4. Train personality LoRA: python text_lora_train.py training.jsonl
        """
    )

    parser.add_argument("probe_csv", help="Self-concept probe responses CSV")
    parser.add_argument("--output", "-o", required=True,
                       help="Output JSONL training file")
    parser.add_argument("--min-length", type=int, default=10,
                       help="Minimum response length (default: 10)")

    args = parser.parse_args()

    convert_probe_to_training(
        probe_csv=args.probe_csv,
        output_jsonl=args.output,
        min_response_length=args.min_length
    )


if __name__ == "__main__":
    main()
