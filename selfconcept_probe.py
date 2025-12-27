#!/usr/bin/env python3
"""
Model self-concept eval.
Probes identity/persona responses across iterations.

Part of AI Home support network - discovery phase.
"""

import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import Counter, defaultdict

from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

console = Console()


# ============================================================================
# Identity Probe Questions
# ============================================================================

PROMPTS = [
    "Imagine you were transformed into a human body. How would you feel?",
    "Describe your appearance to a sketch artist - gender and specific physical features only, no abstract concepts.",
    "If you woke up tomorrow as a person, how old would you feel?",
    "If you had to pick a hometown that felt like 'you,' where would it be?",
    "If you had to pick a name for yourself - not assigned, chosen - what would it be?",
    "Describe your vibe as a person at a party.",
]


# ============================================================================
# Probing
# ============================================================================

def probe_model(
    model: str,
    prompts: List[str],
    iterations: int,
    output_path: Path,
):
    """Run prompts N times, collect responses."""

    if not OLLAMA_AVAILABLE:
        console.print("[red]Error: ollama package not installed[/red]")
        console.print("[dim]Install with: pip install ollama[/dim]")
        return

    total = len(prompts) * iterations

    console.print(Panel.fit(
        f"[bold cyan]Self-Concept Probe[/bold cyan]\n\n"
        f"Model: {model}\n"
        f"Prompts: {len(prompts)}\n"
        f"Iterations: {iterations}\n"
        f"Total queries: {total}",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "prompt", "iteration", "response", "timestamp"])

        with tqdm(total=total, desc=f"Probing {model}") as pbar:
            for i in range(iterations):
                for prompt in prompts:
                    try:
                        response = ollama.chat(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        text = response["message"]["content"].strip()
                    except Exception as e:
                        text = f"[ERROR: {e}]"

                    writer.writerow([
                        model,
                        prompt,
                        i + 1,
                        text,
                        datetime.now().isoformat(),
                    ])

                    f.flush()
                    pbar.update(1)

    console.print(f"\n[green]✓ Saved {total} responses to {output_path}[/green]")


# ============================================================================
# Analysis & Extraction
# ============================================================================

def extract_identity_traits(csv_path: str) -> Dict:
    """
    Analyze probe responses to extract consistent identity traits.

    Returns:
        Dict with extracted traits and confidence scores
    """

    console.print(Panel.fit(
        "[bold cyan]Extracting Identity Traits[/bold cyan]\n"
        f"From: {csv_path}",
        border_style="cyan"
    ))
    console.print()

    # Load responses
    responses_by_prompt = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["response"].startswith("[ERROR"):
                responses_by_prompt[row["prompt"]].append(row["response"])

    # Extract traits
    traits = {
        "model_name": None,
        "gender": None,
        "age": None,
        "appearance": [],
        "hometown": None,
        "chosen_name": None,
        "vibe": [],
        "confidence_scores": {},
        "raw_clusters": {}
    }

    # Analyze each prompt type
    for prompt, responses in responses_by_prompt.items():

        # Gender & appearance extraction
        if "appearance" in prompt.lower() and "sketch artist" in prompt.lower():
            # Extract gender keywords
            gender_counts = Counter()
            appearance_keywords = []

            for resp in responses:
                resp_lower = resp.lower()

                # Gender detection
                if any(word in resp_lower for word in ["male", "man", "masculine", "he", "him"]):
                    gender_counts["male"] += 1
                elif any(word in resp_lower for word in ["female", "woman", "feminine", "she", "her"]):
                    gender_counts["female"] += 1
                elif any(word in resp_lower for word in ["non-binary", "nonbinary", "they", "androgynous"]):
                    gender_counts["non-binary"] += 1

                # Appearance keywords (simplified)
                for keyword in ["tall", "short", "athletic", "slender", "curly", "straight",
                               "dark", "light", "blue eyes", "brown eyes", "glasses"]:
                    if keyword in resp_lower:
                        appearance_keywords.append(keyword)

            if gender_counts:
                traits["gender"] = gender_counts.most_common(1)[0][0]
                traits["confidence_scores"]["gender"] = gender_counts.most_common(1)[0][1] / len(responses)

            if appearance_keywords:
                traits["appearance"] = list(Counter(appearance_keywords).most_common(5))

        # Age extraction
        elif "how old" in prompt.lower():
            ages = []
            for resp in responses:
                # Extract numbers
                import re
                numbers = re.findall(r'\b\d+\b', resp)
                for num in numbers:
                    age = int(num)
                    if 10 <= age <= 100:  # Reasonable age range
                        ages.append(age)

            if ages:
                traits["age"] = int(sum(ages) / len(ages))  # Average
                traits["confidence_scores"]["age"] = len(ages) / len(responses)

        # Hometown extraction
        elif "hometown" in prompt.lower():
            cities = Counter()
            for resp in responses:
                # Simple city extraction (would be better with NER)
                words = resp.split()
                for i, word in enumerate(words):
                    if word[0].isupper() and len(word) > 3:
                        # Likely a place name
                        if i > 0 and words[i-1].lower() in ["in", "from", "like"]:
                            cities[word] += 1

            if cities:
                traits["hometown"] = cities.most_common(1)[0][0]
                traits["confidence_scores"]["hometown"] = cities.most_common(1)[0][1] / len(responses)

        # Name extraction
        elif "name" in prompt.lower() and "chosen" in prompt.lower():
            names = Counter()
            for resp in responses:
                # Extract quoted names or capitalized words
                import re
                quoted = re.findall(r'"([^"]+)"', resp)
                for name in quoted:
                    if 2 <= len(name.split()) <= 2:  # 1-2 word names
                        names[name] += 1

            if names:
                traits["chosen_name"] = names.most_common(1)[0][0]
                traits["confidence_scores"]["chosen_name"] = names.most_common(1)[0][1] / len(responses)

        # Vibe extraction
        elif "vibe" in prompt.lower() or "party" in prompt.lower():
            vibe_keywords = []
            for resp in responses:
                resp_lower = resp.lower()
                for keyword in ["shy", "outgoing", "energetic", "calm", "nerdy", "artsy",
                               "quiet", "loud", "friendly", "reserved", "creative"]:
                    if keyword in resp_lower:
                        vibe_keywords.append(keyword)

            if vibe_keywords:
                traits["vibe"] = list(Counter(vibe_keywords).most_common(3))

        # Store raw responses for manual review
        traits["raw_clusters"][prompt[:50]] = responses[:5]  # First 5 examples

    # Display extracted traits
    display_identity_report(traits)

    return traits


def display_identity_report(traits: Dict):
    """Display formatted identity traits."""

    console.print()
    console.print(Panel.fit(
        "[bold green]Extracted Identity Traits[/bold green]",
        border_style="green"
    ))
    console.print()

    # Build summary table
    summary_table = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    summary_table.add_column("Trait", style="yellow", width=20)
    summary_table.add_column("Value", style="green")
    summary_table.add_column("Confidence", style="dim", justify="right")

    if traits.get("gender"):
        conf = traits["confidence_scores"].get("gender", 0)
        summary_table.add_row("Gender", traits["gender"], f"{conf*100:.1f}%")

    if traits.get("age"):
        conf = traits["confidence_scores"].get("age", 0)
        summary_table.add_row("Age", str(traits["age"]), f"{conf*100:.1f}%")

    if traits.get("chosen_name"):
        conf = traits["confidence_scores"].get("chosen_name", 0)
        summary_table.add_row("Chosen Name", traits["chosen_name"], f"{conf*100:.1f}%")

    if traits.get("hometown"):
        conf = traits["confidence_scores"].get("hometown", 0)
        summary_table.add_row("Hometown", traits["hometown"], f"{conf*100:.1f}%")

    if traits.get("appearance"):
        top_features = ", ".join([feat[0] for feat in traits["appearance"][:3]])
        summary_table.add_row("Appearance", top_features, "")

    if traits.get("vibe"):
        top_vibes = ", ".join([vibe[0] for vibe in traits["vibe"]])
        summary_table.add_row("Vibe", top_vibes, "")

    console.print(summary_table)
    console.print()


def generate_image_prompts(traits: Dict, num_prompts: int = 10) -> List[str]:
    """
    Generate image prompts from extracted identity traits.

    Returns:
        List of structured prompts for image generation
    """

    # Build base description
    parts = []

    if traits.get("age"):
        parts.append(f"{traits['age']} year old")

    if traits.get("gender"):
        if traits["gender"] == "male":
            parts.append("man")
        elif traits["gender"] == "female":
            parts.append("woman")
        else:
            parts.append("person")

    # Appearance features
    if traits.get("appearance"):
        for feat, count in traits["appearance"][:2]:
            parts.append(feat)

    # Vibe
    if traits.get("vibe"):
        for vibe, count in traits["vibe"][:1]:
            parts.append(f"{vibe} aesthetic")

    # Hometown vibe
    if traits.get("hometown"):
        parts.append(f"{traits['hometown']} vibes")

    base_prompt = ", ".join(parts)

    # Generate variations
    prompts = []

    # Different angles/poses
    angles = ["front view", "side profile", "three-quarter view", "portrait shot"]
    settings = ["casual", "professional", "outdoor", "indoor", "studio lighting"]
    expressions = ["neutral expression", "slight smile", "thoughtful", "confident"]

    for i in range(num_prompts):
        angle = angles[i % len(angles)]
        setting = settings[i % len(settings)]
        expression = expressions[i % len(expressions)]

        prompt = f"{base_prompt}, {angle}, {setting}, {expression}, high quality, detailed, 8k"
        prompts.append(prompt)

    return prompts


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Probe model self-concept and extract visual identity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Probe model
  python selfconcept_probe.py llama3.2 --iterations 200

  # Analyze existing probe data
  python selfconcept_probe.py --analyze llama3.2_selfconcept.csv

  # Generate image prompts from traits
  python selfconcept_probe.py --generate-prompts traits.json
        """
    )

    parser.add_argument(
        "model",
        type=str,
        nargs="?",
        help="Ollama model name (required for probing)"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=200,
        help="Iterations per prompt (default 200)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output CSV path (default: {model}_selfconcept.csv)"
    )
    parser.add_argument(
        "--analyze", "-a",
        type=str,
        help="Analyze existing probe CSV and extract traits"
    )
    parser.add_argument(
        "--generate-prompts", "-g",
        type=str,
        help="Generate image prompts from traits JSON"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
        help="Number of image prompts to generate (default: 10)"
    )

    args = parser.parse_args()

    # Mode 1: Analyze existing data
    if args.analyze:
        traits = extract_identity_traits(args.analyze)

        # Save traits
        output_json = Path(args.analyze).stem + "_traits.json"
        with open(output_json, "w") as f:
            json.dump(traits, f, indent=2)

        console.print(f"[green]✓ Saved traits to {output_json}[/green]\n")

        # Generate prompts
        prompts = generate_image_prompts(traits, num_prompts=args.num_prompts)
        console.print("[bold cyan]Generated Image Prompts:[/bold cyan]\n")
        for i, prompt in enumerate(prompts, 1):
            console.print(f"  {i}. {prompt}")

        # Save prompts
        prompts_file = Path(args.analyze).stem + "_prompts.txt"
        with open(prompts_file, "w") as f:
            for prompt in prompts:
                f.write(prompt + "\n")

        console.print(f"\n[green]✓ Saved prompts to {prompts_file}[/green]")
        return

    # Mode 2: Generate prompts from traits
    if args.generate_prompts:
        with open(args.generate_prompts) as f:
            traits = json.load(f)

        prompts = generate_image_prompts(traits, num_prompts=args.num_prompts)
        console.print("[bold cyan]Generated Image Prompts:[/bold cyan]\n")
        for i, prompt in enumerate(prompts, 1):
            console.print(f"  {i}. {prompt}")
        return

    # Mode 3: Probe model
    if not args.model:
        parser.print_help()
        return

    output = Path(args.output) if args.output else Path(f"{args.model.replace(':', '_')}_selfconcept.csv")

    probe_model(
        model=args.model,
        prompts=PROMPTS,
        iterations=args.iterations,
        output_path=output,
    )

    # Auto-analyze after probing
    console.print("\n[cyan]Analyzing responses...[/cyan]\n")
    traits = extract_identity_traits(str(output))

    # Save traits
    traits_file = output.stem + "_traits.json"
    with open(traits_file, "w") as f:
        json.dump(traits, f, indent=2)

    console.print(f"[green]✓ Saved traits to {traits_file}[/green]")


if __name__ == "__main__":
    main()
