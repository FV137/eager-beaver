#!/usr/bin/env python3
"""
Text LoRA Inference - Test personality LoRAs

Load and test trained personality LoRAs for text generation.
Interactive chat mode to explore the learned personality.
"""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import box

console = Console()

# Try to import inference libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False


# ============================================================================
# Model Loading
# ============================================================================

def load_model_with_lora(
    base_model: str,
    lora_path: str,
    use_unsloth: bool = True
):
    """Load base model with LoRA weights."""

    if use_unsloth and UNSLOTH_AVAILABLE:
        return load_with_unsloth(base_model, lora_path)
    elif TRANSFORMERS_AVAILABLE:
        return load_with_transformers(base_model, lora_path)
    else:
        console.print("[red]Error: No inference libraries available[/red]")
        console.print("[dim]Install: pip install transformers peft[/dim]")
        return None, None


def load_with_unsloth(base_model: str, lora_path: str):
    """Load with Unsloth (faster)."""

    console.print("[cyan]Loading model with Unsloth...[/cyan]")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_path)

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    console.print("[green]✓ Model loaded with Unsloth[/green]\n")
    return model, tokenizer


def load_with_transformers(base_model: str, lora_path: str):
    """Load with standard Transformers."""

    console.print("[cyan]Loading model with Transformers...[/cyan]")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, lora_path)

    console.print("[green]✓ Model loaded with Transformers[/green]\n")
    return model, tokenizer


# ============================================================================
# Text Generation
# ============================================================================

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    """Generate text response."""

    # Format prompt (alpaca style)
    formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[1].strip()

    return response


# ============================================================================
# Interactive Chat
# ============================================================================

def interactive_chat(model, tokenizer):
    """Interactive chat mode to explore personality."""

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Interactive Chat Mode[/bold cyan]\n\n"
        "[dim]Chat with the personality LoRA to explore learned patterns\n"
        "Type 'quit' or 'exit' to stop[/dim]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    while True:
        # Get user input
        try:
            user_input = Prompt.ask("[bold yellow]You[/bold yellow]")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if not user_input.strip():
            continue

        # Generate response
        console.print("[dim]Generating...[/dim]")
        response = generate_response(model, tokenizer, user_input)

        # Display
        console.print()
        console.print(Panel(
            response,
            title="[bold cyan]Assistant[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        ))
        console.print()

    console.print("\n[dim]Chat ended[/dim]\n")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test personality LoRA with text generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Load a trained personality LoRA and chat interactively to explore
the learned personality patterns.

Examples:
  # Load and chat
  python infer_text_lora.py meta-llama/Meta-Llama-3-8B \\
    --lora ./text_lora_output

  # Single prompt
  python infer_text_lora.py meta-llama/Meta-Llama-3-8B \\
    --lora ./text_lora_output \\
    --prompt "Tell me about yourself"

  # Use Transformers instead of Unsloth
  python infer_text_lora.py mistralai/Mistral-7B-v0.1 \\
    --lora ./text_lora_output --no-unsloth
        """
    )

    parser.add_argument("base_model", help="Base model path or HF model ID")
    parser.add_argument("--lora", "-l", required=True,
                       help="Path to LoRA weights directory")
    parser.add_argument("--prompt", "-p",
                       help="Single prompt (non-interactive)")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Max generation length (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p (default: 0.9)")
    parser.add_argument("--no-unsloth", action="store_true",
                       help="Use Transformers instead of Unsloth")

    args = parser.parse_args()

    # Check dependencies
    if not TORCH_AVAILABLE:
        console.print("[red]Error: PyTorch not installed[/red]")
        sys.exit(1)

    if not TRANSFORMERS_AVAILABLE:
        console.print("[red]Error: transformers/peft not installed[/red]")
        console.print("[dim]Install: pip install transformers peft[/dim]")
        sys.exit(1)

    # Load model
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Loading Personality LoRA[/bold cyan]\n\n"
        f"Base Model: {args.base_model}\n"
        f"LoRA: {args.lora}",
        border_style="cyan"
    ))
    console.print()

    model, tokenizer = load_model_with_lora(
        base_model=args.base_model,
        lora_path=args.lora,
        use_unsloth=not args.no_unsloth
    )

    if model is None:
        sys.exit(1)

    # Single prompt or interactive
    if args.prompt:
        console.print(f"[yellow]Prompt:[/yellow] {args.prompt}\n")
        response = generate_response(
            model, tokenizer, args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        console.print(Panel(
            response,
            title="[bold cyan]Response[/bold cyan]",
            border_style="cyan"
        ))
        console.print()
    else:
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
