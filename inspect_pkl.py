#!/usr/bin/env python3
"""
PKL Inspector - Safely analyze pickle files before loading.

Checks for:
- Dangerous opcodes (arbitrary code execution)
- Suspicious imports/globals
- File structure and contents
- Safe conversion to .safetensors

NEVER blindly load .pkl files from untrusted sources!
"""

import io
import pickle
import pickletools
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.syntax import Syntax

console = Console()


# ============================================================================
# Pickle Safety Analysis
# ============================================================================

# Dangerous pickle opcodes that can execute arbitrary code
DANGEROUS_OPCODES = {
    'GLOBAL',      # Import and call arbitrary functions
    'REDUCE',      # Call functions with arguments
    'BUILD',       # Call __setstate__ or update __dict__
    'INST',        # Create instances (old pickle protocol)
    'OBJ',         # Build objects (old pickle protocol)
}

# Safe modules for ML models
SAFE_MODULES = {
    'torch',
    'torch.nn',
    'torch.nn.modules',
    'torch.nn.functional',
    'torch.optim',
    'torch.cuda',
    'torch.storage',
    'torch._utils',
    'numpy',
    'numpy.core',
    'numpy.core.multiarray',
    'collections',
    'collections.abc',
    '_codecs',
    'builtins',
}


def disassemble_pickle(pkl_path: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Safely disassemble pickle to inspect opcodes.

    Returns:
        (opcodes_list, global_imports)
    """
    opcodes = []
    globals_found = []

    with open(pkl_path, 'rb') as f:
        # Get disassembly
        output = io.StringIO()
        pickletools.dis(f, out=output)
        disassembly = output.getvalue()

        # Parse for opcodes and globals
        for line in disassembly.split('\n'):
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            opcode = parts[1]
            opcodes.append(opcode)

            # Track GLOBAL imports
            if opcode == 'GLOBAL':
                # Format: "pos GLOBAL 'module' 'name'"
                if len(parts) >= 4:
                    module = parts[2].strip("'\"")
                    name = parts[3].strip("'\"")
                    globals_found.append((module, name))

    return opcodes, globals_found


def analyze_safety(pkl_path: str) -> Dict:
    """
    Analyze pickle file for safety issues.

    Returns dict with:
        - is_safe: bool
        - issues: List[str]
        - warnings: List[str]
        - globals: List[Tuple[module, name]]
    """
    console.print(f"[cyan]Analyzing {Path(pkl_path).name}...[/cyan]\n")

    opcodes, globals_found = disassemble_pickle(pkl_path)

    issues = []
    warnings = []

    # Check for dangerous opcodes
    dangerous_found = set(opcodes) & DANGEROUS_OPCODES
    if dangerous_found:
        warnings.append(f"Contains opcodes: {', '.join(dangerous_found)}")

    # Check global imports
    suspicious_globals = []
    for module, name in globals_found:
        # Check if module is in safe list
        is_safe = any(module.startswith(safe) for safe in SAFE_MODULES)

        if not is_safe:
            suspicious_globals.append(f"{module}.{name}")
            issues.append(f"SUSPICIOUS IMPORT: {module}.{name}")

        # Flag specific dangerous patterns
        if 'os' in module or 'subprocess' in module:
            issues.append(f"DANGER: System access via {module}.{name}")
        if 'eval' in name or 'exec' in name or '__import__' in name:
            issues.append(f"DANGER: Code execution via {module}.{name}")
        if 'open' in name and 'builtins' in module:
            issues.append(f"WARNING: File access via {module}.{name}")

    is_safe = len(issues) == 0

    return {
        "is_safe": is_safe,
        "issues": issues,
        "warnings": warnings,
        "globals": globals_found,
        "suspicious_globals": suspicious_globals,
        "opcodes_count": len(opcodes),
        "dangerous_opcodes": list(dangerous_found)
    }


# ============================================================================
# Safe Loading with Restricted Unpickler
# ============================================================================

class RestrictedUnpickler(pickle.Unpickler):
    """Pickle unpickler that only allows safe operations."""

    def find_class(self, module, name):
        # Only allow whitelisted modules
        is_safe = any(module.startswith(safe) for safe in SAFE_MODULES)

        if not is_safe:
            raise pickle.UnpicklingError(
                f"BLOCKED: Attempted to import {module}.{name}\n"
                f"This is not in the safe modules list!"
            )

        # Additional specific checks
        if name in ['eval', 'exec', '__import__', 'compile']:
            raise pickle.UnpicklingError(
                f"BLOCKED: Attempted dangerous operation {name}"
            )

        return super().find_class(module, name)


def safe_load_pickle(pkl_path: str):
    """
    Attempt to load pickle with restricted unpickler.

    Raises UnpicklingError if unsafe operations detected.
    """
    with open(pkl_path, 'rb') as f:
        return RestrictedUnpickler(f).load()


# ============================================================================
# Conversion to SafeTensors
# ============================================================================

def convert_to_safetensors(pkl_path: str, output_path: str = None):
    """
    Convert .pkl model to .safetensors format.

    Args:
        pkl_path: Path to .pkl file
        output_path: Optional output path (defaults to same name with .safetensors)
    """
    pkl_path = Path(pkl_path)

    if output_path is None:
        output_path = pkl_path.with_suffix('.safetensors')
    else:
        output_path = Path(output_path)

    console.print(f"[cyan]Loading {pkl_path.name} with restricted unpickler...[/cyan]")

    try:
        # Load with restricted unpickler
        state_dict = safe_load_pickle(str(pkl_path))

        # Handle different formats
        if isinstance(state_dict, dict):
            # Check if it's already a state_dict or wrapped
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        elif hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()

        # Ensure all values are tensors
        if not isinstance(state_dict, dict):
            console.print("[red]Error: Loaded object is not a state dict![/red]")
            return False

        # Convert to CPU tensors
        cpu_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_state_dict[key] = value.cpu()
            else:
                console.print(f"[yellow]Warning: Skipping non-tensor key: {key}[/yellow]")

        # Save as safetensors
        from safetensors.torch import save_file
        save_file(cpu_state_dict, str(output_path))

        console.print(f"[green]✓ Converted to {output_path}[/green]")
        console.print(f"[dim]  Keys: {len(cpu_state_dict)}[/dim]")

        return True

    except pickle.UnpicklingError as e:
        console.print(f"[red]✗ BLOCKED: {e}[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✗ Error during conversion: {e}[/red]")
        return False


# ============================================================================
# Display Report
# ============================================================================

def display_safety_report(pkl_path: str, analysis: Dict):
    """Display formatted safety analysis report."""

    # Header
    status = "SAFE" if analysis["is_safe"] else "UNSAFE"
    status_color = "green" if analysis["is_safe"] else "red"

    console.print()
    console.print(Panel.fit(
        f"[bold {status_color}]Safety Analysis: {status}[/bold {status_color}]\n"
        f"File: {Path(pkl_path).name}",
        border_style=status_color,
        box=box.DOUBLE
    ))
    console.print()

    # Issues
    if analysis["issues"]:
        console.print("[bold red]🚨 SECURITY ISSUES FOUND:[/bold red]\n")
        for issue in analysis["issues"]:
            console.print(f"  [red]✗[/red] {issue}")
        console.print()

    # Warnings
    if analysis["warnings"]:
        console.print("[bold yellow]⚠️  WARNINGS:[/bold yellow]\n")
        for warning in analysis["warnings"]:
            console.print(f"  [yellow]![/yellow] {warning}")
        console.print()

    # Global imports table
    if analysis["globals"]:
        imports_table = Table(
            title="Global Imports Found",
            box=box.ROUNDED,
            border_style="cyan"
        )
        imports_table.add_column("Module", style="yellow")
        imports_table.add_column("Name", style="cyan")
        imports_table.add_column("Status", style="green")

        for module, name in analysis["globals"][:20]:  # Show first 20
            is_suspicious = f"{module}.{name}" in analysis["suspicious_globals"]
            status = "[red]SUSPICIOUS[/red]" if is_suspicious else "[green]OK[/green]"
            imports_table.add_row(module, name, status)

        if len(analysis["globals"]) > 20:
            imports_table.add_row("...", "...", f"[dim](+{len(analysis['globals']) - 20} more)[/dim]")

        console.print(imports_table)
        console.print()

    # Summary stats
    stats_table = Table(show_header=False, box=None)
    stats_table.add_row("Total opcodes", str(analysis["opcodes_count"]))
    stats_table.add_row("Dangerous opcodes", str(len(analysis["dangerous_opcodes"])))
    stats_table.add_row("Global imports", str(len(analysis["globals"])))
    stats_table.add_row("Suspicious imports", str(len(analysis["suspicious_globals"])))

    console.print(stats_table)
    console.print()

    # Recommendation
    if analysis["is_safe"]:
        console.print("[green]✓ This file appears safe to load with restricted unpickler[/green]")
        console.print("[dim]  Recommendation: Convert to .safetensors for future use[/dim]")
    else:
        console.print("[red]✗ DO NOT LOAD THIS FILE[/red]")
        console.print("[dim]  Contains suspicious or dangerous operations[/dim]")

    console.print()


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Safely inspect and convert pickle files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a pickle file
  python inspect_pkl.py model.pkl

  # Analyze and convert if safe
  python inspect_pkl.py model.pkl --convert

  # Convert with custom output path
  python inspect_pkl.py model.pkl --convert --output safe_model.safetensors

  # Batch analyze directory
  python inspect_pkl.py models/*.pkl
        """
    )

    parser.add_argument("files", nargs="+", help="Pickle file(s) to inspect")
    parser.add_argument("--convert", "-c", action="store_true",
                       help="Convert to .safetensors if safe")
    parser.add_argument("--output", "-o", help="Output path for conversion")
    parser.add_argument("--force", action="store_true",
                       help="Force conversion even if warnings (not recommended)")

    args = parser.parse_args()

    # Process each file
    for pkl_file in args.files:
        pkl_path = Path(pkl_file)

        if not pkl_path.exists():
            console.print(f"[red]File not found: {pkl_path}[/red]\n")
            continue

        # Analyze
        analysis = analyze_safety(str(pkl_path))
        display_safety_report(str(pkl_path), analysis)

        # Convert if requested
        if args.convert:
            if analysis["is_safe"] or args.force:
                output = args.output if args.output else None
                convert_to_safetensors(str(pkl_path), output)
            else:
                console.print("[red]Skipping conversion - file is not safe[/red]")
                console.print("[dim]Use --force to override (not recommended)[/dim]\n")

        # Separator for multiple files
        if len(args.files) > 1:
            console.print("─" * 80)
            console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
