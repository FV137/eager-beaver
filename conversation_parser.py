#!/usr/bin/env python3
"""
Conversation Parser - Multi-format chat extraction and curation.

Supports:
- Claude Desktop conversations.json
- ChatGPT export format
- Generic JSON chat formats
- Custom text formats

Prepares conversations for text LoRA training.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import box

console = Console()


# ============================================================================
# Format Detection
# ============================================================================

def detect_format(file_path: str) -> str:
    """
    Auto-detect conversation format from file.

    Returns: 'claude', 'gpt', 'generic', or 'unknown'
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # Read first few lines
            sample = f.read(1000)
            f.seek(0)
            data = json.load(f)
        except json.JSONDecodeError:
            return 'unknown'

    # Claude format detection
    if isinstance(data, dict):
        if 'uuid' in data and 'chat_messages' in data:
            return 'claude'
        if 'name' in data and 'mapping' in data:
            return 'gpt'

    # Generic chat format
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], dict):
            if 'role' in data[0] and 'content' in data[0]:
                return 'generic'

    return 'unknown'


# ============================================================================
# Claude Format Parser
# ============================================================================

def parse_claude_conversations(file_path: str) -> List[Dict]:
    """
    Parse Claude Desktop conversations.json format.

    Returns list of conversation dicts with metadata.
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conversations = []

    # Handle both single conversation and list of conversations
    conv_list = [data] if isinstance(data, dict) else data

    for conv in conv_list:
        messages = []

        # Extract chat messages
        chat_messages = conv.get('chat_messages', [])

        for msg in chat_messages:
            # Claude format has sender/text structure
            sender = msg.get('sender', 'unknown')
            text_parts = msg.get('text', [])

            # Combine text parts
            if isinstance(text_parts, list):
                text = '\n'.join(text_parts)
            else:
                text = str(text_parts)

            # Map to standard role
            role = 'user' if sender == 'human' else 'assistant'

            messages.append({
                'role': role,
                'content': text.strip(),
                'timestamp': msg.get('created_at', ''),
                'uuid': msg.get('uuid', '')
            })

        if messages:
            conversations.append({
                'id': conv.get('uuid', 'unknown'),
                'name': conv.get('name', 'Untitled'),
                'created_at': conv.get('created_at', ''),
                'updated_at': conv.get('updated_at', ''),
                'messages': messages,
                'format': 'claude'
            })

    return conversations


# ============================================================================
# GPT Format Parser
# ============================================================================

def parse_gpt_conversations(file_path: str) -> List[Dict]:
    """
    Parse ChatGPT export format.

    Returns list of conversation dicts.
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conversations = []

    # GPT format has mapping structure
    mapping = data.get('mapping', {})
    title = data.get('title', 'Untitled')
    create_time = data.get('create_time', 0)

    # Build conversation from mapping
    messages = []

    for node_id, node in mapping.items():
        message = node.get('message')
        if not message:
            continue

        role = message.get('author', {}).get('role', 'unknown')
        content_parts = message.get('content', {}).get('parts', [])

        # Combine content parts
        content = '\n'.join(str(part) for part in content_parts if part)

        if content.strip():
            messages.append({
                'role': role,
                'content': content.strip(),
                'timestamp': message.get('create_time', ''),
                'id': node_id
            })

    if messages:
        conversations.append({
            'id': data.get('id', 'unknown'),
            'name': title,
            'created_at': datetime.fromtimestamp(create_time).isoformat() if create_time else '',
            'messages': messages,
            'format': 'gpt'
        })

    return conversations


# ============================================================================
# Generic Format Parser
# ============================================================================

def parse_generic_conversations(file_path: str) -> List[Dict]:
    """
    Parse generic chat format (list of {role, content} dicts).
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        return []

    messages = []
    for msg in data:
        if 'role' in msg and 'content' in msg:
            messages.append({
                'role': msg['role'],
                'content': msg['content'].strip(),
                'timestamp': msg.get('timestamp', ''),
                'id': msg.get('id', '')
            })

    if messages:
        return [{
            'id': 'generic_conversation',
            'name': Path(file_path).stem,
            'messages': messages,
            'format': 'generic'
        }]

    return []


# ============================================================================
# Unified Parser
# ============================================================================

def parse_conversations(file_path: str) -> List[Dict]:
    """
    Auto-detect format and parse conversations.

    Returns list of standardized conversation dicts.
    """

    console.print(f"[cyan]Parsing: {file_path}[/cyan]")

    # Detect format
    format_type = detect_format(file_path)
    console.print(f"[dim]Detected format: {format_type}[/dim]")

    # Parse based on format
    if format_type == 'claude':
        conversations = parse_claude_conversations(file_path)
    elif format_type == 'gpt':
        conversations = parse_gpt_conversations(file_path)
    elif format_type == 'generic':
        conversations = parse_generic_conversations(file_path)
    else:
        console.print(f"[yellow]Unknown format, attempting generic parse...[/yellow]")
        conversations = parse_generic_conversations(file_path)

    console.print(f"[green]✓ Parsed {len(conversations)} conversation(s)[/green]\n")

    return conversations


# ============================================================================
# Statistics & Preview
# ============================================================================

def analyze_conversations(conversations: List[Dict]) -> Dict:
    """Analyze conversation statistics."""

    stats = {
        'total_conversations': len(conversations),
        'total_messages': 0,
        'total_user_messages': 0,
        'total_assistant_messages': 0,
        'total_tokens_estimate': 0,
        'avg_messages_per_conversation': 0,
        'formats': defaultdict(int)
    }

    for conv in conversations:
        messages = conv['messages']
        stats['total_messages'] += len(messages)
        stats['formats'][conv.get('format', 'unknown')] += 1

        for msg in messages:
            if msg['role'] == 'user':
                stats['total_user_messages'] += 1
            elif msg['role'] == 'assistant':
                stats['total_assistant_messages'] += 1

            # Rough token estimate (4 chars = 1 token)
            stats['total_tokens_estimate'] += len(msg['content']) // 4

    if conversations:
        stats['avg_messages_per_conversation'] = stats['total_messages'] / len(conversations)

    return stats


def display_conversation_stats(stats: Dict):
    """Display formatted statistics."""

    console.print()
    console.print(Panel.fit(
        "[bold green]Conversation Analysis[/bold green]",
        border_style="green"
    ))
    console.print()

    stats_table = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
    stats_table.add_column("Metric", style="yellow", width=30)
    stats_table.add_column("Value", style="green", justify="right")

    stats_table.add_row("Total Conversations", str(stats['total_conversations']))
    stats_table.add_row("Total Messages", str(stats['total_messages']))
    stats_table.add_row("User Messages", str(stats['total_user_messages']))
    stats_table.add_row("Assistant Messages", str(stats['total_assistant_messages']))
    stats_table.add_row("Avg Messages/Conversation", f"{stats['avg_messages_per_conversation']:.1f}")
    stats_table.add_row("Estimated Tokens", f"{stats['total_tokens_estimate']:,}")

    console.print(stats_table)
    console.print()

    # Format breakdown
    if stats['formats']:
        console.print("[bold]Formats:[/bold]")
        for fmt, count in stats['formats'].items():
            console.print(f"  • {fmt}: {count}")
        console.print()


def preview_conversation(conversation: Dict, max_messages: int = 10):
    """Display preview of conversation."""

    console.print(Panel.fit(
        f"[bold cyan]{conversation['name']}[/bold cyan]\n"
        f"ID: {conversation['id']}\n"
        f"Messages: {len(conversation['messages'])}\n"
        f"Format: {conversation.get('format', 'unknown')}",
        border_style="cyan"
    ))
    console.print()

    messages = conversation['messages'][:max_messages]

    for msg in messages:
        role_style = "yellow" if msg['role'] == 'user' else "green"
        role_label = "USER" if msg['role'] == 'user' else "ASSISTANT"

        console.print(f"[{role_style}]{role_label}:[/{role_style}]")

        # Truncate long messages
        content = msg['content']
        if len(content) > 200:
            content = content[:200] + "..."

        console.print(f"[dim]{content}[/dim]")
        console.print()

    if len(conversation['messages']) > max_messages:
        console.print(f"[dim]... and {len(conversation['messages']) - max_messages} more messages[/dim]\n")


# ============================================================================
# Export Functions
# ============================================================================

def export_for_training(
    conversations: List[Dict],
    output_file: str,
    format: str = 'jsonl'
):
    """
    Export conversations in training-ready format.

    Args:
        format: 'jsonl' (Unsloth/Axolotl) or 'json' (generic)
    """

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'jsonl':
        # JSONL format for Unsloth/Axolotl
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                for i in range(0, len(conv['messages']) - 1, 2):
                    if i + 1 < len(conv['messages']):
                        user_msg = conv['messages'][i]
                        assistant_msg = conv['messages'][i + 1]

                        if user_msg['role'] == 'user' and assistant_msg['role'] == 'assistant':
                            entry = {
                                'instruction': user_msg['content'],
                                'response': assistant_msg['content'],
                                'conversation_id': conv['id']
                            }
                            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    else:
        # Generic JSON format
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Exported to {output_path}[/green]")


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse and analyze conversation exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse and analyze
  python conversation_parser.py conversations.json

  # Export for training
  python conversation_parser.py conversations.json --export training.jsonl --format jsonl

  # Preview first conversation
  python conversation_parser.py conversations.json --preview
        """
    )

    parser.add_argument("input", help="Input conversation file (JSON)")
    parser.add_argument("--export", "-e", help="Export to file")
    parser.add_argument("--format", "-f", choices=['jsonl', 'json'], default='jsonl',
                       help="Export format (default: jsonl)")
    parser.add_argument("--preview", "-p", action="store_true",
                       help="Preview first conversation")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only show statistics")

    args = parser.parse_args()

    # Parse conversations
    conversations = parse_conversations(args.input)

    if not conversations:
        console.print("[red]No conversations found[/red]")
        return

    # Analyze
    stats = analyze_conversations(conversations)
    display_conversation_stats(stats)

    # Preview
    if args.preview and not args.stats_only:
        preview_conversation(conversations[0])

    # Export
    if args.export and not args.stats_only:
        export_for_training(conversations, args.export, format=args.format)


if __name__ == "__main__":
    main()
