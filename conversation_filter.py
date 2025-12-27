#!/usr/bin/env python3
"""
Conversation Filter - Remove specific unwanted phrases.

Primary purpose: Block those two goddamned phrases ("not nothing", "hold space")
and other corporate speak / AI slop patterns.

Filters:
- Corporate speak blacklist ("not nothing", "hold space", "circle back", etc.)
- AI assistant slop ("I appreciate you", "that resonates", etc.)
- Custom regex patterns
- Empty/low-quality messages
- Optional toxicity detection (opt-in with --check-toxicity)

Prepares clean datasets for text LoRA training.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import Counter

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import box

console = Console()

# Try to import detoxify
try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False


# ============================================================================
# Filter Configuration
# ============================================================================

# Corporate speak / annoying patterns blacklist
DEFAULT_BLACKLIST = [
    r"\bnot nothing\b",
    r"\bhold(?:ing)? space\b",
    r"\bunpack that\b",
    r"\bcircle back\b",
    r"\btouch base\b",
    r"\bsynergize\b",
    r"\bleverage\b",  # When used as corporate jargon
    r"\bthink outside the box\b",
    r"\blow-hanging fruit\b",
    r"\bmove the needle\b",
    r"\bparadigm shift\b",
]

# Additional annoying AI assistant patterns
AI_SLOP_PATTERNS = [
    r"\bI appreciate you\b",  # Overused filler
    r"\bI hear you\b",  # Therapy-speak in wrong context
    r"\bthat resonates\b",
    r"\bI'm here for you\b",  # When inappropriate
    r"\bvalidate your feelings\b",
    r"\bin this space\b",  # Vague corporate speak
]

# Empty/low-quality patterns
LOW_QUALITY_PATTERNS = [
    r"^(ok|okay|yes|no|sure|thanks|thank you)\.?$",  # Single word responses
    r"^\.{3,}$",  # Just ellipsis
    r"^\s*$",  # Empty
]


# ============================================================================
# Toxicity Detection
# ============================================================================

class ToxicityDetector:
    """Wrapper for detoxify model."""

    def __init__(self, model_name: str = 'original', threshold: float = 0.8):
        if not DETOXIFY_AVAILABLE:
            console.print("[yellow]detoxify not available - toxicity detection disabled[/yellow]")
            console.print("[dim]Install with: pip install detoxify[/dim]\n")
            self.model = None
            return

        console.print(f"[cyan]Loading detoxify model: {model_name}[/cyan]")
        self.model = Detoxify(model_name)
        self.threshold = threshold
        console.print(f"[green]✓ Toxicity detector ready (threshold: {threshold})[/green]\n")

    def is_toxic(self, text: str) -> bool:
        """Check if text exceeds toxicity threshold."""
        if self.model is None:
            return False

        try:
            results = self.model.predict(text)
            # Check multiple toxicity categories
            is_toxic = (
                results.get('toxicity', 0) > self.threshold or
                results.get('severe_toxicity', 0) > self.threshold or
                results.get('obscene', 0) > self.threshold or
                results.get('threat', 0) > self.threshold or
                results.get('insult', 0) > self.threshold
            )
            return is_toxic
        except Exception:
            return False

    def get_scores(self, text: str) -> Dict:
        """Get detailed toxicity scores."""
        if self.model is None:
            return {}

        try:
            return self.model.predict(text)
        except Exception:
            return {}


# ============================================================================
# Pattern Matching
# ============================================================================

def matches_blacklist(text: str, blacklist: List[str]) -> Optional[str]:
    """
    Check if text matches any blacklist pattern.

    Returns:
        Matched pattern or None
    """
    text_lower = text.lower()

    for pattern in blacklist:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return pattern

    return None


def is_low_quality(text: str) -> bool:
    """Check if message is low quality (empty, single word, etc)."""
    text_stripped = text.strip()

    # Empty
    if not text_stripped:
        return True

    # Too short
    if len(text_stripped) < 3:
        return True

    # Check low quality patterns
    for pattern in LOW_QUALITY_PATTERNS:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return True

    return False


# ============================================================================
# Conversation Filtering
# ============================================================================

class ConversationFilter:
    """Filter conversations based on multiple criteria."""

    def __init__(
        self,
        blacklist: Optional[List[str]] = None,
        enable_ai_slop: bool = True,
        toxicity_threshold: float = 0.8,
        min_message_length: int = 10,
        enable_toxicity: bool = False
    ):
        # Combine blacklists
        self.blacklist = DEFAULT_BLACKLIST.copy()
        if blacklist:
            self.blacklist.extend(blacklist)

        if enable_ai_slop:
            self.blacklist.extend(AI_SLOP_PATTERNS)

        self.min_message_length = min_message_length

        # Toxicity detector
        self.toxicity_detector = None
        if enable_toxicity and DETOXIFY_AVAILABLE:
            self.toxicity_detector = ToxicityDetector(threshold=toxicity_threshold)

        # Stats
        self.stats = {
            'total_messages': 0,
            'filtered_blacklist': 0,
            'filtered_toxicity': 0,
            'filtered_low_quality': 0,
            'filtered_short': 0,
            'kept_messages': 0,
            'blacklist_matches': Counter(),
        }

    def filter_message(self, message: Dict) -> Tuple[bool, Optional[str]]:
        """
        Check if message should be filtered.

        Returns:
            (should_keep, reason_if_filtered)
        """
        content = message.get('content', '')
        self.stats['total_messages'] += 1

        # Check blacklist
        matched_pattern = matches_blacklist(content, self.blacklist)
        if matched_pattern:
            self.stats['filtered_blacklist'] += 1
            self.stats['blacklist_matches'][matched_pattern] += 1
            return False, f"Blacklist: {matched_pattern}"

        # Check toxicity
        if self.toxicity_detector and self.toxicity_detector.is_toxic(content):
            self.stats['filtered_toxicity'] += 1
            return False, "Toxicity detected"

        # Check low quality
        if is_low_quality(content):
            self.stats['filtered_low_quality'] += 1
            return False, "Low quality"

        # Check minimum length
        if len(content.strip()) < self.min_message_length:
            self.stats['filtered_short'] += 1
            return False, f"Too short (<{self.min_message_length} chars)"

        # Keep message
        self.stats['kept_messages'] += 1
        return True, None

    def filter_conversation(self, conversation: Dict) -> Optional[Dict]:
        """
        Filter messages in conversation.

        Returns:
            Filtered conversation or None if too many messages removed
        """
        messages = conversation['messages']
        filtered_messages = []

        for msg in messages:
            should_keep, reason = self.filter_message(msg)

            if should_keep:
                filtered_messages.append(msg)
            else:
                # Optionally log removed message
                if 'filter_reason' not in msg:
                    msg['filter_reason'] = reason

        # Keep conversation if we retained enough messages
        if len(filtered_messages) >= 2:  # Need at least one exchange
            conversation['messages'] = filtered_messages
            conversation['original_message_count'] = len(messages)
            conversation['filtered_message_count'] = len(filtered_messages)
            return conversation

        return None

    def filter_conversations(
        self,
        conversations: List[Dict],
        verbose: bool = False
    ) -> List[Dict]:
        """Filter all conversations."""

        filtered = []

        for conv in track(conversations, description="Filtering conversations"):
            filtered_conv = self.filter_conversation(conv)

            if filtered_conv:
                filtered.append(filtered_conv)

                if verbose:
                    console.print(
                        f"[green]✓[/green] {conv['name']}: "
                        f"{filtered_conv['filtered_message_count']}/"
                        f"{filtered_conv['original_message_count']} messages kept"
                    )

        return filtered

    def display_stats(self):
        """Display filtering statistics."""

        console.print()
        console.print(Panel.fit(
            "[bold cyan]Filtering Statistics[/bold cyan]",
            border_style="cyan"
        ))
        console.print()

        stats_table = Table(box=box.ROUNDED, border_style="cyan", show_header=False)
        stats_table.add_column("Category", style="yellow", width=30)
        stats_table.add_column("Count", style="green", justify="right")
        stats_table.add_column("%", style="dim", justify="right")

        total = self.stats['total_messages']

        stats_table.add_row("Total Messages", str(total), "100%")
        stats_table.add_row("Kept", str(self.stats['kept_messages']),
                           f"{self.stats['kept_messages']/total*100:.1f}%" if total > 0 else "0%")
        stats_table.add_row("", "", "")  # Spacer
        stats_table.add_row("Filtered - Blacklist", str(self.stats['filtered_blacklist']),
                           f"{self.stats['filtered_blacklist']/total*100:.1f}%" if total > 0 else "0%")
        stats_table.add_row("Filtered - Toxicity", str(self.stats['filtered_toxicity']),
                           f"{self.stats['filtered_toxicity']/total*100:.1f}%" if total > 0 else "0%")
        stats_table.add_row("Filtered - Low Quality", str(self.stats['filtered_low_quality']),
                           f"{self.stats['filtered_low_quality']/total*100:.1f}%" if total > 0 else "0%")
        stats_table.add_row("Filtered - Too Short", str(self.stats['filtered_short']),
                           f"{self.stats['filtered_short']/total*100:.1f}%" if total > 0 else "0%")

        console.print(stats_table)
        console.print()

        # Blacklist breakdown
        if self.stats['blacklist_matches']:
            console.print("[bold yellow]Top Blacklist Matches:[/bold yellow]")
            for pattern, count in self.stats['blacklist_matches'].most_common(10):
                console.print(f"  • {pattern}: {count}")
            console.print()


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter conversations for clean training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic filtering (blocks "not nothing", "hold space", corporate speak)
  python conversation_filter.py conversations.json --output clean.json

  # Keep it spicy - only block specific phrases
  python conversation_filter.py conversations.json --no-ai-slop --output clean.json

  # Add your own phrase blacklist
  python conversation_filter.py conversations.json \\
    --blacklist "my_pattern" "another_pattern" --output clean.json

  # Enable toxicity detection (opt-in, if you want it)
  python conversation_filter.py conversations.json \\
    --check-toxicity --toxicity-threshold 0.5 --output clean.json

Note: Toxicity detection is DISABLED by default. We only block specific
annoying phrases like "not nothing" and "hold space". Add --check-toxicity
if you want to filter toxic content too.
        """
    )

    parser.add_argument("input", help="Input conversation JSON file")
    parser.add_argument("--output", "-o", required=True,
                       help="Output filtered JSON file")
    parser.add_argument("--blacklist", "-b", nargs="+",
                       help="Additional blacklist patterns (regex)")
    parser.add_argument("--no-ai-slop", action="store_true",
                       help="Disable AI assistant slop filtering")
    parser.add_argument("--check-toxicity", action="store_true",
                       help="Enable toxicity detection (opt-in, disabled by default)")
    parser.add_argument("--toxicity-threshold", type=float, default=0.8,
                       help="Toxicity threshold when enabled (default: 0.8)")
    parser.add_argument("--min-length", type=int, default=10,
                       help="Minimum message length (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Load conversations
    console.print(f"[cyan]Loading conversations from {args.input}[/cyan]")
    with open(args.input, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    if not isinstance(conversations, list):
        conversations = [conversations]

    console.print(f"[green]✓ Loaded {len(conversations)} conversation(s)[/green]\n")

    # Create filter
    filter_obj = ConversationFilter(
        blacklist=args.blacklist,
        enable_ai_slop=not args.no_ai_slop,
        toxicity_threshold=args.toxicity_threshold,
        min_message_length=args.min_length,
        enable_toxicity=args.check_toxicity
    )

    # Filter
    filtered_conversations = filter_obj.filter_conversations(
        conversations,
        verbose=args.verbose
    )

    # Display stats
    filter_obj.display_stats()

    console.print(f"[bold]Result:[/bold] {len(filtered_conversations)}/{len(conversations)} conversations kept\n")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_conversations, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Saved filtered conversations to {output_path}[/green]")


if __name__ == "__main__":
    main()
