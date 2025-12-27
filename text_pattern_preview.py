#!/usr/bin/env python3
"""
Text Pattern Preview - Predict what a text LoRA will learn.

Analyzes training data BEFORE training to show:
- Linguistic patterns and speaking style
- Common phrases that will be emphasized
- Vocabulary and complexity metrics
- Personality indicators
- Dataset quality and sufficiency warnings

This is like gap_analysis.py but for TEXT - tells you what patterns
the LoRA will learn before you spend hours training.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import box

console = Console()


# ============================================================================
# Text Analysis
# ============================================================================

def load_training_data(jsonl_path: str) -> List[Dict]:
    """Load training data from JSONL."""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_responses(data: List[Dict]) -> List[str]:
    """Extract just the AI responses."""
    responses = []
    for entry in data:
        response = entry.get('response', '') or entry.get('output', '')
        if response:
            responses.append(response)
    return responses


def analyze_ngrams(texts: List[str], n: int = 3, top_k: int = 50) -> List[Tuple[str, int]]:
    """Extract most common n-grams."""
    ngrams = []

    for text in texts:
        # Lowercase and clean
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        # Generate n-grams
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)

    counter = Counter(ngrams)
    return counter.most_common(top_k)


def analyze_sentence_starters(texts: List[str], top_k: int = 30) -> List[Tuple[str, int]]:
    """Analyze how sentences typically start."""
    starters = []

    for text in texts:
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Get first 1-3 words
            words = sent.split()
            if words:
                if len(words) >= 3:
                    starter = ' '.join(words[:3]).lower()
                elif len(words) >= 2:
                    starter = ' '.join(words[:2]).lower()
                else:
                    starter = words[0].lower()
                starters.append(starter)

    counter = Counter(starters)
    return counter.most_common(top_k)


def analyze_vocabulary(texts: List[str]) -> Dict:
    """Analyze vocabulary richness and complexity."""
    all_words = []

    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)

    total_words = len(all_words)
    unique_words = len(set(all_words))

    # Type-Token Ratio (vocabulary richness)
    ttr = unique_words / total_words if total_words > 0 else 0

    # Average word length (complexity indicator)
    avg_word_length = sum(len(w) for w in all_words) / total_words if total_words > 0 else 0

    # Most common words
    word_freq = Counter(all_words)

    return {
        'total_words': total_words,
        'unique_words': unique_words,
        'ttr': ttr,
        'avg_word_length': avg_word_length,
        'most_common': word_freq.most_common(20)
    }


def analyze_style_markers(texts: List[str]) -> Dict:
    """Detect speaking style markers."""

    markers = {
        'questions': 0,
        'exclamations': 0,
        'first_person': 0,
        'second_person': 0,
        'contractions': 0,
        'technical_terms': 0,
        'casual_terms': 0,
        'hedging': 0,
        'certainty': 0,
    }

    # Patterns
    question_pattern = re.compile(r'\?')
    exclamation_pattern = re.compile(r'!')
    first_person_pattern = re.compile(r'\b(i|i\'m|i\'ll|i\'ve|my|mine|me)\b', re.IGNORECASE)
    second_person_pattern = re.compile(r'\b(you|you\'re|you\'ll|you\'ve|your|yours)\b', re.IGNORECASE)
    contraction_pattern = re.compile(r'\b\w+\'[a-z]+\b')

    # Technical vs casual
    technical_pattern = re.compile(r'\b(algorithm|function|implementation|architecture|optimization|parameter)\b', re.IGNORECASE)
    casual_pattern = re.compile(r'\b(yeah|nope|gonna|wanna|kinda|sorta|cool|awesome|lol)\b', re.IGNORECASE)

    # Hedging vs certainty
    hedging_pattern = re.compile(r'\b(maybe|perhaps|possibly|might|could|seems|appears|probably)\b', re.IGNORECASE)
    certainty_pattern = re.compile(r'\b(definitely|certainly|absolutely|clearly|obviously|exactly)\b', re.IGNORECASE)

    for text in texts:
        markers['questions'] += len(question_pattern.findall(text))
        markers['exclamations'] += len(exclamation_pattern.findall(text))
        markers['first_person'] += len(first_person_pattern.findall(text))
        markers['second_person'] += len(second_person_pattern.findall(text))
        markers['contractions'] += len(contraction_pattern.findall(text))
        markers['technical_terms'] += len(technical_pattern.findall(text))
        markers['casual_terms'] += len(casual_pattern.findall(text))
        markers['hedging'] += len(hedging_pattern.findall(text))
        markers['certainty'] += len(certainty_pattern.findall(text))

    return markers


def analyze_personality_indicators(texts: List[str]) -> Dict:
    """Detect personality traits from language patterns."""

    indicators = {
        'creative': 0,
        'analytical': 0,
        'empathetic': 0,
        'assertive': 0,
        'playful': 0,
        'formal': 0,
    }

    # Creative indicators
    creative_pattern = re.compile(r'\b(imagine|creative|idea|innovative|unique|original|artistic)\b', re.IGNORECASE)

    # Analytical indicators
    analytical_pattern = re.compile(r'\b(analyze|data|logic|reason|evidence|conclude|therefore|because)\b', re.IGNORECASE)

    # Empathetic indicators
    empathetic_pattern = re.compile(r'\b(understand|feel|empathy|care|support|help|listen)\b', re.IGNORECASE)

    # Assertive indicators
    assertive_pattern = re.compile(r'\b(will|must|should|need to|important|critical|essential)\b', re.IGNORECASE)

    # Playful indicators
    playful_pattern = re.compile(r'\b(fun|play|enjoy|exciting|adventure|explore|experiment)\b|[!]{2,}|😄|😊|🎉', re.IGNORECASE)

    # Formal indicators
    formal_pattern = re.compile(r'\b(however|furthermore|moreover|thus|hence|therefore|consequently)\b', re.IGNORECASE)

    for text in texts:
        indicators['creative'] += len(creative_pattern.findall(text))
        indicators['analytical'] += len(analytical_pattern.findall(text))
        indicators['empathetic'] += len(empathetic_pattern.findall(text))
        indicators['assertive'] += len(assertive_pattern.findall(text))
        indicators['playful'] += len(playful_pattern.findall(text))
        indicators['formal'] += len(formal_pattern.findall(text))

    return indicators


def calculate_quality_score(
    num_samples: int,
    avg_length: float,
    vocab_ttr: float,
    pattern_diversity: int
) -> Tuple[float, str, List[str]]:
    """
    Calculate dataset quality score and warnings.

    Returns: (score, grade, warnings)
    """
    score = 0.0
    warnings = []

    # Sample size scoring (0-40 points)
    if num_samples >= 1000:
        score += 40
    elif num_samples >= 500:
        score += 30
        warnings.append("Dataset size adequate but more samples would improve quality")
    elif num_samples >= 100:
        score += 20
        warnings.append("Dataset size small - consider collecting more conversations")
    else:
        score += 10
        warnings.append("⚠️  Dataset size very small - training may overfit")

    # Response length scoring (0-20 points)
    if 50 <= avg_length <= 500:
        score += 20
    elif 30 <= avg_length < 50:
        score += 15
        warnings.append("Average response length short - may limit personality depth")
    elif avg_length > 500:
        score += 15
        warnings.append("Average response length very long - consider truncation")
    else:
        score += 5
        warnings.append("⚠️  Average response length too short for effective training")

    # Vocabulary richness (0-20 points)
    if vocab_ttr >= 0.5:
        score += 20
    elif vocab_ttr >= 0.3:
        score += 15
    elif vocab_ttr >= 0.2:
        score += 10
        warnings.append("Vocabulary diversity moderate - responses may be repetitive")
    else:
        score += 5
        warnings.append("⚠️  Low vocabulary diversity - dataset may be too repetitive")

    # Pattern diversity (0-20 points)
    if pattern_diversity >= 200:
        score += 20
    elif pattern_diversity >= 100:
        score += 15
    elif pattern_diversity >= 50:
        score += 10
        warnings.append("Pattern diversity moderate - personality may be limited")
    else:
        score += 5
        warnings.append("⚠️  Low pattern diversity - may not capture personality well")

    # Grade
    if score >= 85:
        grade = "A - Excellent"
    elif score >= 70:
        grade = "B - Good"
    elif score >= 55:
        grade = "C - Fair"
    elif score >= 40:
        grade = "D - Poor"
    else:
        grade = "F - Insufficient"

    return score, grade, warnings


# ============================================================================
# Prediction & Display
# ============================================================================

def predict_learned_patterns(
    ngrams: List[Tuple[str, int]],
    starters: List[Tuple[str, int]],
    style: Dict,
    personality: Dict,
    vocab: Dict
) -> Dict:
    """Predict what the LoRA will learn."""

    predictions = {
        'speaking_style': [],
        'personality_traits': [],
        'common_patterns': [],
        'vocabulary_level': '',
        'formality': '',
    }

    # Speaking style
    total_markers = sum(style.values())

    if style['questions'] > total_markers * 0.15:
        predictions['speaking_style'].append("Questioning/Inquisitive")

    if style['exclamations'] > total_markers * 0.1:
        predictions['speaking_style'].append("Energetic/Emphatic")

    if style['contractions'] > total_markers * 0.1:
        predictions['speaking_style'].append("Casual/Conversational")

    if style['technical_terms'] > style['casual_terms']:
        predictions['speaking_style'].append("Technical/Professional")
    elif style['casual_terms'] > style['technical_terms']:
        predictions['speaking_style'].append("Casual/Informal")

    if style['hedging'] > style['certainty']:
        predictions['speaking_style'].append("Cautious/Hedging")
    elif style['certainty'] > style['hedging']:
        predictions['speaking_style'].append("Confident/Certain")

    # Personality traits
    total_personality = sum(personality.values())
    sorted_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)

    for trait, count in sorted_traits[:3]:
        if count > total_personality * 0.15:
            predictions['personality_traits'].append(trait.capitalize())

    # Common patterns (top n-grams)
    predictions['common_patterns'] = [phrase for phrase, _ in ngrams[:10]]

    # Vocabulary level
    if vocab['avg_word_length'] > 6:
        predictions['vocabulary_level'] = "Advanced/Academic"
    elif vocab['avg_word_length'] > 5:
        predictions['vocabulary_level'] = "Moderate/Professional"
    else:
        predictions['vocabulary_level'] = "Simple/Accessible"

    # Formality
    formality_score = personality.get('formal', 0) / (total_personality or 1)
    if formality_score > 0.2:
        predictions['formality'] = "Formal"
    elif formality_score > 0.1:
        predictions['formality'] = "Semi-formal"
    else:
        predictions['formality'] = "Informal"

    return predictions


def display_preview(
    data: List[Dict],
    responses: List[str],
    ngrams: List[Tuple[str, int]],
    starters: List[Tuple[str, int]],
    vocab: Dict,
    style: Dict,
    personality: Dict,
    predictions: Dict,
    quality_score: float,
    quality_grade: str,
    warnings: List[str]
):
    """Display comprehensive pattern preview."""

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Text Pattern Preview - What Will This LoRA Learn?[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Dataset Overview
    console.print("[bold yellow]═══ Dataset Overview ═══[/bold yellow]")

    avg_response_length = sum(len(r) for r in responses) / len(responses) if responses else 0

    overview_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    overview_table.add_column("Metric", style="cyan")
    overview_table.add_column("Value", style="green")

    overview_table.add_row("Training Samples", str(len(data)))
    overview_table.add_row("Total Responses", str(len(responses)))
    overview_table.add_row("Avg Response Length", f"{avg_response_length:.0f} chars")
    overview_table.add_row("Total Words", f"{vocab['total_words']:,}")
    overview_table.add_row("Unique Words", f"{vocab['unique_words']:,}")
    overview_table.add_row("Vocabulary Richness (TTR)", f"{vocab['ttr']:.3f}")
    overview_table.add_row("Avg Word Length", f"{vocab['avg_word_length']:.1f} chars")

    console.print(overview_table)
    console.print()

    # Quality Score
    console.print("[bold yellow]═══ Dataset Quality ═══[/bold yellow]")

    quality_color = "green" if quality_score >= 70 else "yellow" if quality_score >= 55 else "red"
    console.print(f"Quality Score: [{quality_color}]{quality_score:.1f}/100 - {quality_grade}[/{quality_color}]")
    console.print()

    if warnings:
        console.print("[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")
        console.print()

    # Predictions
    console.print("[bold yellow]═══ Predicted Patterns ═══[/bold yellow]")
    console.print()

    console.print(f"[cyan]Speaking Style:[/cyan] {', '.join(predictions['speaking_style']) if predictions['speaking_style'] else 'Neutral'}")
    console.print(f"[cyan]Personality Traits:[/cyan] {', '.join(predictions['personality_traits']) if predictions['personality_traits'] else 'Neutral'}")
    console.print(f"[cyan]Vocabulary Level:[/cyan] {predictions['vocabulary_level']}")
    console.print(f"[cyan]Formality:[/cyan] {predictions['formality']}")
    console.print()

    # Most Common Phrases (what LoRA will emphasize)
    console.print("[bold yellow]═══ Common Phrases (Will Be Emphasized) ═══[/bold yellow]")

    phrase_table = Table(box=box.ROUNDED, border_style="cyan")
    phrase_table.add_column("Phrase", style="yellow", width=40)
    phrase_table.add_column("Count", justify="right", style="green")
    phrase_table.add_column("Frequency", justify="right", style="dim")

    total_ngrams = sum(count for _, count in ngrams)
    for phrase, count in ngrams[:15]:
        freq = (count / total_ngrams * 100) if total_ngrams > 0 else 0
        phrase_table.add_row(phrase, str(count), f"{freq:.1f}%")

    console.print(phrase_table)
    console.print()

    # Sentence Starters (response patterns)
    console.print("[bold yellow]═══ Typical Response Starters ═══[/bold yellow]")

    starter_table = Table(box=box.ROUNDED, border_style="cyan")
    starter_table.add_column("How Responses Start", style="yellow", width=40)
    starter_table.add_column("Count", justify="right", style="green")

    for starter, count in starters[:12]:
        starter_table.add_row(starter, str(count))

    console.print(starter_table)
    console.print()

    # Style Markers
    console.print("[bold yellow]═══ Style Markers ═══[/bold yellow]")

    style_table = Table(box=box.ROUNDED, border_style="cyan")
    style_table.add_column("Marker", style="cyan")
    style_table.add_column("Count", justify="right", style="green")

    for marker, count in sorted(style.items(), key=lambda x: x[1], reverse=True):
        style_table.add_row(marker.replace('_', ' ').title(), str(count))

    console.print(style_table)
    console.print()

    # Top Vocabulary
    console.print("[bold yellow]═══ Most Used Words ═══[/bold yellow]")

    top_words = ' '.join([f"[yellow]{word}[/yellow]({count})" for word, count in vocab['most_common'][:30]])
    console.print(top_words)
    console.print()

    # Final Summary
    console.print(Panel.fit(
        "[bold green]Training Recommendation:[/bold green]\n\n" +
        (f"✓ Dataset quality is {quality_grade.split(' - ')[1].lower()}. " if quality_score >= 70 else
         f"⚠ Dataset quality is {quality_grade.split(' - ')[1].lower()}. Consider improving data before training. ") +
        f"\n\n[dim]This LoRA will learn to write with: {', '.join(predictions['speaking_style'][:2]) if len(predictions['speaking_style']) >= 2 else 'a neutral style'}[/dim]",
        border_style="green" if quality_score >= 70 else "yellow"
    ))
    console.print()


# ============================================================================
# Main
# ============================================================================

def preview_patterns(jsonl_file: str, verbose: bool = False):
    """
    Complete pattern preview analysis.
    Shows what the LoRA will learn BEFORE training.
    """

    console.print(f"[cyan]Loading training data from {jsonl_file}...[/cyan]")
    data = load_training_data(jsonl_file)

    if not data:
        console.print("[red]Error: No training data found[/red]")
        return

    responses = extract_responses(data)

    if not responses:
        console.print("[red]Error: No responses found in data[/red]")
        return

    console.print(f"[green]✓ Loaded {len(data)} samples with {len(responses)} responses[/green]")
    console.print()

    # Run analyses
    console.print("[cyan]Analyzing patterns...[/cyan]")

    ngrams = analyze_ngrams(responses, n=3, top_k=50)
    starters = analyze_sentence_starters(responses, top_k=30)
    vocab = analyze_vocabulary(responses)
    style = analyze_style_markers(responses)
    personality = analyze_personality_indicators(responses)

    # Generate predictions
    predictions = predict_learned_patterns(ngrams, starters, style, personality, vocab)

    # Calculate quality
    avg_length = sum(len(r) for r in responses) / len(responses)
    quality_score, quality_grade, warnings = calculate_quality_score(
        num_samples=len(data),
        avg_length=avg_length,
        vocab_ttr=vocab['ttr'],
        pattern_diversity=len(ngrams)
    )

    # Display
    display_preview(
        data=data,
        responses=responses,
        ngrams=ngrams,
        starters=starters,
        vocab=vocab,
        style=style,
        personality=personality,
        predictions=predictions,
        quality_score=quality_score,
        quality_grade=quality_grade,
        warnings=warnings
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preview what patterns a text LoRA will learn - BEFORE training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool analyzes your training data and predicts:
  • Speaking style and personality traits
  • Common phrases that will be emphasized
  • Vocabulary level and complexity
  • Dataset quality and sufficiency warnings

Like gap_analysis.py for images, but for TEXT training data.

Examples:
  # Preview before training
  python text_pattern_preview.py training_data.jsonl

  # Save to file
  python text_pattern_preview.py training_data.jsonl > preview_report.txt

Use this BEFORE running text_lora_train.py to understand what you're about to train.
        """
    )

    parser.add_argument("jsonl_file", help="Training data JSONL file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    preview_patterns(args.jsonl_file, verbose=args.verbose)


if __name__ == "__main__":
    main()
