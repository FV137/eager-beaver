#!/usr/bin/env python3
"""
Conversation Quality Scorer - Assess training data quality.

Like blur detection and face quality for images, but for text conversations.
Scores individual messages and conversation pairs on multiple quality dimensions.

Quality Dimensions:
- Response length appropriateness
- Information density (not just filler words)
- Specificity vs vagueness
- Engagement level
- Coherence and relevance
- Question quality

Helps identify low-quality training samples before expensive training.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich import box

console = Console()


# ============================================================================
# Pre-compiled Regex Patterns for Performance
# ============================================================================

# Filler word patterns
FILLER_PATTERNS = [
    re.compile(r'\bum\b', re.IGNORECASE),
    re.compile(r'\buh\b', re.IGNORECASE),
    re.compile(r'\blike\b', re.IGNORECASE),
    re.compile(r'\byou know\b', re.IGNORECASE),
    re.compile(r'\bI mean\b', re.IGNORECASE),
    re.compile(r'\bkind of\b', re.IGNORECASE),
    re.compile(r'\bsort of\b', re.IGNORECASE),
    re.compile(r'\bjust\b', re.IGNORECASE),
    re.compile(r'\breally\b', re.IGNORECASE),
    re.compile(r'\bactually\b', re.IGNORECASE),
    re.compile(r'\bbasically\b', re.IGNORECASE),
]

# Vague patterns
VAGUE_PATTERNS = [
    re.compile(r'\bthing\b', re.IGNORECASE),
    re.compile(r'\bstuff\b', re.IGNORECASE),
    re.compile(r'\bsomething\b', re.IGNORECASE),
    re.compile(r'\bsomehow\b', re.IGNORECASE),
    re.compile(r'\bsomewhat\b', re.IGNORECASE),
    re.compile(r'\bkind of\b', re.IGNORECASE),
    re.compile(r'\bsort of\b', re.IGNORECASE),
    re.compile(r'\bwhatever\b', re.IGNORECASE),
    re.compile(r'\betc\b', re.IGNORECASE),
    re.compile(r'\band so on\b', re.IGNORECASE),
]

# Specific markers
SPECIFIC_PATTERNS = [
    re.compile(r'\b\d+\b'),  # Numbers
    re.compile(r'\b[A-Z][a-z]+\b'),  # Proper nouns
    re.compile(r'"[^"]+"'),  # Quotes
    re.compile(r'\bfor example\b', re.IGNORECASE),
    re.compile(r'\bspecifically\b', re.IGNORECASE),
    re.compile(r'\bnamely\b', re.IGNORECASE),
]

# Emotion patterns
EMOTION_PATTERNS = [
    re.compile(r'\blove\b', re.IGNORECASE),
    re.compile(r'\bhate\b', re.IGNORECASE),
    re.compile(r'\bexcited\b', re.IGNORECASE),
    re.compile(r'\bamazing\b', re.IGNORECASE),
    re.compile(r'\bterrible\b', re.IGNORECASE),
    re.compile(r'\bawesome\b', re.IGNORECASE),
    re.compile(r'\bfrustrat\w+\b', re.IGNORECASE),
    re.compile(r'\bhappy\b', re.IGNORECASE),
    re.compile(r'\bsad\b', re.IGNORECASE),
    re.compile(r'\bangry\b', re.IGNORECASE),
    re.compile(r'\bfear\b', re.IGNORECASE),
]

# Word pattern (used frequently)
WORD_PATTERN = re.compile(r'\b\w+\b')

# Passive voice patterns
PASSIVE_PATTERNS = [
    re.compile(r'\bwas\b', re.IGNORECASE),
    re.compile(r'\bwere\b', re.IGNORECASE),
    re.compile(r'\bbeen\b', re.IGNORECASE),
]

# Active voice patterns
ACTIVE_PATTERNS = [
    re.compile(r'\bI\b'),
    re.compile(r'\bwe\b', re.IGNORECASE),
    re.compile(r'\byou\b', re.IGNORECASE),
]


# ============================================================================
# Quality Metrics - Message Level
# ============================================================================

def score_message_length(text: str) -> Tuple[float, str]:
    """
    Score message length appropriateness.

    Returns: (score 0-100, reason)
    """
    length = len(text)

    if 50 <= length <= 500:
        return 100, "Optimal length"
    elif 30 <= length < 50:
        return 75, "Slightly short"
    elif 500 < length <= 1000:
        return 85, "Good detail"
    elif 20 <= length < 30:
        return 50, "Too brief"
    elif 1000 < length <= 2000:
        return 70, "Very detailed (may be verbose)"
    elif length < 20:
        return 25, "Extremely short"
    else:
        return 40, "Excessively long (>2000 chars)"


def score_information_density(text: str) -> Tuple[float, str]:
    """
    Score information density - not just filler words.

    High density = specific nouns, verbs, meaningful content
    Low density = lots of filler ("um", "like", "you know", etc.)
    """

    # Count fillers using pre-compiled patterns
    filler_count = sum(len(pattern.findall(text)) for pattern in FILLER_PATTERNS)

    # Count total words
    words = WORD_PATTERN.findall(text)
    total_words = len(words)

    if total_words == 0:
        return 0, "Empty"

    # Filler ratio
    filler_ratio = filler_count / total_words

    # Count content words (proper nouns, numbers, longer words)
    content_markers = len(re.findall(r'\b[A-Z][a-z]+\b', text))  # Proper nouns
    content_markers += len(re.findall(r'\b\d+\b', text))  # Numbers
    content_markers += len(re.findall(r'\b\w{7,}\b', text))  # Longer words (often more specific)

    content_ratio = content_markers / total_words

    # Score
    if filler_ratio < 0.05 and content_ratio > 0.3:
        return 100, "High information density"
    elif filler_ratio < 0.1 and content_ratio > 0.2:
        return 85, "Good content"
    elif filler_ratio < 0.15:
        return 70, "Moderate content"
    elif filler_ratio < 0.25:
        return 50, "Some filler"
    else:
        return 30, "Excessive filler words"


def score_specificity(text: str) -> Tuple[float, str]:
    """
    Score specificity vs vagueness.

    Specific: concrete examples, numbers, names, details
    Vague: "thing", "stuff", "something", "kind of"
    """

    words = WORD_PATTERN.findall(text)
    total_words = len(words)

    if total_words == 0:
        return 0, "Empty"

    # Use pre-compiled patterns
    vague_count = sum(len(pattern.findall(text)) for pattern in VAGUE_PATTERNS)
    specific_count = sum(len(pattern.findall(text)) for pattern in SPECIFIC_PATTERNS)

    vague_ratio = vague_count / total_words
    specific_ratio = specific_count / total_words

    # Score
    if specific_ratio > 0.1 and vague_ratio < 0.03:
        return 100, "Highly specific"
    elif specific_ratio > 0.05 and vague_ratio < 0.05:
        return 85, "Good specificity"
    elif vague_ratio < 0.1:
        return 70, "Moderate specificity"
    elif vague_ratio < 0.15:
        return 50, "Somewhat vague"
    else:
        return 30, "Very vague"


def score_engagement(text: str) -> Tuple[float, str]:
    """
    Score engagement level.

    Engaged: questions, emotion, energy, active voice
    Disengaged: passive, minimal, monotone
    """

    # Engagement markers
    questions = len(re.findall(r'\?', text))
    exclamations = len(re.findall(r'!', text))

    # Use pre-compiled emotion patterns
    emotion_count = sum(len(pattern.findall(text)) for pattern in EMOTION_PATTERNS)

    # Use pre-compiled voice patterns
    active_markers = sum(len(pattern.findall(text)) for pattern in ACTIVE_PATTERNS)
    passive_markers = sum(len(pattern.findall(text)) for pattern in PASSIVE_PATTERNS)

    words = len(WORD_PATTERN.findall(text))
    if words == 0:
        return 0, "Empty"

    # Calculate engagement score
    engagement_score = 0

    if questions > 0:
        engagement_score += 20
    if exclamations > 0:
        engagement_score += 15
    if emotion_count > 0:
        engagement_score += 25
    if active_markers > passive_markers:
        engagement_score += 20
    if words > 50:  # Sufficient length to show engagement
        engagement_score += 20

    if engagement_score >= 80:
        return 100, "Highly engaged"
    elif engagement_score >= 60:
        return 85, "Good engagement"
    elif engagement_score >= 40:
        return 70, "Moderate engagement"
    elif engagement_score >= 20:
        return 50, "Limited engagement"
    else:
        return 30, "Disengaged"


def score_message_quality(text: str) -> Dict:
    """
    Complete message quality scoring.

    Returns dict with individual scores and overall score.
    """

    length_score, length_reason = score_message_length(text)
    density_score, density_reason = score_information_density(text)
    specificity_score, specificity_reason = score_specificity(text)
    engagement_score, engagement_reason = score_engagement(text)

    # Weighted average
    overall = (
        length_score * 0.2 +
        density_score * 0.3 +
        specificity_score * 0.3 +
        engagement_score * 0.2
    )

    return {
        'overall': overall,
        'length': {'score': length_score, 'reason': length_reason},
        'density': {'score': density_score, 'reason': density_reason},
        'specificity': {'score': specificity_score, 'reason': specificity_reason},
        'engagement': {'score': engagement_score, 'reason': engagement_reason},
    }


# ============================================================================
# Quality Metrics - Conversation Pair Level
# ============================================================================

def score_question_quality(question: str) -> Tuple[float, str]:
    """Score the quality of the question/instruction."""

    # Good questions are:
    # - Specific (not "tell me about X")
    # - Clear (not ambiguous)
    # - Focused (not multi-part confusion)

    question_lower = question.lower()

    # Check if it's actually a question
    has_question_mark = '?' in question
    has_question_words = any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which'])

    # Vague question markers
    vague_questions = ['tell me about', 'what do you think', 'how do you feel']
    is_vague = any(vague in question_lower for vague in vague_questions)

    # Multi-part questions (often confusing)
    question_marks = question.count('?')
    is_multipart = question_marks > 1 or ' and ' in question_lower and has_question_mark

    # Length
    words = len(re.findall(r'\b\w+\b', question))

    score = 50  # baseline

    if has_question_mark or has_question_words:
        score += 20
    if not is_vague:
        score += 15
    if not is_multipart:
        score += 10
    if 5 <= words <= 20:
        score += 15
    elif words > 20:
        score -= 10  # Too complex

    if score >= 85:
        return 100, "Excellent question"
    elif score >= 70:
        return 85, "Good question"
    elif score >= 55:
        return 70, "Fair question"
    else:
        return 50, "Vague or complex question"


def score_response_relevance(question: str, response: str) -> Tuple[float, str]:
    """
    Score how relevant the response is to the question.

    Crude but useful: keyword overlap, question type matching.
    """

    # Extract key words from question (non-stopwords)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'you', 'your', 'me', 'my', 'i'}

    question_words = set(WORD_PATTERN.findall(question.lower()))
    response_words = set(WORD_PATTERN.findall(response.lower()))

    question_keywords = question_words - stopwords
    response_keywords = response_words - stopwords

    if not question_keywords:
        return 70, "Unable to assess (no keywords in question)"

    # Keyword overlap
    overlap = question_keywords & response_keywords
    overlap_ratio = len(overlap) / len(question_keywords)

    # Check for direct acknowledgment
    acknowledges = any(phrase in response.lower() for phrase in [
        question.lower()[:30],  # Repeats part of question
        'yes', 'no', 'that', 'this', 'it'
    ])

    score = 50  # baseline

    if overlap_ratio > 0.5:
        score += 30
    elif overlap_ratio > 0.3:
        score += 20
    elif overlap_ratio > 0.1:
        score += 10

    if acknowledges:
        score += 20

    if score >= 85:
        return 100, "Highly relevant"
    elif score >= 70:
        return 85, "Good relevance"
    elif score >= 55:
        return 70, "Moderate relevance"
    else:
        return 50, "Limited relevance"


def score_conversation_pair(instruction: str, response: str) -> Dict:
    """Score a complete conversation pair."""

    question_score, question_reason = score_question_quality(instruction)
    response_quality = score_message_quality(response)
    relevance_score, relevance_reason = score_response_relevance(instruction, response)

    # Overall pair quality
    overall = (
        question_score * 0.2 +
        response_quality['overall'] * 0.5 +
        relevance_score * 0.3
    )

    return {
        'overall': overall,
        'question': {'score': question_score, 'reason': question_reason},
        'response': response_quality,
        'relevance': {'score': relevance_score, 'reason': relevance_reason},
    }


# ============================================================================
# Dataset-Level Analysis
# ============================================================================

def analyze_dataset_quality(conversations: List[Dict]) -> Dict:
    """Analyze overall dataset quality."""

    all_scores = []
    length_distribution = []
    low_quality_samples = []

    for i, conv in enumerate(track(conversations, description="Scoring conversations...")):
        instruction = conv.get('instruction', '') or conv.get('question', '')
        response = conv.get('response', '') or conv.get('output', '')

        if not instruction or not response:
            continue

        pair_score = score_conversation_pair(instruction, response)
        all_scores.append(pair_score['overall'])
        length_distribution.append(len(response))

        # Flag low quality
        if pair_score['overall'] < 60:
            low_quality_samples.append({
                'index': i,
                'score': pair_score['overall'],
                'instruction': instruction[:100],
                'response': response[:100],
                'issues': [
                    pair_score['question']['reason'],
                    pair_score['response']['density']['reason'],
                    pair_score['relevance']['reason']
                ]
            })

    # Calculate statistics
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    min_score = min(all_scores) if all_scores else 0
    max_score = max(all_scores) if all_scores else 0

    # Grade distribution
    grades = {
        'A (90-100)': sum(1 for s in all_scores if s >= 90),
        'B (80-89)': sum(1 for s in all_scores if 80 <= s < 90),
        'C (70-79)': sum(1 for s in all_scores if 70 <= s < 80),
        'D (60-69)': sum(1 for s in all_scores if 60 <= s < 70),
        'F (0-59)': sum(1 for s in all_scores if s < 60),
    }

    # Length statistics
    avg_length = sum(length_distribution) / len(length_distribution) if length_distribution else 0

    return {
        'total_samples': len(conversations),
        'scored_samples': len(all_scores),
        'average_score': avg_score,
        'min_score': min_score,
        'max_score': max_score,
        'grade_distribution': grades,
        'average_length': avg_length,
        'low_quality_samples': low_quality_samples[:20],  # Top 20 worst
        'all_scores': all_scores
    }


# ============================================================================
# Display
# ============================================================================

def display_quality_report(analysis: Dict):
    """Display comprehensive quality report."""

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Conversation Quality Report[/bold cyan]",
        border_style="cyan",
        box=box.DOUBLE
    ))
    console.print()

    # Overall Statistics
    console.print("[bold yellow]═══ Dataset Overview ═══[/bold yellow]")

    avg_score = analysis['average_score']
    score_color = "green" if avg_score >= 80 else "yellow" if avg_score >= 70 else "red"

    overview_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    overview_table.add_column("Metric", style="cyan")
    overview_table.add_column("Value", style="green")

    overview_table.add_row("Total Samples", str(analysis['total_samples']))
    overview_table.add_row("Scored Samples", str(analysis['scored_samples']))
    overview_table.add_row("Average Quality Score", f"[{score_color}]{avg_score:.1f}/100[/{score_color}]")
    overview_table.add_row("Score Range", f"{analysis['min_score']:.1f} - {analysis['max_score']:.1f}")
    overview_table.add_row("Average Response Length", f"{analysis['average_length']:.0f} chars")

    console.print(overview_table)
    console.print()

    # Grade Distribution
    console.print("[bold yellow]═══ Quality Distribution ═══[/bold yellow]")

    grade_table = Table(box=box.ROUNDED, border_style="cyan")
    grade_table.add_column("Grade", style="yellow")
    grade_table.add_column("Count", justify="right", style="green")
    grade_table.add_column("Percentage", justify="right", style="dim")

    total = analysis['scored_samples']
    for grade, count in analysis['grade_distribution'].items():
        pct = (count / total * 100) if total > 0 else 0
        grade_table.add_row(grade, str(count), f"{pct:.1f}%")

    console.print(grade_table)
    console.print()

    # Low Quality Samples
    if analysis['low_quality_samples']:
        console.print("[bold yellow]═══ Low Quality Samples (Bottom 20) ═══[/bold yellow]")
        console.print("[dim]Review these for potential removal or improvement[/dim]\n")

        for sample in analysis['low_quality_samples'][:10]:  # Show top 10
            console.print(f"[red]Sample {sample['index']}[/red] - Score: {sample['score']:.1f}")
            console.print(f"[dim]Q: {sample['instruction']}...[/dim]")
            console.print(f"[dim]A: {sample['response']}...[/dim]")
            console.print(f"[yellow]Issues: {', '.join(sample['issues'][:2])}[/yellow]")
            console.print()

    # Recommendations
    console.print("[bold yellow]═══ Recommendations ═══[/bold yellow]")

    recommendations = []

    if avg_score < 70:
        recommendations.append("⚠️  Dataset quality is below recommended threshold")
        recommendations.append("→  Consider filtering out samples with score < 60")

    f_grade_pct = (analysis['grade_distribution']['F (0-59)'] / total * 100) if total > 0 else 0
    if f_grade_pct > 20:
        recommendations.append(f"⚠️  {f_grade_pct:.1f}% of samples are low quality (F grade)")
        recommendations.append("→  Review and remove low quality samples before training")

    if analysis['average_length'] < 50:
        recommendations.append("⚠️  Average response length is very short")
        recommendations.append("→  Short responses may not provide enough personality signal")

    if analysis['average_length'] > 1000:
        recommendations.append("⚠️  Average response length is very long")
        recommendations.append("→  Consider truncating or splitting long responses")

    if avg_score >= 80:
        recommendations.append("✓  Dataset quality is excellent - ready for training!")
    elif avg_score >= 70:
        recommendations.append("✓  Dataset quality is good - should train well")

    for rec in recommendations:
        console.print(f"  {rec}")

    console.print()

    # Training Recommendation
    if avg_score >= 80:
        training_rec = "[bold green]RECOMMENDED TO TRAIN[/bold green]\nDataset quality is excellent."
        border_color = "green"
    elif avg_score >= 70:
        training_rec = "[bold yellow]ACCEPTABLE FOR TRAINING[/bold yellow]\nDataset quality is good. Consider filtering low-quality samples."
        border_color = "yellow"
    else:
        training_rec = "[bold red]NOT RECOMMENDED[/bold red]\nDataset quality is too low. Filter and improve data before training."
        border_color = "red"

    console.print(Panel.fit(
        training_rec,
        border_style=border_color
    ))
    console.print()


# ============================================================================
# Export Low Quality Samples
# ============================================================================

def export_low_quality_samples(analysis: Dict, output_file: str):
    """Export low quality samples for review."""

    low_quality = analysis['low_quality_samples']

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(low_quality, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Exported {len(low_quality)} low quality samples to {output_file}[/green]")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Score conversation quality for text LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Like blur detection for images - scores conversation quality BEFORE training.

Scoring Dimensions:
  • Response length appropriateness
  • Information density (not just filler)
  • Specificity vs vagueness
  • Engagement level
  • Question quality
  • Response relevance

Examples:
  # Score training data
  python conversation_quality.py training.jsonl

  # Export low quality samples for review
  python conversation_quality.py training.jsonl --export-low-quality bad_samples.json

  # Get detailed per-sample scores
  python conversation_quality.py training.jsonl --verbose

Use this BEFORE training to ensure data quality!
        """
    )

    parser.add_argument("data_file", help="Training data JSONL file")
    parser.add_argument("--export-low-quality", help="Export low quality samples to file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show per-sample scores")

    args = parser.parse_args()

    # Load data
    console.print(f"[cyan]Loading training data from {args.data_file}...[/cyan]")

    conversations = []
    with open(args.data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    console.print(f"[green]✓ Loaded {len(conversations)} conversation pairs[/green]\n")

    # Analyze
    analysis = analyze_dataset_quality(conversations)

    # Display report
    display_quality_report(analysis)

    # Export low quality if requested
    if args.export_low_quality:
        export_low_quality_samples(analysis, args.export_low_quality)


if __name__ == "__main__":
    main()
