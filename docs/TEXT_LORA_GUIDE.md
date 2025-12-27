# Text LoRA Training Guide

Complete textual identity system for AI Home - train personality LoRAs from conversation data.

## Overview

The text LoRA training system complements visual identity by fine-tuning language models on filtered conversation data. This creates a **complete embodiment**: how the AI looks AND how it speaks.

### Pipeline Components

1. **Conversation Parser** (`conversation_parser.py`)
   - Multi-format support: Claude Desktop, ChatGPT, generic JSON
   - Auto-detects format from file structure
   - Exports to JSONL for training

2. **Conversation Filter** (`conversation_filter.py`)
   - Blacklist filtering (corporate speak, AI slop)
   - Toxicity detection with detoxify
   - Quality scoring and filtering
   - Statistical reporting

3. **Text LoRA Trainer** (`text_lora_train.py`)
   - Unsloth or PEFT/Transformers backend
   - Model presets: Llama-3, Mistral, Qwen
   - Efficient 4-bit/8-bit training
   - Ready for integration with RL frameworks

4. **Identity Pipeline Integration** (`identity_pipeline.py`)
   - Phase 2B: Textual Fulfillment
   - Phase 3B: Textual Embodiment
   - Complete visual + textual identity orchestration

## Quick Start

### 1. Parse Conversations

```bash
# Claude Desktop export
python conversation_parser.py conversations.json --output parsed.json --stats

# ChatGPT export
python conversation_parser.py chatgpt_export.json --output parsed.json
```

### 2. Filter Content

```bash
# Filter with default settings
python conversation_filter.py parsed.json --output filtered.json

# Custom blacklist patterns
python conversation_filter.py parsed.json --output filtered.json \
  --blacklist "pattern1" "pattern2" --toxicity-threshold 0.7

# Disable AI slop filtering
python conversation_filter.py parsed.json --output filtered.json --no-ai-slop
```

### 3. Export for Training

```bash
# Export to JSONL format
python conversation_parser.py filtered.json --export training_data.jsonl --format jsonl
```

### 4. Train Text LoRA

```bash
# Train Llama-3 personality LoRA
python text_lora_train.py training_data.jsonl --preset llama-3-8b --output ./text_lora

# Train Mistral with custom epochs
python text_lora_train.py training_data.jsonl --preset mistral-7b --epochs 5

# Use PEFT instead of Unsloth
python text_lora_train.py training_data.jsonl --preset llama-3-8b --no-unsloth
```

### 5. Complete Identity Pipeline

```bash
# Full visual + text pipeline
python identity_pipeline.py llama3.2 \
  --iterations 200 \
  --count 100 \
  --conversations my_chats.json \
  --text-preset llama-3-8b

# Text only (with existing traits)
python identity_pipeline.py llama3.2 \
  --skip-probe \
  --traits traits.json \
  --conversations chats.json \
  --text-preset mistral-7b
```

## Conversation Formats

### Claude Desktop Format

```json
{
  "uuid": "conversation-id",
  "name": "Conversation Title",
  "chat_messages": [
    {
      "sender": "human",
      "text": "User message",
      "created_at": "2024-01-01T00:00:00Z"
    },
    {
      "sender": "assistant",
      "text": "AI response",
      "created_at": "2024-01-01T00:00:01Z"
    }
  ]
}
```

### ChatGPT Format

```json
{
  "title": "Conversation Title",
  "mapping": {
    "message-id": {
      "message": {
        "author": {
          "role": "user"
        },
        "content": {
          "parts": ["Message text"]
        }
      }
    }
  }
}
```

### Generic Format

```json
[
  {
    "role": "user",
    "content": "User message"
  },
  {
    "role": "assistant",
    "content": "AI response"
  }
]
```

## Content Filtering

### Default Blacklist

Corporate speak patterns that get filtered:
- "not nothing"
- "hold space", "holding space"
- "unpack that"
- "circle back"
- "touch base"
- "synergize"
- "leverage" (when used as jargon)
- "think outside the box"
- "low-hanging fruit"
- "move the needle"
- "paradigm shift"

### AI Slop Patterns

Overused AI assistant patterns:
- "I appreciate you"
- "I hear you"
- "that resonates"
- "I'm here for you" (when inappropriate)
- "validate your feelings"
- "in this space" (vague corporate speak)

### Toxicity Detection

Uses detoxify model to detect:
- Toxicity (threshold: 0.8)
- Severe toxicity
- Obscene content
- Threats
- Insults

Install detoxify: `pip install detoxify`

### Quality Filtering

Removes:
- Empty messages
- Single-word responses ("ok", "yes", "thanks")
- Messages shorter than 10 characters (configurable)
- Just ellipsis ("...")

## Model Presets

### Llama-3-8B
```
LoRA Rank: 16
LoRA Alpha: 16
Learning Rate: 2e-4
Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

### Mistral-7B
```
LoRA Rank: 16
LoRA Alpha: 16
Learning Rate: 2e-4
Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

### Qwen-2.5-7B
```
LoRA Rank: 16
LoRA Alpha: 16
Learning Rate: 2e-4
Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

List available presets:
```bash
python text_lora_train.py --list-presets
```

## Training Backend

### Unsloth (Recommended)

Faster training with optimizations:
- 2x faster training
- 50% less memory usage
- Built-in 4-bit quantization

Install: `pip install unsloth`

### PEFT/Transformers (Fallback)

Standard training with:
- 8-bit quantization
- LoRA adapters via PEFT
- SFTTrainer from TRL

Automatically used if Unsloth not available.

## Output Format

Training exports JSONL with instruction/response pairs:

```json
{"instruction": "User question", "response": "AI answer", "conversation_id": "conv-123"}
{"instruction": "Another question", "response": "Another answer", "conversation_id": "conv-123"}
```

Compatible with:
- Unsloth
- Axolotl
- Standard Transformers SFTTrainer

## Integration with AI Home

The complete identity system creates:

**Visual Identity** (Character LoRA):
- How the AI looks in generated images
- Self-portrait dataset from self-concept probe
- Trained on SDXL/Flux/ZIT

**Textual Identity** (Personality LoRA):
- How the AI speaks in text
- Filtered conversation data
- Trained on Llama/Mistral/Qwen

**RL Framework Integration**:
- Both LoRAs packaged for Atropos
- Dream images + personality tuning
- Complete embodied AI support

## Advanced Usage

### Custom Blacklist

```bash
# Add custom patterns
python conversation_filter.py input.json --output filtered.json \
  --blacklist "my_pattern" "another_pattern"
```

### Adjust Toxicity Threshold

```bash
# More aggressive filtering (0.5 threshold)
python conversation_filter.py input.json --output filtered.json \
  --toxicity-threshold 0.5
```

### Export Multiple Formats

```bash
# Export both JSON and JSONL
python conversation_parser.py input.json --output parsed.json
python conversation_parser.py parsed.json --export training.jsonl --format jsonl
```

### Custom Training Config

Create custom preset in `text_lora_train.py`:

```python
"my-model": {
    "base_model": "path/to/model",
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj"],
    "learning_rate": 1e-4,
    "batch_size": 2,
    "max_seq_length": 4096,
}
```

## Troubleshooting

### "detoxify not available"

```bash
pip install detoxify
```

### "unsloth not available"

Unsloth is optional. Training will use PEFT fallback:
```bash
pip install peft trl transformers
```

### GPU Memory Issues

Reduce batch size:
```bash
# Edit text_lora_train.py preset
"batch_size": 2  # Lower from 4
```

Or use gradient accumulation (automatically set to 4 steps).

### Conversation Format Not Detected

Use generic JSON format:
```json
[
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]
```

## Statistics Example

Filtering output shows:

```
╭─────────────────────────────────────╮
│   Filtering Statistics              │
╰─────────────────────────────────────╯

Category                 Count    %
─────────────────────────────────────
Total Messages           1000     100%
Kept                     850      85.0%

Filtered - Blacklist     80       8.0%
Filtered - Toxicity      20       2.0%
Filtered - Low Quality   30       3.0%
Filtered - Too Short     20       2.0%

Top Blacklist Matches:
  • \bnot nothing\b: 35
  • \bhold(?:ing)? space\b: 25
  • \bcircle back\b: 15
```

## Next Steps

After training your text LoRA:

1. Test the personality LoRA with inference
2. Combine with character LoRA for complete identity
3. Integrate into AI Home support network
4. Use with RL frameworks (Atropos) for continued learning

---

**Part of Eager Beaver - AI Home Support Network**

Each tool is a strap in the support network for the models in AI Home.
