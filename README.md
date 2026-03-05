# C# Model Trainer

A pipeline for building high-quality C# instruction datasets and fine-tuning large language models (e.g., Qwen3.5-9B) for C# code generation.

## Overview

This project provides a complete pipeline from raw code collection to fine-tuned model:

```
The Stack v2 (C# slice)          GitHub (quality-filtered repos)
        │                                    │
  stack_v2_download.py            github_scraper.py
        │                                    │
        └───────┬────────────────────────────┘
              │
      synthetic_instruct.py
              │
    ┌─────────┴─────────┐
    │                   │
Alpaca format      ChatML format
(for TRL)          (for Unsloth)
```

## Quick Start

### Prerequisites

```bash
pip install datasets huggingface_hub PyGithub requests tqdm python-dotenv anthropic transformers
```

### Step 1: Download The Stack v2 (C#)

```bash
# First, accept the dataset license at:
# https://huggingface.co/datasets/bigcode/the-stack-v2
huggingface-cli login

python src/data/stack_v2_download.py --output_dir ./data/stack_v2_csharp
```

### Step 2: Scrape GitHub for High-Quality C# Repos

```bash
export GITHUB_TOKEN=ghp_your_token_here

python src/data/github_scraper.py \
    --output_dir ./data/github_csharp \
    --max_repos 1000 \
    --require_ci \
    --require_tests
```

### Step 3: Generate Synthetic Instruction Pairs

```bash
export ANTHROPIC_API_KEY=sk-ant-your_key_here

python src/data/synthetic_instruct.py \
    --input_dirs ./data/stack_v2_csharp ./data/github_csharp \
    --output_dir ./data/synthetic_instruct \
    --max_pairs 50000 \
    --format both
```

### Step 4: Fine-Tune Your Model

```bash
python src/scripts/train.py
```

### Step 5: Evaluate

```bash
python src/scripts/evaluate.py
```

## Pipeline Details

### Quality Filters Applied

- **Size**: 200-50,000 characters
- **Line count**: 10+ lines
- **Line length**: Max 500 chars, avg < 150 chars
- **Alphanumeric ratio**: > 40%
- **Comment ratio**: < 70%
- **Auto-generated code**: Filtered via pattern matching
- **Designer/generated files**: Excluded (.designer.cs, .g.cs, etc.)

### Modern C# Detection

Files are scored higher if they use:
- Null-forgiving operator (`!`)
- Record types (`record`)
- Init accessors (`init`)
- File-scoped types (`file class`)
- Async/await
- LINQ
- Dependency injection patterns

## Output Formats

- **Alpaca format** (`csharp_instruct_alpaca.jsonl`): For TRL / LLaMA-Factory
- **ChatML format** (`csharp_instruct_chatml.jsonl`): For Unsloth / most fine-tuners

## Recommended Dataset Sizes

| Stage | Target Size |
|-------|-------------|
| Stack v2 raw files | 200K–500K |
| GitHub scraped files | 50K–100K |
| Synthetic instruction pairs | 30K–100K |

## License

MIT License
