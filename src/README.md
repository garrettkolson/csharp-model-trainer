# C# Fine-Tuning Dataset Pipeline

Three scripts to build a high-quality C# instruction dataset for fine-tuning
a code generation model (e.g. Qwen3.5-9B).

## Pipeline Overview

```
The Stack v2 (C# slice)          GitHub (quality-filtered repos)
        │                                    │
  01_stack_v2_csharp.py           02_github_scraper.py
        │                                    │
        └──────────────┬─────────────────────┘
                       │
               03_synthetic_instruct.py
                       │
        ┌──────────────┴─────────────────────┐
        │                                    │
  csharp_instruct_alpaca.jsonl    csharp_instruct_chatml.jsonl
  (for TRL/LLaMA-Factory)         (for Unsloth / most fine-tuners)
```

---

## Setup

```bash
pip install datasets huggingface_hub PyGithub requests tqdm python-dotenv anthropic
```

Create a `.env` file:
```
GITHUB_TOKEN=ghp_your_token_here
ANTHROPIC_API_KEY=sk-ant-your_key_here
```

---

## Script 1: The Stack v2

```bash
# Accept dataset license at https://huggingface.co/datasets/bigcode/the-stack-v2
huggingface-cli login

python 01_stack_v2_csharp.py --output_dir ./data/stack_v2_csharp

# Test run (first 5000 files only)
python 01_stack_v2_csharp.py --output_dir ./data/stack_v2_csharp --max_samples 5000
```

**Output:** JSONL shards in `./data/stack_v2_csharp/`
**Expected pass rate:** ~40-60% of raw files pass quality filters
**Runtime:** Several hours (streaming, no large download needed)

---

## Script 2: GitHub Scraper

```bash
python 02_github_scraper.py \
    --output_dir ./data/github_csharp \
    --max_repos 1000 \
    --require_ci \
    --require_tests \
    --modern_only
```

**Flags:**
- `--require_ci` — only repos with GitHub Actions workflows (default: on)
- `--require_tests` — only repos with test directories (default: on)
- `--modern_only` — only files using C# 8+ features (default: off)
- `--max_repos` — cap total repos scraped (default: 2000)

**Output:** JSONL shards in `./data/github_csharp/`
**Rate limits:** ~5000 GitHub API requests/hour with a token. The script
handles this automatically.

---

## Script 3: Synthetic Instruction Generation

```bash
python 03_synthetic_instruct.py \
    --input_dirs ./data/stack_v2_csharp ./data/github_csharp \
    --output_dir ./data/synthetic_instruct \
    --max_pairs 50000 \
    --format both
```

**Flags:**
- `--max_pairs` — target number of (instruction, code) pairs
- `--format` — `alpaca`, `chatml`, or `both`
- `--resume` — restart without re-generating already-written pairs
- `--max_chunks_per_file` — limit methods extracted per source file (default: 3)

**Output:**
- `csharp_instruct_alpaca.jsonl` — for TRL / LLaMA-Factory
- `csharp_instruct_chatml.jsonl` — for Unsloth / most fine-tuners
- `csharp_instruct_raw.jsonl` — raw with full metadata

**Cost estimate:** At ~300 tokens/call average, 50K pairs ≈ 15M tokens
≈ ~$45 at claude-sonnet-4-6 pricing. Run a 500-pair test first.

---

## Recommended Dataset Sizes

| Stage | Source | Target Size |
|-------|--------|-------------|
| Foundation | The Stack v2 (C# slice) | 200K–500K files |
| High quality | GitHub scraper | 50K–100K files |
| Instruction tuning | Synthetic pairs | 30K–100K pairs |

For fine-tuning, the **synthetic pairs** are what matter most for
instruction-following quality. The raw code files from scripts 1 and 2
can also be used for continued pre-training before the instruction SFT step.

---

## Fine-Tuning Next Steps

After generating your dataset, fine-tune with Unsloth:

```bash
pip install unsloth

# Use the chatml output directly with Unsloth's conversational format
# See: https://docs.unsloth.ai/basics/qwen3-how-to-run-and-fine-tune
```

Use `csharp_instruct_chatml.jsonl` for supervised fine-tuning with Unsloth,
or `csharp_instruct_alpaca.jsonl` for TRL/LLaMA-Factory.
