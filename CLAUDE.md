# CLAUDE.md - C# Model Trainer Project Guide

## Project Structure

```
csharp-model-trainer/
├── src/
│   ├── data/
│   │   ├── stack_v2_download.py      # Script 1: Download & filter Stack v2 C#
│   │   ├── github_scraper.py         # Script 2: Scrape high-quality GitHub repos
│   │   └── synthetic_instruct.py     # Script 3: Generate instruction pairs
│   ├── scripts/
│   │   ├── train.py                  # Fine-tuning script
│   │   └── evaluate.py               # Evaluation script
│   └── configs/
│       └── training_args.json        # Training configuration
└── README.md                         # User-facing documentation
```

## Key Variables & Paths

| Variable | Purpose | Default |
|----------|---------|---------|
| `MODEL_NAME` | Base model for fine-tuning | `Qwen/Qwen3.5-9B` |
| `DATA_PATH` | Input data for training | `../data/csharp_code.txt` |
| `output_dir` | Model output path | `../outputs/qwen-csharp-specialized` |

## Script Descriptions

### stack_v2_download.py
- Downloads The Stack v2 dataset (streaming, ~67TB available but only C# subset)
- Applies quality filters
- Outputs JSONL shards in `./data/stack_v2_csharp/`

### github_scraper.py
- Searches GitHub for high-quality C# repos using queries with stars, pushed date, and topics
- Uses GitHub REST API (requires `GITHUB_TOKEN` env var)
- Applies same quality filters + CI/test detection
- Outputs JSONL shards in `./data/github_csharp/`

### synthetic_instruct.py
- Extracts methods/classes from source files
- Uses Anthropic API to generate instruction pairs (OSS-Instruct style)
- Outputs both Alpaca and ChatML formats
- Supports `--resume` flag for resuming long-running jobs

## Environment Variables

| Variable | Required For | Description |
|----------|--------------|-------------|
| `GITHUB_TOKEN` | github_scraper.py | GitHub API token (repo:read scope) |
| `ANTHROPIC_API_KEY` | synthetic_instruct.py | Anthropic API key for instruction generation |

## Common Workflows

### Building a Dataset
1. Run `stack_v2_download.py` (hours, streaming)
2. Run `github_scraper.py` (depends on API rate limits)
3. Run `synthetic_instruct.py` (most expensive, uses Claude API)

### Fine-Tuning
1. Convert outputs to appropriate format
2. Run `train.py` with configured training args
3. Save to `../outputs/qwen-csharp-specialized`

### Evaluation
1. Run `evaluate.py` with prompts of interest
2. Model loads from `../outputs/qwen-sharp`

## Quality Filter Heuristics

All three scripts share these filters:
- `MIN_CHARS = 200`, `MAX_CHARS = 50_000`
- `MIN_LINES = 10`, `MAX_LINE_LENGTH = 500`
- `MAX_AVG_LINE_LENGTH = 150`, `MIN_ALPHANUM_RATIO = 0.4`
- `MAX_COMMENT_RATIO = 0.7`

## Cost Optimization Tips

1. Test with `--max_samples 5000` first
2. Use `--resume` flag for long-running jobs
3. Run instruction generation during off-peak hours for potentially better pricing
