"""
Script 3: Synthetic instruction pair generation (OSS-Instruct style)
=====================================================================
Prerequisites:
    pip install anthropic tqdm python-dotenv

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python 03_synthetic_instruct.py \
        --input_dirs ./data/stack_v2_csharp ./data/github_csharp \
        --output_dir ./data/synthetic_instruct \
        --model claude-sonnet-4-6

This implements the OSS-Instruct methodology:
  - Takes real C# code snippets as "seeds"
  - Uses a capable LLM to generate a natural language instruction
    that would plausibly produce that code
  - Optionally generates a cleaned/improved version of the code too
  - Filters pairs where the model expressed low confidence
  - Outputs in both raw JSONL and Alpaca-format JSONL for fine-tuning

The resulting dataset teaches the model to respond to natural language
prompts with correct C# code — which is your actual inference-time task.

Strategy for seed selection:
  - Prefers modern C# files
  - Extracts individual methods/classes where possible (not whole files)
    so each training example is a focused, completable unit
  - Skips trivial getters/setters and boilerplate
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Code chunking — extract meaningful units from files
# ---------------------------------------------------------------------------

# Match C# method or class declarations
METHOD_RE = re.compile(
    r"""
    (?:\/\/\/.*\n)*              # optional XML doc comments
    (?:\/\/.*\n)*                # optional line comments
    \s*
    (?:public|private|protected|internal|static|async|override|virtual|abstract|sealed|partial|new)
    (?:\s+(?:public|private|protected|internal|static|async|override|virtual|abstract|sealed|partial|new))*
    \s+
    [\w<>\[\],\s\?]+             # return type
    \s+\w+\s*                    # method name
    (?:<[^>]+>)?\s*              # optional generic params
    \(                           # open paren
    """,
    re.VERBOSE,
)

CLASS_RE = re.compile(
    r"""
    (?:\/\/\/.*\n)*
    \s*(?:public|internal|private|protected)?\s*
    (?:abstract|sealed|static|partial)?\s*
    (?:class|interface|record|struct|enum)\s+
    \w+
    """,
    re.VERBOSE,
)

MIN_CHUNK_CHARS = 100
MAX_CHUNK_CHARS = 4000


def extract_chunks(content: str) -> list[str]:
    """
    Extract individual methods and classes from a C# file.
    Falls back to the whole file if parsing fails.
    """
    chunks = []
    lines = content.splitlines(keepends=True)

    # Simple brace-balanced extraction
    def extract_from_match(start_line: int) -> Optional[str]:
        depth = 0
        started = False
        extracted_lines = []
        for line in lines[start_line:]:
            extracted_lines.append(line)
            depth += line.count("{") - line.count("}")
            if "{" in line:
                started = True
            if started and depth <= 0:
                break
            if len("".join(extracted_lines)) > MAX_CHUNK_CHARS:
                break
        result = "".join(extracted_lines).strip()
        if len(result) >= MIN_CHUNK_CHARS:
            return result
        return None

    for i, line in enumerate(lines):
        if METHOD_RE.match(line) or CLASS_RE.match(line):
            chunk = extract_from_match(i)
            if chunk:
                chunks.append(chunk)

    if not chunks:
        # Fall back: return the whole file if it's not too large
        if MIN_CHUNK_CHARS <= len(content) <= MAX_CHUNK_CHARS:
            chunks.append(content.strip())

    return chunks


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INSTRUCT_GENERATION_PROMPT = """\
You are an expert C# developer and technical writer.

Below is a C# code snippet. Your task is to write a clear, specific natural language instruction that a developer would write to request exactly this code (or code very similar to it). The instruction should:

- Be written as a task request (e.g. "Write a C# method that...", "Implement a class that...", "Create a function that...")
- Be specific enough that a capable model would generate functionally equivalent code
- Mention the key inputs, outputs, and behaviors visible in the code
- NOT reference variable names unless they're meaningful domain concepts
- Be 1-3 sentences long

After the instruction, on a new line write "---" and then provide a clean, corrected version of the code. Fix any minor issues you see (missing null checks, non-idiomatic patterns, etc.), but preserve the overall structure. If the code is already excellent, reproduce it as-is.

CODE:
```csharp
{code}
```

Respond in this exact format:
INSTRUCTION: <your instruction here>
---
```csharp
<clean code here>
```
"""

FILTER_CONFIDENCE_RE = re.compile(
    r"I('m| am) not sure|unclear|cannot determine|not enough context|too short|trivial",
    re.IGNORECASE,
)


def parse_response(response_text: str) -> Optional[tuple[str, str]]:
    """Parse the model's response into (instruction, clean_code)."""
    if "INSTRUCTION:" not in response_text:
        return None
    if FILTER_CONFIDENCE_RE.search(response_text):
        return None

    try:
        instruction_part, code_part = response_text.split("---", 1)
        instruction = instruction_part.replace("INSTRUCTION:", "").strip()

        # Extract code from fences
        code_match = re.search(r"```(?:csharp)?\n(.*?)```", code_part, re.DOTALL)
        if not code_match:
            return None
        clean_code = code_match.group(1).strip()

        if len(instruction) < 20 or len(clean_code) < MIN_CHUNK_CHARS:
            return None

        return instruction, clean_code
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Alpaca format output
# ---------------------------------------------------------------------------

def to_alpaca(instruction: str, code: str) -> dict:
    return {
        "instruction": instruction,
        "input": "",
        "output": f"```csharp\n{code}\n```",
    }


def to_chatml(instruction: str, code: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": f"```csharp\n{code}\n```"},
        ]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all_records(input_dirs: list[str]) -> list[dict]:
    records = []
    for d in input_dirs:
        for jsonl_path in Path(d).rglob("*.jsonl"):
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def prioritize_records(records: list[dict]) -> list[dict]:
    """
    Sort records so we process the highest-quality seeds first:
    modern C# > github-scraped > stack-v2, then by stars desc.
    """
    def score(r):
        is_modern = int(r.get("modern_csharp", False))
        is_github = int(r.get("source") == "github-scraped")
        stars = r.get("stars", 0) or 0
        return (is_modern, is_github, stars)

    return sorted(records, key=score, reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+",
                        default=["./data/stack_v2_csharp", "./data/github_csharp"])
    parser.add_argument("--output_dir", default="./data/synthetic_instruct")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max_pairs", type=int, default=50_000,
                        help="Target number of instruction pairs to generate")
    parser.add_argument("--max_chunks_per_file", type=int, default=3,
                        help="Max chunks to extract from a single file")
    parser.add_argument("--format", choices=["alpaca", "chatml", "both"],
                        default="both")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed files based on output count")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    client = anthropic.Anthropic(api_key=api_key)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpaca_path = out_dir / "csharp_instruct_alpaca.jsonl"
    chatml_path = out_dir / "csharp_instruct_chatml.jsonl"
    raw_path = out_dir / "csharp_instruct_raw.jsonl"

    # Resume support: count existing pairs
    existing_pairs = 0
    if args.resume and raw_path.exists():
        with open(raw_path) as f:
            existing_pairs = sum(1 for _ in f)
        print(f"Resuming from {existing_pairs} existing pairs.")

    print("Loading source records...")
    records = load_all_records(args.input_dirs)
    print(f"  Loaded {len(records):,} source files")

    records = prioritize_records(records)

    # Open output files in append mode for resume support
    alpaca_f = open(alpaca_path, "a", encoding="utf-8")
    chatml_f = open(chatml_path, "a", encoding="utf-8")
    raw_f = open(raw_path, "a", encoding="utf-8")

    pairs_generated = existing_pairs
    errors = 0
    skipped = 0

    progress = tqdm(
        total=args.max_pairs,
        initial=existing_pairs,
        desc="Generating pairs",
        unit=" pairs",
    )

    for record in records:
        if pairs_generated >= args.max_pairs:
            break

        content = record.get("content", "")
        if not content:
            continue

        chunks = extract_chunks(content)[:args.max_chunks_per_file]
        if not chunks:
            continue

        for chunk in chunks:
            if pairs_generated >= args.max_pairs:
                break

            prompt = INSTRUCT_GENERATION_PROMPT.format(code=chunk)

            try:
                response = client.messages.create(
                    model=args.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.content[0].text
            except anthropic.RateLimitError:
                time.sleep(60)
                continue
            except anthropic.APIError as e:
                errors += 1
                if errors > 100:
                    print("Too many API errors, stopping.")
                    break
                continue

            parsed = parse_response(response_text)
            if not parsed:
                skipped += 1
                continue

            instruction, clean_code = parsed
            pairs_generated += 1

            # Write raw
            raw_record = {
                "instruction": instruction,
                "code": clean_code,
                "source_path": record.get("path", ""),
                "source_repo": record.get("repo_name", ""),
                "source": record.get("source", ""),
                "modern_csharp": record.get("modern_csharp", False),
            }
            raw_f.write(json.dumps(raw_record, ensure_ascii=False) + "\n")
            raw_f.flush()

            # Write formatted
            if args.format in ("alpaca", "both"):
                alpaca_f.write(json.dumps(to_alpaca(instruction, clean_code), ensure_ascii=False) + "\n")
                alpaca_f.flush()

            if args.format in ("chatml", "both"):
                chatml_f.write(json.dumps(to_chatml(instruction, clean_code), ensure_ascii=False) + "\n")
                chatml_f.flush()

            progress.update(1)
            progress.set_postfix(skipped=skipped, errors=errors)

            # Small delay to avoid hammering the API
            time.sleep(0.1)

    progress.close()
    alpaca_f.close()
    chatml_f.close()
    raw_f.close()

    print(f"\nDone.")
    print(f"  Pairs generated : {pairs_generated:,}")
    print(f"  Skipped (low quality): {skipped:,}")
    print(f"  API errors      : {errors:,}")
    print(f"  Alpaca format   : {alpaca_path}")
    print(f"  ChatML format   : {chatml_path}")
    print(f"  Raw             : {raw_path}")


if __name__ == "__main__":
    main()

