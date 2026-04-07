#!/usr/bin/env python3
"""
Filter SFT data to only include examples that fit within max_seq_length 
when tokenized with the actual Qwen tokenizer.

Usage:
    python scripts/filter_sft_by_tokens.py \
        --input data/sft_cot_v3.json \
        --output data/sft_cot_v3_filtered.json \
        --max-tokens 4096
"""

import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Filter SFT data by actual token count")
    parser.add_argument("--input", default="data/sft_cot_v3.json", help="Input JSON path")
    parser.add_argument("--output", default="data/sft_cot_v3.json", help="Output JSON path (overwrites input if same)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens after chat template formatting")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer name")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading data: {args.input}")
    with open(args.input) as f:
        data = json.load(f)
    print(f"  Total examples: {len(data)}")

    kept = []
    dropped = 0
    lengths = []
    for ex in data:
        text = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
        if n_tokens <= args.max_tokens:
            # Update metadata with actual token count
            if "metadata" in ex:
                ex["metadata"]["actual_tokens"] = n_tokens
            kept.append(ex)
            lengths.append(n_tokens)
        else:
            dropped += 1

    print(f"\n✅ Kept: {len(kept)} examples")
    print(f"❌ Dropped: {dropped} examples (exceeded {args.max_tokens} tokens)")
    if lengths:
        print(f"📊 Token stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved to: {args.output}")


if __name__ == "__main__":
    main()
