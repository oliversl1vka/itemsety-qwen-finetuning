"""LLM Council session: v3 Training Plan — Sequential Reliable Version.

Queries each model one-by-one (not parallel) and saves each response
immediately so no work is lost on connection failures.

Usage:
    python scripts/council_v3_sequential.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "docs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "council_v3_plan_2026-03-09.json"

COUNCIL_MODELS = [
    "google/gemini-3-flash-preview",   # reliable, fast
    "deepseek/deepseek-v3.2",          # reliable, strong reasoning
    "x-ai/grok-4.1-fast",             # reliable, strong agentic
    "anthropic/claude-sonnet-4.6",     # may drop connections, but worth retry
]

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _load_api_key() -> str:
    """Load OpenRouter key from env or openrouter.env."""
    import os
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = PROJECT_ROOT / "openrouter.env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == "OPENROUTER_API_KEY":
                return v.strip().strip("\"'")
    raise ValueError("No OPENROUTER_API_KEY found")


def query_model_sync(model: str, messages: list, api_key: str, max_retries: int = 3) -> str | None:
    """Query a single model synchronously with retries."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/oliversl1vka/itemsety-qwen-finetuning",
        "X-Title": "Itemsety Council",
    }
    payload = {"model": model, "messages": messages}
    model_short = model.split("/")[-1]

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            with httpx.Client(timeout=600.0) as client:
                resp = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"].get("content", "")
                elapsed = time.time() - t0
                print(f"  ✅ {model_short} responded ({elapsed:.1f}s, {len(content)} chars)")
                return content
        except Exception as exc:
            elapsed = time.time() - t0
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"  ⚠️  {model_short} attempt {attempt}/{max_retries} failed after {elapsed:.1f}s: {exc}")
                print(f"      Waiting {wait}s before retry…")
                time.sleep(wait)
            else:
                print(f"  ❌ {model_short} FAILED all {max_retries} attempts: {exc}")
                return None


def build_query() -> str:
    """Build comprehensive query (same as before)."""
    query = """\
You are an expert panel in LLM fine-tuning, LoRA/QLoRA training, preference optimization (DPO/GRPO),
structured output generation, and frequent itemset mining. You are advising on a critical v3 training
plan for a Qwen2.5-7B model fine-tuned to extract frequent itemsets from CSV datasets.

══════════════════════════════════════════════════════════════════════
PROJECT GOAL
══════════════════════════════════════════════════════════════════════
Fine-tune Qwen2.5-7B-Instruct to extract frequent itemsets from CSV transactional data.
- Input: CSV dataset (5-25 rows, 3-12 columns) + min_support threshold
- Output: JSON array of {"itemset": ["col:val", ...], "count": N, "evidence_rows": ["Row 1", ...]}
- Ground truth: Apriori algorithm output (deterministic, always correct)
- Target: F1 ≥ 0.80, JSON parse rate ≥ 0.90, exact match ≥ 0.50

The model must:
1. Produce <think> chain-of-thought reasoning (singles → pairs → triples)
2. Output items in EXACT col:val format (e.g., "default:0", "carloan:1")
3. Output valid JSON array after </think>
4. Correctly count support and list evidence rows as "Row N"

══════════════════════════════════════════════════════════════════════
TRAINING HISTORY (v1 → v2 → diagnostics)
══════════════════════════════════════════════════════════════════════

--- v1 Training (2026-03-01) ---
Config: r=64, alpha=16 (ratio 0.25), SFT 2 epochs lr=2e-4, DPO 2 epochs lr=5e-5 beta=0.1,
        GRPO 200 steps lr=5e-6, merged_4bit_forced
Result: F1 = 0.4%, total failure. GRPO failed (collapsed to empty outputs).
Root cause: merged_4bit_forced destroys LoRA deltas during quantization.
Previous council verdict: "Unanimous — merged_4bit_forced is the root cause."

--- v2 Training (2026-03-07) ---
Config: SFT-only, 5 epochs, lr=2e-4, r=64, alpha=16, max_seq_length=4096
        348 SFT-CoT examples (with <think> reasoning)
        Saved via model.save_pretrained (adapter-only)
Result: F1 = 0.4% when merged to 4-bit for inference (same merge bug)
        F1 = 78.6% when loaded as adapter on base model (for 10×3 dataset)
Diagnosis: The adapter WORKS but only on small/simple datasets. Larger datasets
           trigger repetition loops that hit the 8192 token limit.

--- Adapter Evaluation Detailed Results (2026-03-08) ---
Loaded as: base model (Qwen2.5-7B-Instruct 4-bit) + LoRA adapter overlay
Inference: max_new_tokens=8192, temperature=0.1, top_p=0.95

TWO separate evaluation runs were performed:

══════════════════════════════════════════════════════════════════════
EVAL 1: RAW CAPTURE (no repetition penalties)
══════════════════════════════════════════════════════════════════════
Config: temperature=0.1, top_p=0.95, max_new_tokens=8192, NO repetition penalties
15 datasets tested, results:

✅ SUCCESSES (2/15):
- ds_0280 (10×3): F1=78.6%, P=64.7%, R=100%, parse=OK
  Small dataset, only 3 columns. Model handled it perfectly.
- ds_0126 (7×12): F1=33.8%, P=22.8%, R=65.2%, parse=OK
  Medium dataset. Got some right but hallucinated extras.

❌ FAILURES (13/15) — ALL hit 8192 token limit:
- ds_0013 (18×12): Output = 8192 tokens, repetitive loop after initial <think>
- ds_0045 (6×12): 8192 tokens, loop after singles section
- ds_0053 (9×8): 8192 tokens, loop in pairs section
- ds_0058 (10×12): 8192 tokens, loop in pairs
- ds_0072 (6×9): 8192 tokens, loop in pairs
- ds_0115 (5×8): 8192 tokens, loop after singles
- ds_0141 (9×12): 8192 tokens, loop
- ds_0303 (21×3): 8192 tokens, loop (despite only 3 cols!)
- ds_0328 (13×5): 8192 tokens, loop
- ds_0347 (10×4): 8192 tokens, loop
- ds_0378 (25×4): 8192 tokens, loop
- ds_0380 (12×4): 8192 tokens, loop
- ds_0457 (21×9): 8192 tokens, loop

Pattern: Model starts <think> correctly, enumerates singles, then enters an
infinite loop repeating "- col:val: N rows → [Row ...] ✓" patterns.
The loop repeats the SAME items with slight variations endlessly.
Even 3-column datasets with many rows (21×3) trigger the loop.

Overall: Parse rate 13.3%, Think rate 100%, Mean F1 7.5%

══════════════════════════════════════════════════════════════════════
EVAL 2: REPETITION PENALTY (attempted fix)
══════════════════════════════════════════════════════════════════════
Config: temperature=0.1, top_p=0.95, max_new_tokens=8192,
        repetition_penalty=1.3, no_repeat_ngram_size=3
30 datasets tested, results:

✅ SUCCESSES: 0/30 (ZERO!)
❌ ALL 30 FAILED — different failure mode than raw capture:
- Think rate: 93.3% (model still produces <think>)
- Parse rate: 0% (model NEVER produces valid JSON!)

Failure pattern: no_repeat_ngram_size=3 DESTROYS structured output.
The model cannot repeat "Row 1", "Row 2", etc. because "Row" + number
is considered a repeated n-gram. Evidence rows become garbled.
JSON array syntax (repeated brackets, colons, quotes) also breaks.

Example broken output:
  "evidence_rows": ["Row 1", " 2, 3," Row 4"... (garbage)
  Can't output proper {"itemset": [...], "count": N, ...} because
  the structure requires repeating patterns.

Mean F1: 0.0%, Parse rate: 0.0%

Conclusion: repetition_penalty and no_repeat_ngram_size CANNOT be used
for structured output tasks. They destroy the very patterns needed for JSON.

══════════════════════════════════════════════════════════════════════
ROOT CAUSE ANALYSIS
══════════════════════════════════════════════════════════════════════
The CORE PROBLEM is the repetition loop in the <think> section:
1. Model correctly starts CoT: "## Singles (min_support=3)"
2. Enumerates first few items correctly
3. Then enters a loop repeating similar patterns:
   "- col:val: N rows → [Row 1, Row 3, ...] ✓"
   "- col:val: N rows → [Row 1, Row 3, ...] ✓"  (repeated!)
4. Never reaches </think> or the JSON output section
5. Hits max_new_tokens=8192 and truncates

This happens because:
- SFT training data has 348 examples with avg 1547 tokens
- But the <think> CoT is repetitive by nature (similar format for each item)
- The model over-learns the pattern and can't stop generating it
- Larger datasets = more items = more pattern repetitions = higher loop probability

WHAT WE KNOW WORKS:
- The adapter itself is correctly trained (78.6% F1 on compatible datasets)
- The <think> reasoning structure is learned correctly
- The model knows the col:val format
- Small datasets (≤10 rows, ≤3 columns) work well

WHAT DOESN'T WORK:
- Any dataset requiring >~2000 tokens of CoT reasoning → repetition loop
- Merged models (merged_4bit_forced) → complete failure
- Repetition penalties → destroy JSON structure
- no_repeat_ngram_size → completely incompatible with structured output

══════════════════════════════════════════════════════════════════════
QUESTIONS FOR THE COUNCIL (answer ALL 9)
══════════════════════════════════════════════════════════════════════

1. ROOT CAUSE — REPETITION LOOP: Why does the model enter repetition loops
   specifically in the <think> section? Is this a known failure mode of SFT
   with chain-of-thought? Is it caused by:
   a) Insufficient training data diversity (348 examples)?
   b) Too many epochs (5) causing memorization of patterns?
   c) The CoT format itself being too repetitive?
   d) LoRA rank too high (r=64) for this task?
   e) Something else entirely?

2. FIXING REPETITION WITHOUT PENALTIES: How do we prevent the repetition
   loop WITHOUT using repetition_penalty or no_repeat_ngram_size (which
   destroy structured output)? Specific inference-time fixes:
   - Different sampling parameters?
   - Different max_new_tokens strategy?
   - Post-processing truncation at </think>?
   - Constrained decoding / grammar-guided generation?

3. TRAINING-LEVEL FIXES: What changes to training data or process would
   reduce repetition tendency at its source?
   - Add "stop repeating" examples in training data?
   - Use shorter CoT format (remove detailed evidence)?
   - Limit CoT to key findings only?
   - Use curriculum learning (small → large datasets)?
   - Add negative examples showing what NOT to do?

4. DPO STRATEGY: Should we add DPO training to v3? We have 606 real LLM
   failure pairs. Could DPO help by:
   - Teaching the model to avoid long, repetitive outputs (rejected)?
   - Preferring concise, complete outputs (chosen)?
   - Specific DPO pairs: chosen=correct+concise, rejected=repetitive+truncated?

5. GRPO ALTERNATIVE: GRPO failed in v1 (collapsed outputs). Should we:
   - Skip GRPO entirely in v3?
   - Try GRPO after fixing the repetition issue?
   - Use a simpler reward (just JSON parse + F1)?
   - What reward signal would penalize repetition loops?

6. HYPERPARAMETER CHANGES for v3:
   - LoRA rank: keep r=64 or reduce to r=32 or r=16?
   - Alpha: keep 16 or change ratio?
   - Learning rate: keep 2e-4 or adjust?
   - Epochs: 5 was too many? Try 1-3?
   - Sequence length: reduce from 4096 to help focus?

7. DATA AUGMENTATION: Should we:
   - Generate more training examples (348 → 1000+)?
   - Add examples with deliberately SHORT CoT (no detailed evidence)?
   - Include examples where model must handle larger datasets?
   - Add examples showing proper termination after CoT?
   - Mix complexity levels (simple 5×3 to complex 25×12)?

8. SAVE & DEPLOY METHOD: Given that merged_4bit_forced fails:
   - Best method to save adapter for inference on Jupyter server?
   - How to load adapter + base model for production inference?
   - Is there a safe merge method that preserves LoRA quality?
   - Should we use GGUF export instead?

9. OVERALL v3 PLAN: Provide a concrete, step-by-step training plan with:
   - Phases (which to include, which to skip)
   - Data preparation steps
   - Expected training time
   - Success criteria / checkpoints
   - Fallback if it doesn't work

Please be specific with numbers, not vague ("increase" → "set to 128"). Prioritize by impact.
"""
    return query


def main():
    print("=" * 70)
    print("🏛️  LLM COUNCIL SESSION: v3 Training Plan (Sequential)")
    print("=" * 70)
    print(f"Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Models: {len(COUNCIL_MODELS)}")
    print()

    api_key = _load_api_key()
    query = build_query()
    print(f"Query: {len(query):,} chars\n")

    messages = [{"role": "user", "content": query}]

    # Intermediate results file
    partial_file = OUTPUT_DIR / "council_v3_partial.json"
    results = {"stage1": [], "timestamp": datetime.now(UTC).isoformat()}

    # Stage 1: Query each model sequentially
    print("━" * 70)
    print("STAGE 1: Individual Opinions (sequential)")
    print("━" * 70)

    for i, model in enumerate(COUNCIL_MODELS, 1):
        model_short = model.split("/")[-1]
        print(f"\n[{i}/{len(COUNCIL_MODELS)}] Querying {model_short}…")

        content = query_model_sync(model, messages, api_key, max_retries=3)
        if content:
            results["stage1"].append({"model": model, "response": content})
        else:
            results["stage1"].append({"model": model, "response": None, "failed": True})

        # Save after EACH model response (crash-safe)
        with open(partial_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved progress ({len([r for r in results['stage1'] if r.get('response')])} responses so far)")

    successful = [r for r in results["stage1"] if r.get("response") and not r.get("failed")]
    print(f"\n{'━' * 70}")
    print(f"Stage 1 complete: {len(successful)}/{len(COUNCIL_MODELS)} models responded")
    print(f"{'━' * 70}")

    if len(successful) < 2:
        print("❌ Fewer than 2 models responded. Aborting.")
        sys.exit(1)

    # Stage 2: Self-synthesis (no chairman — we do it ourselves)
    # Just save all results and print summaries
    results["metadata"] = {
        "council_models": COUNCIL_MODELS,
        "successful_models": [r["model"] for r in successful],
        "failed_models": [r["model"] for r in results["stage1"] if r.get("failed")],
        "query_length": len(query),
    }
    results["query"] = query

    # Save final
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print each response summary
    print(f"\n{'=' * 70}")
    print("📋 COUNCIL RESPONSES")
    print("=" * 70)
    for resp in successful:
        model_short = resp["model"].split("/")[-1]
        text = resp["response"]
        # Print first 4000 chars of each
        print(f"\n{'─' * 60}")
        print(f"🤖 {model_short} ({len(text)} chars)")
        print(f"{'─' * 60}")
        print(text[:4000])
        if len(text) > 4000:
            print(f"\n… [{len(text) - 4000} more chars in JSON file]")

    print(f"\n💾 Full results saved to: {OUTPUT_FILE}")
    print(f"💾 Partial results at: {partial_file}")
    print(f"\n✅ Council session complete!")


if __name__ == "__main__":
    main()
