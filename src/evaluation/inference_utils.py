#!/usr/bin/env python3
"""
Inference utilities for fine-tuned itemset extraction models.

Implements all council-recommended fixes for repetition loops (2026-03-09):
  1. ThinkStoppingCriteria — stops generation at </think> boundary
  2. Dynamic token budget — dataset-specific max_new_tokens
  3. Two-phase generation — <think> at temp=0.3, JSON at temp=0.05
  4. Repetition loop detection — monitors for repeated lines

These are INFERENCE-ONLY fixes — no retraining needed. Can be applied
to any adapter (v2, v3) immediately.
"""

import re
import json
import torch
from typing import Optional, List, Dict, Any

# ═══════════════════════════════════════════════════════════════════════════════
# 1. StoppingCriteria at </think> boundary
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except ImportError:
    # Minimal fallback for environments without transformers
    class StoppingCriteria:
        def __call__(self, input_ids, scores, **kwargs):
            return False

    class StoppingCriteriaList(list):
        pass


class ThinkStoppingCriteria(StoppingCriteria):
    """
    Stop generation when </think> token sequence is produced.

    Council rationale: The model often enters repetition loops inside <think>.
    By stopping at </think>, we can then generate the JSON phase separately
    with different parameters (lower temp, constrained decoding).
    """

    def __init__(self, tokenizer, max_think_tokens: int = 3000):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_think_tokens = max_think_tokens
        # Pre-encode the stop string
        self.stop_ids = tokenizer.encode("</think>", add_special_tokens=False)
        self.stop_len = len(self.stop_ids)
        self._generated_count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._generated_count += 1

        # Hard cap on think tokens
        if self._generated_count >= self.max_think_tokens:
            return True

        # Check if last N tokens match </think>
        if input_ids.shape[1] >= self.stop_len:
            last_tokens = input_ids[0, -self.stop_len:].tolist()
            if last_tokens == self.stop_ids:
                return True

        return False

    def reset(self):
        self._generated_count = 0


class RepetitionDetector(StoppingCriteria):
    """
    Stop generation if the model enters a repetition loop.

    Detects when the same line pattern appears N times consecutively.
    Council rationale: Even with concise CoT, some edge cases may loop.
    """

    def __init__(self, tokenizer, max_repeats: int = 3, check_every: int = 50):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_repeats = max_repeats
        self.check_every = check_every
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._step += 1
        if self._step % self.check_every != 0:
            return False

        # Decode last 500 tokens and check for repeated lines
        text = self.tokenizer.decode(input_ids[0, -500:], skip_special_tokens=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        if len(lines) < self.max_repeats * 2:
            return False

        # Check if last N lines are duplicates of the preceding N lines
        recent = lines[-self.max_repeats:]
        if len(set(recent)) == 1 and len(recent) >= self.max_repeats:
            return True  # Same line repeated N times

        return False

    def reset(self):
        self._step = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dynamic token budget calculator
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_dynamic_budget(n_rows: int, n_cols: int, n_unique_vals: int = 0) -> int:
    """
    Calculate dataset-specific max_new_tokens.

    Council rationale: If a dataset has 10 unique values, a correct answer
    cannot possibly require 4000 tokens. By setting a tight budget, the model
    physically cannot loop indefinitely.

    Returns total token budget (think + JSON).
    """
    if n_unique_vals == 0:
        n_unique_vals = n_cols * 3  # heuristic

    # Estimate output size from input characteristics
    singles_est = n_unique_vals
    pairs_est = min(singles_est * (singles_est - 1) // 4, 50)
    triples_est = min(pairs_est // 3, 20)

    # Concise CoT: ~15 tokens per single, ~20 per pair, ~25 per triple
    cot_tokens = singles_est * 15 + pairs_est * 20 + triples_est * 25 + 80
    # JSON: ~60 tokens per itemset
    json_tokens = (singles_est + pairs_est + triples_est) * 60

    budget = int((cot_tokens + json_tokens) * 1.5)
    return max(512, min(budget, 6000))


def count_unique_values(csv_text: str) -> int:
    """Count unique cell values in CSV text (Row N: val1, val2, ...) format."""
    values = set()
    for line in csv_text.strip().split("\n"):
        if ":" in line:
            items_part = line.split(":", 1)[1].strip()
            for item in items_part.split(","):
                item = item.strip().lower()
                if item:
                    values.add(item)
    return len(values)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Two-phase generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_two_phase(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    think_temperature: float = 0.3,
    json_temperature: float = 0.05,
    think_max_tokens: int = 2000,
    json_max_tokens: int = 1500,
    top_k: int = 50,
    top_p: float = 0.90,
) -> str:
    """
    Two-phase generation: <think> phase + JSON phase.

    Phase 1: Generate reasoning with higher temperature (escape attractors)
             Stop at </think> using StoppingCriteria
    Phase 2: Generate JSON with very low temperature (precision)

    Council rationale (Claude): temp=0.1 makes loops WORSE because the model
    is near-deterministic and locks into the highest-probability attractor.
    temp=0.3 adds enough randomness to escape, with <3% F1 degradation.
    """
    device = input_ids.device

    # ── Phase 1: Think phase ──────────────────────────────────────────────
    think_stopper = ThinkStoppingCriteria(tokenizer, max_think_tokens=think_max_tokens)
    rep_detector = RepetitionDetector(tokenizer, max_repeats=3)

    with torch.no_grad():
        think_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=think_max_tokens,
            temperature=think_temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([think_stopper, rep_detector]),
        )

    think_text = tokenizer.decode(think_output[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Check if </think> was found naturally
    if "</think>" not in think_text:
        # Truncate at last complete line and add </think>
        lines = think_text.split("\n")
        # Find last "meaningful" line (with ✓ or ## or RESULT)
        last_good = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            if any(m in lines[i] for m in ["✓", "✗", "##", "RESULT", "FREQUENT", "SCAN"]):
                last_good = i
                break
        think_text = "\n".join(lines[:last_good + 1]) + "\n</think>\n"
    elif not think_text.endswith("\n"):
        think_text += "\n"

    # ── Phase 2: JSON phase ───────────────────────────────────────────────
    # Build continuation prompt: everything so far + prime with "["
    full_so_far = tokenizer.decode(think_output[0], skip_special_tokens=False)
    if not full_so_far.rstrip().endswith("</think>"):
        full_so_far = full_so_far.rstrip() + "\n</think>\n"

    json_prompt = full_so_far + "["
    json_input_ids = tokenizer(json_prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        json_output = model.generate(
            input_ids=json_input_ids,
            max_new_tokens=json_max_tokens,
            temperature=json_temperature,
            top_k=20,  # Tighter for JSON precision
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    json_text = tokenizer.decode(
        json_output[0][json_input_ids.shape[1]:], skip_special_tokens=True
    )

    # Reconstruct full response
    full_response = think_text
    if not full_response.rstrip().endswith("</think>"):
        full_response = full_response.rstrip() + "\n</think>\n"
    full_response += "[" + json_text

    return full_response


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Single-phase generation with all fixes
# ═══════════════════════════════════════════════════════════════════════════════

def generate_with_guards(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 2048,
    temperature: float = 0.3,
    top_k: int = 50,
    top_p: float = 0.90,
) -> str:
    """
    Single-phase generation with StoppingCriteria guards.

    Simpler than two-phase but still applies:
    - ThinkStoppingCriteria (catches infinite loops in <think>)
    - RepetitionDetector (catches repeated line patterns)
    - Council-recommended temperature=0.3

    Use this if two-phase is too complex for your setup.
    """
    think_stopper = ThinkStoppingCriteria(tokenizer, max_think_tokens=max_new_tokens)
    rep_detector = RepetitionDetector(tokenizer, max_repeats=3)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([think_stopper, rep_detector]),
        )

    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Post-processing: extract JSON from potentially messy output
# ═══════════════════════════════════════════════════════════════════════════════

def extract_and_repair_json(raw_output: str) -> tuple:
    """
    Extract JSON array from model output, handling various failure modes.

    Returns (parsed_items: list, parse_ok: bool, json_text: str)
    """
    has_think = "<think>" in raw_output and "</think>" in raw_output
    json_text = raw_output

    if has_think:
        parts = raw_output.split("</think>", 1)
        json_text = parts[1].strip() if len(parts) > 1 else ""

    # Try direct parse
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, list):
            return parsed, True, json_text
    except json.JSONDecodeError:
        pass

    # Try regex extraction
    m = re.search(r"\[.*\]", json_text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return parsed, True, m.group()
        except json.JSONDecodeError:
            pass

    # Try from full raw output
    m = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                return parsed, True, m.group()
        except json.JSONDecodeError:
            pass

    return [], False, json_text


# ═══════════════════════════════════════════════════════════════════════════════
# 6. V3 inference configuration (council consensus)
# ═══════════════════════════════════════════════════════════════════════════════

V3_INFERENCE_CONFIG = {
    # Two-phase generation (recommended)
    "two_phase": {
        "think_temperature": 0.3,   # Escape attractor states
        "json_temperature": 0.05,   # Precision for structured output
        "think_max_tokens": 2000,
        "json_max_tokens": 1500,
        "top_k": 50,
        "top_p": 0.90,
    },
    # Single-phase fallback
    "single_phase": {
        "temperature": 0.3,
        "max_new_tokens": 2048,
        "top_k": 50,
        "top_p": 0.90,
    },
    # Notes from council
    "council_notes": {
        "why_temp_0.3": "Claude: temp=0.1 makes loops WORSE (near-deterministic, locks into attractor). temp=0.3 adds randomness to escape. F1 degradation <3%.",
        "why_top_k_50": "Prunes long tail of low-probability tokens that trigger loops.",
        "why_two_phase": "Different requirements for reasoning (diversity) vs JSON (precision).",
        "dynamic_budget": "Use calculate_dynamic_budget(n_rows, n_cols) for per-dataset max_new_tokens.",
    },
}
