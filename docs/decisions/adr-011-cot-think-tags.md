# ADR-011: Chain-of-Thought with think Tags

**Status:** Accepted  
**Date:** 2026-02

## Context

The model must reason about which items co-occur in which rows before producing the final JSON output. Should it produce the JSON directly or reason step-by-step first?

## Options Considered

| Format | Pros | Cons |
|--------|------|------|
| Direct JSON (no CoT) | Shorter output, faster inference | Model must "think" implicitly; error-prone |
| Scratchpad / comments | Human-readable reasoning | No native model support |
| **`<think>...</think>` tags** | Qwen-native; separable from JSON | Longer output; must learn tag format |

## Decision

**`<think>` tags** for reasoning, followed by a JSON array.

## Rationale

Qwen2.5-Instruct has native pre-training support for `<think>` tags as part of its instruct tuning. This means the model already understands the semantic role of these tags (internal reasoning before final answer), reducing the learning burden during SFT.

The explicit separation between reasoning and output enables a critical inference optimization: **two-phase generation** ([ADR-016](adr-016-two-phase-inference.md)). By detecting the `</think>` token with a custom `StoppingCriteria`, the system can halt generation after reasoning, then regenerate the JSON at a lower temperature. This decouples the temperature tradeoff between exploratory reasoning and precise structured output.

## Trade-offs

- Longer context usage per example (CoT adds ~500-1500 tokens)
- Model must learn the tag format exactly (open and close tags)
- If the model fails to close the `</think>` tag, the StoppingCriteria cannot trigger

## Source Evidence

- `src/training/training_utils.py:22-34` -- system prompt instructs `<think>` tag usage
- `src/evaluation/inference_utils.py` -- `ThinkStoppingCriteria` implementation
