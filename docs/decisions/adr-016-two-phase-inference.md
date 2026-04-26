# ADR-016: Two-Phase Inference

**Status:** Accepted  
**Date:** 2026-03

## Context

During inference, the model generates both reasoning (`<think>` tags) and structured output (JSON). These have conflicting temperature requirements: reasoning benefits from exploration (higher temperature), while JSON output requires precision (near-zero temperature).

## Options Considered

| Strategy | Reasoning Quality | JSON Quality | Complexity |
|----------|------------------|-------------|------------|
| Single-pass, low temp | Repetition loops | Good | Simple |
| Single-pass, high temp | Good | Noisy/broken JSON | Simple |
| **Two-phase generation** | Good (temp=0.3) | Good (temp=0.05) | Medium |
| Beam search | Moderate | Moderate | High |
| Constrained decoding | N/A | Perfect format | Complex |

## Decision

**Two-phase generation:**

1. **Phase 1 (Think):** Generate at temperature 0.3 until `ThinkStoppingCriteria` detects `</think>`
2. **Phase 2 (JSON):** Regenerate the output portion at temperature 0.05

## Rationale

Single-pass generation forces a single temperature for both reasoning and JSON:

- **Low temperature** (0.0--0.1) causes the reasoning phase to enter repetition loops -- the model deterministically picks the same next token, repeating the same reasoning step indefinitely
- **High temperature** (0.5+) produces good reasoning but noisy JSON (malformed brackets, hallucinated fields)

Two-phase generation decouples these tradeoffs by using different temperatures for each stage. The `ThinkStoppingCriteria` detects the `</think>` token sequence and halts Phase 1, then Phase 2 regenerates only the JSON with near-greedy decoding.

Additionally, a `RepetitionDetector` monitors Phase 1 for infinite reasoning loops (repeated line patterns) and triggers early termination if detected.

## Trade-offs

- 2x generation overhead (two forward passes)
- Requires custom `StoppingCriteria` implementation
- If the model fails to produce `</think>`, Phase 1 runs until the hard token limit (6000)

## Source Evidence

- `src/evaluation/inference_utils.py:159-245` -- two-phase implementation
- `src/evaluation/inference_utils.py:36-80` -- `ThinkStoppingCriteria`
- `src/evaluation/inference_utils.py:82-110` -- `RepetitionDetector`
