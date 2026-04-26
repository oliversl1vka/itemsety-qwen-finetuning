# DPO Training (Phase 2)

Direct Preference Optimization refines the SFT model by training it to prefer correct outputs over real LLM failure outputs. Unlike PPO-based RLHF, DPO does not require a separate reward model -- it directly optimizes the policy using preference pairs.

## What is DPO

DPO minimizes the following loss:

\[
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
\]

Where:

- \(\pi_\theta\) is the model being trained (initialized from SFT checkpoint)
- \(\pi_{\text{ref}}\) is the frozen reference model (the SFT checkpoint)
- \(y_w\) is the preferred (chosen) completion -- Apriori ground truth with CoT
- \(y_l\) is the dispreferred (rejected) completion -- a real LLM failure
- \(\beta\) controls how far the trained policy can deviate from the reference

## Why DPO Over PPO

See [ADR-008](../decisions/adr-008-dpo-not-ppo.md) for the full analysis. Key reasons:

- **No reward model needed**: PPO requires training a separate reward model, which is expensive and training-unstable. DPO directly uses preference pairs.
- **Naturally paired data**: our data structure (Apriori correct output vs. LLM failure output for the same dataset) is exactly what DPO is designed for.
- **Stable training**: DPO loss is a standard cross-entropy variant, well-understood and debuggable.

## Data Composition

606 preference pairs sourced from real LLM failures in `runs.db`.

| Component | Description |
|-----------|-------------|
| **Chosen** | Apriori ground truth output formatted with v3 CoT reasoning |
| **Rejected** | Real LLM failure from one of 4 models |
| **Split** | 546 train / 60 validation (seed=42, 10%) |

**Source models for rejected outputs:**

- GPT-4.1-mini
- GPT-4.1-nano
- GPT-4o
- o4-mini

**Max 3 rejected outputs per dataset** to prevent any single dataset from dominating the training distribution.

## Error Distribution

The real LLM failures have a distinctive error profile:

| Error Type | Frequency | Description |
|------------|-----------|-------------|
| `item_missing_in_row` | 99.5% | Model cites an item as appearing in a row where it does not exist |
| Other | 0.5% | Count mismatch, JSON malformation, wrong row format |

This extreme concentration on a single failure mode (`item_missing_in_row`) is why real failures are more valuable than synthetic corruption. Random perturbations would generate a uniform distribution of error types that doesn't match what models actually get wrong. See [ADR-009](../decisions/adr-009-real-failures-dpo.md).

## Why Real Failures, Not Synthetic

Synthetic corruption strategies (randomly flipping items, deleting rows, shuffling counts) produce errors that LLMs don't actually make. The dominant real failure mode -- confidently citing evidence that doesn't exist in the CSV -- is qualitatively different from random noise. Training on real failures teaches the model to recognize and avoid its actual behavioral patterns, not arbitrary corruptions.

Source: `src/training/export_real_dpo_data.py:11-14`

## Hyperparameters

From `notebooks/training_3phase_2026-03-09_v3.ipynb` Cell 2:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 5e-5 | Lower than SFT -- DPO loss is more sensitive |
| Beta | 0.1 | KL penalty; higher = more conservative (stays closer to SFT) |
| Batch size | 1 (per device) | DPO processes chosen+rejected pairs, doubling effective memory |
| Gradient accumulation | 4 | Effective batch = 4 |
| Epochs | 1 | Reduced from 2 in v2 -- DPO overfits quickly on 606 pairs |
| Loss type | Standard DPO | TRL DPOTrainer default |
| Reference model | SFT checkpoint (frozen) | Standard DPO setup |

See [ADR-014](../decisions/adr-014-dpo-hyperparams.md).

## Results and Honest Assessment

DPO did **not** improve aggregate F1:

| Model | Precision | Recall | F1 |
|-------|-----------|--------|----|
| SFT only | 13.4% | 19.2% | 12.6% |
| SFT + DPO | 11.4% | 15.7% | 11.8% |

This is an honest negative result. Analysis of per-dataset performance shows:

- DPO improved on some datasets (e.g., eval_002: 81% to 90% F1, eval_030: 70% to 80% F1)
- DPO regressed on others (e.g., eval_010: 25% to 0% F1, eval_017: 7% to 0% F1)
- The model became slightly more conservative (lower recall) without proportionally improving precision

**Possible explanations:**

1. **Small DPO dataset** (606 pairs) may be insufficient to shift the policy meaningfully
2. **One epoch** limits the optimization trajectory
3. **The SFT model's dominant failure mode** (inability to produce valid JSON on large datasets) is not addressable by DPO -- it requires more SFT data covering larger datasets
4. **DPO may have helped with specific error patterns** (like the col:val format leakage seen in SFT outputs) while hurting overall recall

**What DPO did achieve:** The zero hallucination rate was maintained (0.0% for both SFT and DPO), confirming that preference optimization preserves the grounding behavior learned during SFT.
