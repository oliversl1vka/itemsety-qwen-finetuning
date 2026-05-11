# Evaluation

## Evaluation Philosophy

The same Apriori algorithm used to generate training data serves as the evaluation oracle. This ensures consistent measurement: a predicted itemset is correct if and only if Apriori produces the same itemset for the same dataset with the same parameters.

## Evaluation Set

**30 held-out datasets** from `data/eval_datasets_v2/`, versioned and never changed between model comparisons.

- Size range: 5--15 rows, 3--15 columns
- Explicitly excluded from training data by filename matching against `sft_cot_v3.json` and `dpo_real_v2.json`
- Zero hash overlap with training data (verified)
- Published to [OliverSlivka/itemset-eval-v2](https://huggingface.co/datasets/OliverSlivka/itemset-eval-v2)

See [ADR-024](../decisions/adr-024-fixed-eval-set.md) for why a fixed evaluation set matters.

## Metrics

Seven metrics capture different aspects of model quality:

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Precision** | TP / (TP + FP) | Are predicted itemsets correct? |
| **Recall** | TP / (TP + FN) | Are all true itemsets found? |
| **F1** | 2 * P * R / (P + R) | Harmonic mean -- primary metric |
| **Exact Match** | F1 == 1.0 | Perfect correctness (binary per dataset) |
| **Count Accuracy** | correct_counts / matched | Support count within +/-1 tolerance |
| **Hallucination Rate** | FP items not in CSV / all predicted | Primary failure mode tracking |
| **JSON Parse Rate** | parseable outputs / total | Format adherence |

**Canonical form for comparison:** `frozenset(str(x).strip().lower() for x in itemset)` -- items are lowercased, trimmed, and compared as frozen sets.

Source: `src/evaluation/eval_finetuned_model.py:417-500`

## Inference Protocol

!!! warning "Protocol difference between models"
    The fine-tuned models (SFT, DPO) and the commercial GPT baselines use **different prompts and inference protocols**. This is an important methodological caveat caused by the chronological development of the project: the commercial baseline pipeline used the older API extraction prompt, while the fine-tuned model was later trained and evaluated with the compact CoT prompt.

### Fine-tuned models (SFT, DPO): Two-Phase CoT

The fine-tuned models use the inference protocol they were trained for:

1. **System prompt:** compact CoT training/evaluation prompt from `src/training/training_utils.py`
2. **Phase 1 (Think):** Generate reasoning in `<think>` tags at temperature 0.3
3. **Stop:** `ThinkStoppingCriteria` halts generation at the `</think>` token
4. **Phase 2 (JSON):** Regenerate the structured JSON output at temperature 0.05
5. **Safety:** `RepetitionDetector` catches infinite reasoning loops; hard cap at 6000 tokens

Source: `src/evaluation/inference_utils.py:159-245`

### Commercial GPT baselines: Pipeline Extraction

The commercial GPT baselines use the older, simpler pipeline protocol:

1. **Single pass:** Send CSV with extraction prompt, receive JSON directly
2. **System prompt:** `extractor_system_prompt.md` (legacy API baseline prompt, no CoT instruction)
3. **Temperature:** 0.0 (deterministic)

| Aspect | CoT Two-Phase (Fine-tuned) | Pipeline (Baselines) |
|--------|---------------------------|---------------------|
| Prompt | Compact CoT training/evaluation prompt | Legacy API baseline extraction prompt |
| Generation | Two separate phases | Single pass |
| Token budget | Dynamic per dataset | Fixed (API default) |
| Stopping | Custom StoppingCriteria | API default |

**Implication:** The fine-tuned models are evaluated with the CoT protocol they were trained for, while the commercial baseline numbers come from the earlier single-pass protocol. This difference should be treated as a methodological caveat, not as evidence that the same system prompt was used in every experiment. A stricter like-for-like comparison would run GPT-4.1-mini through the same two-phase protocol -- recommended as future work.

The Base Qwen row in the aggregate table below is an unfine-tuned local model reference, not a commercial API baseline call using `extractor_system_prompt.md`.

## Results

### Aggregate Performance

The aggregate table reports archived local primary_v3 values from the April 2026 evaluation run. The published Hugging Face SFT adapter was later re-downloaded and verified on the same 30-dataset primary_v3 profile with F1=13.07%, which is within the documented tolerance of the archived SFT value (12.64%). GPT-4.1-mini's 49.1% F1 here belongs to the same 30 held-out evaluation pool; it is distinct from the earlier F1=0.33 measured on the 500-dataset commercial-baseline pool.

| Model | P | R | F1 | Exact Match | Hallucination | JSON Parse |
|-------|---|---|----|-------------|---------------|------------|
| Base Qwen2.5-7B | 6.7% | 0.5% | 1.0% | 0.0% | 6.7% | 20% (6/30) |
| + SFT (Phase 1) | 13.4% | 19.2% | **12.6%** | 0.0% | **0.0%** | 27% (8/30) |
| + SFT+DPO (Final) | 11.4% | 15.7% | 11.8% | 0.0% | **0.0%** | 20% (6/30) |
| GPT-4.1-mini | **89.6%** | **38.2%** | **49.1%** | 3.3% | 3.3% | 100% (30/30) |

### Per-Dataset Performance (GPT-4.1-mini)

| Dataset | Rows x Cols | P | R | F1 | Apriori GT |
|---------|-------------|---|---|----|------------|
| eval_001_15x9 | 15 x 9 | 96% | 36% | 53% | 151 |
| eval_002_8x4 | 8 x 4 | 100% | 43% | 60% | 21 |
| eval_003_14x8 | 14 x 8 | 89% | 50% | 64% | 32 |
| eval_007_10x8 | 10 x 8 | 76% | 54% | 63% | 24 |
| eval_020_11x8 | 11 x 8 | 88% | 64% | 74% | 22 |
| eval_024_14x4 | 14 x 4 | 100% | 68% | 81% | 28 |
| eval_027_7x4 | 7 x 4 | 100% | 100% | 100% | 3 |
| eval_028_6x5 | 6 x 5 | 100% | 86% | 92% | 7 |
| eval_029_6x4 | 6 x 4 | 75% | 100% | 86% | 3 |

GPT-4.1-mini shows a clear pattern: high precision / lower recall. It reports itemsets it is confident about and stays silent on the rest. Small datasets (6-8 rows) achieve near-perfect performance; performance degrades on larger datasets with more items.

### Per-Dataset Performance (SFT Fine-tuned)

The SFT model produces output on all 30 datasets but performance collapses on larger datasets:

| Dataset | Rows x Cols | P | R | F1 | JSON |
|---------|-------------|---|---|----|------|
| eval_002_8x4 | 8 x 4 | 81% | 81% | 81% | valid |
| eval_018_5x6 | 5 x 6 | 36% | 73% | 48% | valid |
| eval_019_6x4 | 6 x 4 | 70% | 93% | 80% | valid |
| eval_027_7x4 | 7 x 4 | 11% | 100% | 20% | valid |
| eval_028_6x5 | 6 x 5 | 33% | 86% | 48% | valid |
| eval_030_8x4 | 8 x 4 | 68% | 72% | 70% | valid |
| *22 other datasets* | *10+ rows* | *0%* | *0%* | *0%* | *invalid* |

The model succeeds on small datasets (under 8 rows, under 6 columns) where the reasoning task fits within its capacity. On larger datasets, the CoT reasoning becomes too long, the model fails to produce valid JSON, and F1 drops to zero.

## Key Findings

1. **Fine-tuning teaches the task.** F1 from 1.0% (base) to 12.6% (SFT). The base model produces valid JSON on only 20% of datasets (6/30); the SFT model achieves valid JSON on 27% (8/30) but produces non-empty output on all 30 datasets. The 8 valid-JSON datasets are small (under 8 rows), where the reasoning fits within model capacity.

2. **Zero hallucination on the primary_v3 profile is the clearest win.** Both SFT and DPO models achieve 0.0% hallucination rate on the primary_v3 evaluation profile -- they never invent items absent from the input CSV under this configuration. The base model hallucinates at 6.7%, GPT-4.1-mini at 3.3%. (Under the alternative `reppenalty` profile, a 3% hallucination rate was observed.)

3. **DPO regression is an honest negative result.** SFT+DPO (11.8% F1) performs slightly worse than SFT alone (12.6% F1). DPO improved specific datasets but hurt others, suggesting the 606-pair DPO dataset was insufficient to consistently improve the policy.

4. **Scale dominates domain-specific fine-tuning.** GPT-4.1-mini reaches 49.1% F1 on the same 30 held-out evaluation pool, while the fine-tuned 7B SFT adapter reaches 12.6% archived / 13.07% verified under primary_v3, even with a stronger two-phase protocol. This demonstrates that at this task complexity, model capacity matters more than task-specific training data.

5. **Dataset size is the performance cliff.** Both fine-tuned models and GPT-4.1-mini show clear performance degradation on larger datasets. The fine-tuned model's failure mode is more abrupt: valid JSON on small datasets, complete parse failure on large ones.
