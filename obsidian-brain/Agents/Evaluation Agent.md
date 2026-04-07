# Evaluation Agent Memory

Persistent knowledge store for model evaluation insights.

**Agent file:** `.github/agents/evaluation-agent.md`  
**Tags:** #agent/evaluation

---

<!-- Append new memories below, newest last -->

## 2026-02-22 – LLM Council Advisor integrated

### What was added
A **3-stage multi-LLM council** (inspired by [karpathy/llm-council](https://github.com/karpathy/llm-council)) is now part of the evaluation pipeline. It uses **OpenRouter** to query multiple LLMs in parallel and produces a synthesised expert verdict.

**New files:**
- `src/utils/openrouter_client.py` – async OpenRouter HTTP client (`query_model`, `query_models_parallel`)
- `src/evaluation/council_advisor.py` – `CouncilAdvisor` class + standalone CLI

**Modified files:**
- `src/evaluation/eval_finetuned_model.py` – added `--council`, `--council-training-advice`, `--training-script`, `--council-models`, `--chairman` flags
- `requirements.txt` – added `httpx>=0.27.0`
- `openai.env.template` – added OPENROUTER_API_KEY comment
- `openrouter.env.template` – new dedicated template for OpenRouter config

### Council flow
| Stage | What happens |
|---|---|
| 1 – Individual opinions | 4 council LLMs independently answer in parallel |
| 2 – Peer ranking | Each LLM ranks the other (anonymised) responses |
| 3 – Chairman synthesis | Chairman LLM produces a final, prioritised recommendation |

**Default council models:** `claude-3-5-haiku`, `gpt-4o-mini`, `gemini-flash-1.5`, `llama-3.3-70b`  
**Default chairman:** `claude-3-5-sonnet`

### Two council use-cases

1. **Eval analysis** – council analyses F1 / precision / recall / Jaccard and explains root causes + recommendations.
2. **Training advice** – council reads the training script + metrics and suggests concrete hyperparameter / data improvements.

### Setup required
```bash
cp openrouter.env.template openrouter.env
# fill in OPENROUTER_API_KEY=sk-or-v1-...
pip install httpx
```

### Usage examples
```bash
# After eval, auto-trigger council analysis
python src/evaluation/eval_finetuned_model.py \
    --data-dir eval_datasets --min-support 3 \
    --council --council-training-advice

# Standalone eval analysis
python src/evaluation/council_advisor.py analyze \
    --eval-results eval_results/evaluation_summary.json

# Standalone training advice
python src/evaluation/council_advisor.py advise \
    --training-script src/training/run_sft_full.py \
    --eval-results eval_results/evaluation_summary.json
```

### Output files
| File | Contents |
|---|---|
| `eval_results/council_eval_analysis.json` | Full 3-stage council on eval metrics |
| `eval_results/council_training_advice.json` | Full 3-stage council on training script |

### See also
[[Training Agent]] [[Pipeline Agent]]

---

## 2026-03-01 – Catastrophic Evaluation: Qwen2.5-7B v1 (0% F1)

### Model Evaluated
- **Model:** `OliverSlivka/qwen2.5-7b-itemset-extractor`
- **Base:** `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- **Training:** 3-phase (SFT-CoT 348ex → DPO-Real 606pairs → GRPO 200steps)
- **Eval datasets:** 20 from `OliverSlivka/itemset-eval-v2` (seed 99999, zero training overlap)

### Results Summary
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Avg F1 | **0.0%** | ≥80% | ❌ |
| Precision | **0.0%** | ≥80% | ❌ |
| Recall | **0.0%** | ≥80% | ❌ |
| JSON Parse Rate | **95.0%** | ≥90% | ✅ |
| Think Rate | **0.0%** | 100% | ❌ |
| Hallucination Rate | **88.8%** | ≤5% | ❌ |
| Avg Inference Time | **4.6s** | ≤60s | ✅ |

### Root Causes Identified (Council-confirmed, unanimous)

1. **GRPO Reward Functions Blind to Format** — `itemset_f1_reward` used raw `frozenset()` comparison with NO normalization. `frozenset(["age:15"]) ≠ frozenset(["age","15"])` → F1 was **0.0 for all 200 GRPO steps**. Model received zero correctness signal.

2. **GRPO Caused Catastrophic Forgetting** — Only `json_format_reward` (0.95-1.0) gave signal. Model gradient-descended into "output pretty JSON with bare column names" and abandoned `col:value` format + `<think>` reasoning learned in SFT/DPO.

3. **Insufficient Training Data** — 348 SFT examples for 7B model → memorized format without generalizing algorithm.

4. **Possible DPO Mislabeling** — Some "rejected" examples may contain correct `col:value` format, training model to AVOID correct format.

### Failure Patterns (20 datasets)
| Pattern | Count | % |
|---------|-------|---|
| no_think | 20/20 | 100% |
| column_only (bare names) | 16/20 | 80% |
| correct_col_value | 3/20 | 15% |
| item_split | 2/20 | 10% |

### Council Analysis (4 models + chairman)
- **Council models:** claude-sonnet-4.6, gemini-3-flash-preview, deepseek-v3.2, grok-4.1-fast
- **Chairman:** claude-opus-4.6
- **Rankings:** Claude Sonnet #1 (avg 1.0), Grok #2 (avg 2.25), Gemini #3 (avg 2.75), DeepSeek #4 (avg 4.0, truncated response)
- **Full report:** `docs/reports/council_eval_analysis.json`

### Key Council Recommendations (Unanimous)
1. **Skip GRPO** until SFT achieves ≥50% F1
2. **Increase SFT data** to 2,000+ examples
3. **LoRA r=32** (up from 16)
4. **SFT LR=1e-4** (down from 2e-4), 5 epochs
5. **Fix reward functions** — add `col:value` format check + item normalization
6. **Audit DPO data** for mislabeled pairs
7. **Try 3B first** for rapid iteration, fallback to 7B if needed

### Lessons Learned (NEVER repeat these)
- ❌ Never run GRPO with reward functions that give ~0 correctness signal
- ❌ Never skip normalization in itemset matching (lowercase, strip, `col:value` format)
- ❌ Never use `json_format_reward` alone without format-content checks
- ❌ Never assume DPO `reward_accuracy=100%` means model learned useful preferences
- ✅ Always test SFT baseline before adding DPO/GRPO
- ✅ Always verify reward functions produce non-zero signal before GRPO training

### Files Created
- `docs/reports/EVALUATION_REPORT_v1_2026-03-01.md` — Full evaluation report
- `docs/reports/council_eval_analysis.json` — Council 1: eval analysis output
- `docs/reports/council_training_advice.json` — Council 2: training advice output
- `scripts/analyze_eval_outputs.py` — Systematic failure pattern analyzer
- `scripts/run_council_analysis.py` — Council runner script

### See also
[[Training Agent]] [[Experiments/v1_3phase_7b]] [[Decisions/Skip GRPO Until SFT Works]]

---

## 2026-03-01 – Council 2: Training Advice (Synthesized)

### Council Setup
- **Models:** claude-sonnet-4.6, gemini-3-flash-preview, deepseek-v3.2, grok-4.1-fast
- **Chairman:** claude-opus-4.6
- **Rankings:** Claude Sonnet #1 (1.0), Grok #2 (2.5), Gemini #3 (2.5), DeepSeek #4 (4.0)
- **Full output:** `docs/reports/council_training_advice.json`

### Critical Bug: LoRA Alpha/Rank Ratio (3/4 consensus)
Original `lora_alpha=16, lora_r=64` → ratio = **0.25** (weights barely updated).  
Fix: `lora_alpha=64, lora_r=32` → ratio = **2.0** (standard fine-tuning).  
**Impact:** This alone may explain why SFT format didn't stick — the updates were 8× too small.

### Critical Bug: Destructive 4-bit Merge (3/4 consensus)
`merged_4bit_forced` between phases introduces cascading quantization noise.  
Fix: **Save adapters only**, reload base + adapters for next phase.

### Unanimous: Remove DPO Permanently (4/4)
DPO's contrastive loss is designed for stylistic preference, not hard structural constraints.  
DPO destroyed `<think>` reasoning because chosen/rejected share early tokens → model uncertain about ALL formats.

### Unanimous: Stay 7B (4/4)
Contradicts Council 1's suggestion to "try 3B first."  
**Resolution:** Council 2 had the full training code + deeper analysis → 7B recommendation is stronger.  
Failures are configuration issues, not model capacity issues.

### Corrected CONFIG (Council-Synthesized)
| Parameter | Original | Fixed |
|-----------|----------|-------|
| lora_r | 64 | 32 |
| lora_alpha | 16 | 64 (ratio 0.25→2.0) |
| lora_dropout | 0 | 0.05 |
| sft_epochs | 2 | 5 |
| sft_lr | 2e-4 | 1e-4 |
| sft_grad_accum | 4 | 8 |
| packing | unset | False |
| weight_decay | unset | 0.01 |
| max_grad_norm | unset | 1.0 |
| DPO | present | **REMOVED** |
| save method | merged_4bit | adapters only |
| grpo_steps | 200 | 500 |
| grpo_lr | 5e-6 | 2e-6 |
| grpo_num_generations | 4 | 6 |
| grpo_completion_length | 2048 | 3000 |
| max_seq_length | 4096 | 6144 |

### 5 GRPO Reward Functions (Council-Designed)
1. `reward_structure` — checks `<think>` + `<answer>` tags → [0, 1]
2. `reward_colval_format` — fraction of items using `col:val` → [0, 1]
3. `reward_f1` — F1 with sqrt scaling for partial credit → [0, 1]
4. `reward_recall_bonus` — 70% recall + 30% precision → [0, 1]
5. `reward_grounding` — anti-hallucination (items in CSV) → [0, 1]

### Format Verification Gate
Between SFT and GRPO: generate 20 val samples, check `<think>`, `<answer>`, `col:val`.  
Gate: ≥80% → proceed. <50% → increase SFT epochs.

### Action Checklist
1. ☐ Fix LoRA: r=32, alpha=64
2. ☐ Set packing=False
3. ☐ SFT 5 epochs, LR 1e-4
4. ☐ Add weight_decay=0.01, max_grad_norm=1.0
5. ☐ Remove DPO entirely
6. ☐ Save adapters only (no merged_4bit_forced)
7. ☐ Implement 5 reward functions with partial credit
8. ☐ Add format verification gate
9. ☐ Test reward functions on good/bad examples before training
10. ☐ Run SFT-only first, eval, then GRPO if compliance >80%

### Key Lessons (NEVER repeat)
- ❌ Never use lora_alpha < lora_r (ratio <1.0 means near-zero learning)
- ❌ Never merge to 4-bit between training phases (quantization cascade)
- ❌ Never use DPO for hard structural constraint tasks
- ❌ Never run GRPO without testing reward functions on known examples first
- ✅ Always save adapters only between phases
- ✅ Always add format verification gate before RL phases
- ✅ Always use partial-credit reward functions (not binary)

### Council 1 vs Council 2 Disagreements Resolved
| Topic | Council 1 | Council 2 | Resolution |
|-------|-----------|-----------|------------|
| Model size | Try 3B first | Stay 7B (4/4) | **7B** — Council 2 had full code context |
| DPO | Skip temporarily | Remove permanently | **Remove permanently** |
| LoRA r | 32 | 32 (consensus) | **32** |
| max_seq_length | 4096 | 6144 | **6144** — long CSVs need space |

### See also
[[Training Agent]] [[Experiments/v1_3phase_7b]] [[Decisions/Skip DPO Permanently]]

---

## [2026-03-17] 🔬 Diamond Knowledge — Inference & Eval Patterns

**Source:** Review of 11 Unsloth × Qwen notebooks (diamond phase extraction)

### Inference Temperature Calibration

| Source | Temperature | Context |
|--------|-------------|---------|
| Unsloth default | 1.5 | Creative, high diversity |
| Qwen3 team (reasoning) | 0.6, top_p=0.95, top_k=20 | Precision tasks |
| Qwen3 team (chat) | 0.7, top_p=0.8, top_k=20 | General instruction |
| Our v3 Council | 0.3, top_k=50, top_p=0.90 | Escape repetition loops |

**Decision:** Keep Council's 0.3 — our model had a proven repetition loop problem. Once the repetition issue is resolved by better training (v3.2+), consider testing Qwen3 team's 0.6 for reasoning mode.

### `FastLanguageModel.for_inference()` — Confirmed Essential

Already applied in our notebook Cell 19. Confirmed across all 11 reviewed notebooks: this enables 2× native inference speedup. Must be called before `.generate()`.

### A/B Comparison Pattern (Future Enhancement)

From DeepSeek-R1 notebook: 20-sample A/B loop comparing with/without LoRA:
```python
for dataset in eval_datasets[:20]:
    output_base = generate(model, dataset, use_lora=False)
    output_ft = generate(model, dataset, use_lora=True)
    # Compare F1, exact match, JSON validity
```
**Actionable:** Add `--ab-comparison` flag to `eval_finetuned_model.py` in future versions.

### Label Masking Verification Before Eval

Diamond review: wrong `instruction_part`/`response_part` strings fail silently — masking breaks and model trains on prompts instead of responses. Always decode labels after `train_on_responses_only()` and visually confirm masking. New verification cell added to v3.2 notebook.

**Tags:** #diamond-knowledge #inference-config #evaluation-patterns

See also: [[Training Agent]] [[References/Unsloth Notebook Patterns]]
