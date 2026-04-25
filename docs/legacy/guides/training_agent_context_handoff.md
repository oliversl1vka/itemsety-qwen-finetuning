# 🔄 Training Agent Context Handoff — v3 Implementation

## Current Status
**Stage 6 (`/validate`) is `in_progress`** — but we're between v2 eval and v3 training.

**What happened:** LLM Council (4 frontier models) diagnosed v2's 0.4% F1 catastrophic failure → root cause was `merged_4bit_forced` destroying LoRA deltas + repetition loops from verbose CoT. **All v3 code changes are implemented** but data has NOT been regenerated yet.

## v3 Changes Already Applied to Code (2026-03-09)

| File | Changes |
|------|---------|
| `src/training/training_utils.py` | New concise CoT generator (`generate_cot()`), column-grouped format, RESULT SUMMARY termination signal, `calculate_token_budget()` |
| `src/training/generate_cot_sft_data.py` | Default output → `data/sft_cot_v3.json`, max_tokens → 1800 |
| `src/evaluation/inference_utils.py` | **NEW** — ThinkStoppingCriteria, RepetitionDetector, two-phase generation (think@0.3, JSON@0.05), dynamic token budget |
| `src/evaluation/eval_finetuned_model.py` | v3 inference config: temp=0.3, `--two-phase` flag, imports from inference_utils |
| `notebooks/training_3phase_7b.ipynb` | 12 critical changes: seq_len 4096→2048, r=32 alpha=64, dropout=0.05, lr=1e-4, 3 epochs, DPO 1 epoch, GRPO SKIPPED, **adapter-only saves** (no more `merged_4bit_forced`), inference temp→0.3 |
| `obsidian-brain/Agents/Training Agent.md` | v3 implementation log appended |
| `.github/copilot-instructions.md` | Updated to v6.0 |

## What Needs to Happen Next (Operational Steps)

1. **Regenerate SFT data** with concise CoT: `python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v3.json`
2. **Rebuild HF dataset**: `python src/training/build_hf_dataset_v2.py --sft data/sft_cot_v3.json --dpo data/dpo_real_v2.json --output data/hf_dataset_v3`
3. **Push to HF Hub**: `python src/training/upload_dataset_to_hf.py --dataset-path data/hf_dataset_v3 --repo OliverSlivka/itemset-extraction-v3`
4. **Train on TLJH server** (H200 NVL GPU) using updated `notebooks/training_3phase_7b.ipynb`
5. **Evaluate** with `--temperature 0.3` and optionally `--two-phase`

## Key v3 Config (Council Consensus)

- **LoRA:** r=32, alpha=64 (ratio 2.0), dropout=0.05
- **SFT:** 3 epochs, lr=1e-4, warmup=0.10, weight_decay=0.01, seq_len=2048, eval every 50 steps, load_best_model=True
- **DPO:** 1 epoch, lr=5e-5, beta=0.1
- **GRPO:** SKIPPED (Unsloth bugs)
- **Save:** Adapter-only via `save_pretrained()` — NEVER `merged_4bit_forced`
- **Inference:** temp=0.3, top_k=50, top_p=0.90, StoppingCriteria at `</think>`

## Important Files to Read First
- `obsidian-brain/Agents/Training Agent.md` — full training history + v3 log
- `.github/agents_memory/workflow_state.json` — current workflow state
- `docs/reports/council_v2_eval_2026-03-07.json` — full council analysis

## Previous Results
- **v1:** 0% F1 (alpha/r=0.25, DPO killed `<think>`, GRPO rewards broken)
- **v2:** 0.4% F1 (SFT converged perfectly but `merged_4bit_forced` destroyed LoRA)
- **v2 adapter eval:** 7.5% mean F1 (one dataset hit 78.6% F1!) but 87% hit repetition loops
- **v3 target:** Conservative 50-65% F1, optimistic 75-85% F1
