# Evaluation Results — v3 Final

These are the canonical evaluation results for the thesis, produced on 2026-04-12.

## Model Checkpoints

| Key | Label | Source |
|-----|-------|--------|
| `sft_local` | SFT local adapter | `./sft_checkpoint` (Phase 1) |
| `dpo_local` | DPO local adapter | `./dpo_checkpoint` (Phase 1+2) |
| `dpo_hf` | DPO HuggingFace adapter | `OliverSlivka/qwen2.5-7b-itemset-extractor` |

## Evaluation Profiles

| Profile | Description |
|---------|-------------|
| `primary_v3` | Production-style v3 two-phase inference |
| `raw_capture` | Large-budget single-phase diagnostic capture |
| `reppenalty` | Repetition-penalty diagnostic (anti-loop baseline) |

## Eval Dataset

- **Source:** `data/eval_datasets_v2/` (30 held-out CSV files, no training overlap)
- **Prefix:** `eval_` (validated zero overlap with `data/datasets_v2/`)
- **Parameters:** min_support=3, max_size=3

## Results Summary

| Model | Profile | Avg P | Avg R | Avg F1 | Parse | Hallucination |
|-------|---------|-------|-------|--------|-------|---------------|
| SFT | primary_v3 | 0.1343 | 0.1917 | 0.1264 | — | 0.0000 |
| DPO | primary_v3 | 0.1141 | 0.1574 | 0.1176 | — | 0.0000 |
| DPO | raw_capture | 0.1804 | 0.2399 | 0.1753 | — | 0.0000 |
| DPO | reppenalty | 0.0000 | 0.0000 | 0.0000 | — | 0.0333 |

## Files

- `eval_v3_run_results.md` — Full per-dataset run log with aggregate summary table
- `council_eval_analysis.json` — Council diagnostic analysis
- `council_input_eval_v3_summary.json` — Council input summary

## Regenerating

```bash
# From the eval notebook
jupyter execute notebooks/eval_model_2026-04-12_v3.ipynb

# Or via eval script
python src/evaluation/eval_finetuned_model.py --model-path OliverSlivka/qwen2.5-7b-itemset-extractor
```
