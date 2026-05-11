# Itemset Extraction via Fine-Tuned Qwen2.5-7B

Bakalárska práca — Fine-tuning malého LLM modelu pre úlohu hľadania frequent itemsetov.
Bachelor's thesis at FIS VŠE (Prague University of Economics and Business).

## What This Is

A modular pipeline that uses the **Apriori algorithm as a deterministic oracle** to generate
ground-truth training data, then **fine-tunes Qwen2.5-7B-Instruct** (LoRA/QLoRA) to approximate
frequent itemset extraction directly from tabular data — without executing Apriori at inference time.

**Core question:** Can a small autoregressive model *learn* an algorithm, rather than just *call* one?
Tool-use (LLM calling Apriori via function calling) was a known alternative from the start,
deliberately excluded to study the fundamental limits of pure LLM reasoning.

**Key finding:** Iteration 4 (SFT with corrected configuration) achieved F1 = 0.13 (average, primary_v3)
and up to F1 = 0.94 on small datasets (complexity ≤ 32). A controlled DPO ablation study confirmed
Council I's recommendation that DPO is contraproductive for rigid-format tasks. The best commercial
baseline (o4-mini) achieves F1 = 0.41. Full analysis in the thesis.

## Repository Scope

The documentation in this repository intentionally records the full thesis research journey, including
methodology, experiments, decisions, results, and references to generated artifacts. The Git repository
itself is kept code-focused for simplicity and reusability: generated datasets, training exports,
HuggingFace dataset shards, local databases, logs, and evaluation outputs are not committed.

## Quick Start

```bash
# Prerequisites: Python 3.10+, CUDA 12.x for training (inference works on CPU)

# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys (copy template and fill in)
cp openai.env.template openai.env

# 3. Run the pipeline on a single dataset
python pipeline.py --data data/datasets_v2/ds_0001_7x12.csv \
    --min-support 3 --max-size 3 --llm-full --llm-model gpt-4.1-mini

# 4. Run on entire dataset directory (500 datasets)
python pipeline.py --data-dir data/datasets_v2 \
    --min-support 3 --max-size 3 --llm-full --llm-model gpt-4.1-mini
```

## Training Data Generation

```bash
# Phase 1 — SFT with Chain-of-Thought (272 examples v3)
python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v3.json

# Phase 2 — DPO from real LLM failures (606 preference pairs)
# Note: DPO was run as a controlled ablation study on top of SFT.
# Iteration 4's primary result is the SFT-only checkpoint.
python src/training/export_real_dpo_data.py --db runs.db --output data/dpo_real_v2.json

# Upload to HuggingFace Hub
python src/training/upload_dataset_to_hf.py \
    --dataset-path data/hf_dataset_v3 --repo OliverSlivka/itemset-extraction-v3
```

## Evaluation

```bash
# Evaluate fine-tuned model from HuggingFace Hub
python src/evaluation/eval_finetuned_model.py \
    --model-path OliverSlivka/qwen2.5-7b-itemset-extractor

# Generate visualizations from runs.db
python src/utils/visualization.py --db runs.db --outdir visuals --bins 5
```

## Project Structure

```
itemsety-qwen-finetuning/
├── pipeline.py              # Main entry point (~835 lines)
├── requirements.txt         # Core dependencies
├── src/
│   ├── data_generation/     # Dataset generation from real-world sources
│   ├── training/            # SFT/DPO/GRPO data generators + training utils
│   ├── evaluation/          # Fine-tuned model evaluation
│   └── utils/               # Visualization, council advisor
├── data/
│   ├── datasets_v2/         # 500 generated training datasets (4-26 rows, 3-20 cols)
│   └── eval_datasets_v2/    # 30 holdout evaluation datasets
├── real_datasets/           # 8 real-world source files (5 domains) — NOT on GitHub
├── config/                  # Training configuration YAML files
└── artifacts/               # Pipeline outputs (Apriori, LLM extraction, validation)
```

## Dataset

Training dataset published on HuggingFace Hub:
[OliverSlivka/itemset-extraction-v3](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v3)
— 272 SFT-CoT examples + 606 DPO preference pairs in 3 configurations (sft/dpo/grpo).

## Architecture

**Pipeline flow:** CSV loading → Apriori ground truth → LLM extraction → validation (13 checks) → SQLite persistence

**Training strategy:**
- Phase 1 (SFT-CoT): 272 examples with structured `<think>` reasoning — this is Iteration 4's primary result
- Phase 2 (DPO ablation): 606 preference pairs run as a controlled experiment to verify Council I's recommendation. Apriori = chosen, real LLM failures = rejected. Result: DPO slightly degraded performance (SFT F1=0.13 vs DPO F1=0.12), confirming DPO is contraproductive for rigid-format tasks.
- Phase 3 (GRPO): Diagnosed as harmful through reward hacking analysis. Skipped in Iteration 4.

**Key design:**
- Apriori is the deterministic oracle — no human annotation needed
- DPO ablation uses real failures from 4 commercial models (GPT-4.1-mini, GPT-4.1-nano, o4-mini, GPT-4o), with 99.5% of errors being hallucinated evidence rows
- LoRA + Unsloth (2× faster, 70% less VRAM); adapter-only saves on 4-bit models
- LLM Council methodology: 4 expert models independently diagnose failures, then synthesize consensus
- Hash-based artifact naming (SHA256 prefix) prevents overwrites
- Exit codes: 0 = success, 1 = all runs failed, 2 = bad data-dir, 3 = missing LLM credentials

## Hardware Requirements

| Use Case | GPU VRAM | Recommended |
|----------|----------|-------------|
| Pipeline inference (API) | None | Any CPU |
| Model inference (local) | 16 GB | Single consumer GPU |
| Training (QLoRA 4-bit) | 16–24 GB | NVIDIA RTX 4090 / A10G |
| Full training (reported) | 144 GB | NVIDIA H200 NVL |

## Results Summary

| Model | F1 | Precision | Recall | Notes |
|-------|----|-----------|--------|-------|
| o4-mini (baseline) | 0.41 | 0.69 | 0.37 | Best commercial model, 496 runs |
| gpt-4.1-mini (baseline) | 0.33 | 0.79 | 0.24 | 496 runs |
| Iteration 1 (0.5B) | — | — | — | Format failure — insufficient capacity |
| Iteration 2 (7B) | 0.0 | 0.0 | 0.0 | 6 config errors identified by Council I |
| **Iteration 4 SFT** (7B, primary_v3) | **0.13** | 0.13 | 0.19 | **Primary result.** Up to 0.94 on small datasets |
| Iteration 4 DPO ablation (raw_capture) | 0.18 | 0.18 | 0.24 | Larger token budget (8192). 70% hit limit. |

**Note:** Iteration 4's headline results come from the SFT-only checkpoint (F1=0.13 average, F1=0.94 on small datasets, 0% hallucinations, 100% think rate). The DPO raw_capture result (F1=0.18) is from the ablation study and reflects a larger token budget, not DPO superiority. Full results and methodology in the thesis.

## Model

Fine-tuned LoRA adapter published on HuggingFace Hub:
[OliverSlivka/qwen2.5-7b-itemset-extractor](https://huggingface.co/OliverSlivka/qwen2.5-7b-itemset-extractor)

## License

Code: See [LICENSE](LICENSE)
Dataset: [OliverSlivka/itemset-extraction-v3](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v3)
Thesis: FIS VŠE 2026
