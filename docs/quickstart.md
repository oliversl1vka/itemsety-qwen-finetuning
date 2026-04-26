# Quick Start

## Prerequisites

- Python 3.10+
- Git
- API key for OpenAI or OpenRouter (for running the extraction pipeline)
- GPU with 8+ GB VRAM (for running the fine-tuned model locally; optional -- pipeline works with cloud APIs)

## Installation

```bash
git clone https://github.com/oliversl1vka/itemsety-qwen-finetuning
cd itemsety-qwen-finetuning
pip install -r requirements.txt
```

## Configure Credentials

```bash
cp openai.env.template openai.env       # Fill in your OpenAI API key
cp openrouter.env.template openrouter.env # Fill in your OpenRouter key
cp hf.env.template hf.env               # Fill in your HuggingFace token
```

Each `.env` file is gitignored and will never be committed.

## Run the Pipeline

Extract frequent itemsets from a single dataset using GPT-4.1-mini:

```bash
python pipeline.py \
  --data data/datasets_v2/ds_0001_7x12.csv \
  --min-support 3 \
  --max-size 3 \
  --llm-full \
  --llm-model gpt-4.1-mini
```

This produces:

- A JSON artifact in `artifacts/` with itemsets, counts, and evidence rows
- A row in `runs.db` (SQLite) tracking the run metadata and validation results

To run the full 500-dataset suite:

```bash
python pipeline.py \
  --data-dir data/datasets_v2 \
  --min-support 3 --max-size 3 \
  --llm-full --llm-model gpt-4.1-mini
```

## Evaluate the Fine-tuned Model

```bash
python src/evaluation/eval_finetuned_model.py \
  --model-path OliverSlivka/qwen2.5-7b-itemset-extractor
```

This loads the LoRA adapter from HuggingFace Hub, runs two-phase inference on the 30 held-out evaluation datasets, and reports precision, recall, F1, hallucination rate, and other metrics.

## Reproduce Training

!!! warning "Hardware requirements"
    Training requires a GPU with 16+ GB VRAM (e.g., NVIDIA T4, A100). The canonical training was performed on an A100 40GB.

The training notebook is at `notebooks/training_3phase_2026-03-09_v3.ipynb`. It implements:

1. **Phase 1 (SFT-CoT)**: 3 epochs on 272 examples with chain-of-thought reasoning
2. **Phase 2 (DPO)**: 1 epoch on 606 real LLM failure pairs

## Regenerate Training Data

If reproducing from scratch (requires a populated `runs.db`):

```bash
# Phase 1: SFT data with chain-of-thought
python src/training/generate_cot_sft_data.py \
  --db runs.db --output data/sft_cot_v3.json

# Phase 2: DPO data from real LLM failures
python src/training/export_real_dpo_data.py \
  --db runs.db --output data/dpo_real_v2.json

# Build HuggingFace dataset (3 configs: sft/dpo/grpo)
python src/training/build_hf_dataset_v2.py \
  --sft data/sft_cot_v3.json \
  --dpo data/dpo_real_v2.json \
  --output data/hf_dataset_v3
```

## Project Structure

```
pipeline.py                  # Core pipeline: CSV -> Apriori -> LLM -> validate -> SQLite
src/
  training/                  # Training data generation (SFT, DPO, HF dataset builder)
  evaluation/                # Model evaluation and inference utilities
  data_generation/           # Synthetic dataset generators
  utils/                     # OpenRouter client, visualization, diversity metrics
scripts/                     # Utilities: deployment, DB maintenance, analysis
notebooks/                   # Training and evaluation Jupyter notebooks
data/                        # Datasets, training data JSON, HF datasets
docs/                        # This documentation site
obsidian-brain/              # AI agent memory vault (development audit trail)
.github/agents/              # 9 specialized agent definitions
```
