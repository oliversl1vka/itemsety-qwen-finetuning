# Frequent Itemset Extraction + Qwen Fine-Tuning Pipeline

A complete ML pipeline for extracting frequent itemsets from CSV datasets using Apriori algorithm + OpenAI, then fine-tuning Qwen models to perform the same task without Apriori.

## 🎯 Project Goal

Train Qwen2.5-7B to extract frequent itemsets from transactional data, replacing the traditional Apriori algorithm with a learned model using a **3-phase training approach**: SFT with Chain-of-Thought → DPO with real LLM failures → GRPO with Apriori reward.

## 📁 Repository Structure

```
itemsety-qwen-finetuning/
│
├── pipeline.py                    # Core extraction pipeline (Apriori + LLM)
├── extractor_system_prompt.md     # LLM system prompt for extraction
├── requirements.txt               # Python dependencies
│
├── src/                           # Source code modules
│   ├── training/                  # Fine-tuning scripts
│   │   ├── training_utils.py      # Shared utilities (CoT generator, prompts)
│   │   ├── generate_cot_sft_data.py          # Phase 1: SFT-CoT data (348 examples)
│   │   ├── export_real_dpo_data.py           # Phase 2: DPO with real LLM failures (606 pairs)
│   │   ├── build_hf_dataset_v2.py            # Build HF dataset (SFT/DPO/GRPO configs)
│   │   └── upload_dataset_to_hf.py           # Push dataset to HuggingFace Hub
│   ├── evaluation/                # Model evaluation
│   │   ├── eval_finetuned_model.py
│   │   └── council_advisor.py     # LLM Council multi-model analysis
│   ├── data_generation/           # Dataset generation
│   │   ├── generate_datasets_v2.py
│   │   └── generate_eval_datasets_v2.py
│   └── utils/                     # Utility scripts
│       ├── visualization.py
│       ├── compute_stats.py
│       └── openrouter_client.py
│
├── data/                          # All data files
│   ├── datasets_v2/               # Generated CSV datasets (500)
│   ├── sft_cot_v2.json            # SFT-CoT training examples (348)
│   ├── dpo_real_v2.json           # DPO preference pairs (606)
│   └── hf_dataset_v2/             # HuggingFace dataset (3 configs: sft/dpo/grpo)
│
├── notebooks/                     # Jupyter notebooks
│   ├── training_3phase_7b.ipynb   # 3-phase training notebook (SFT→DPO→GRPO)
│   └── training_sft_dpo_template.ipynb  # Legacy 2-phase template
│
├── obsidian-brain/                # 🧠 Obsidian knowledge vault
│   ├── Home.md                    # Vault navigation hub
│   ├── Agents/                    # Agent memory notes (9 files)
│   ├── References/                # API limits, model comparison, etc.
│   ├── Logs/                      # Per-run activity logs
│   ├── Experiments/               # Training experiment reports
│   └── Decisions/                 # Architecture decisions
│
├── .github/
│   ├── agents/                    # 9 agent definition files
│   ├── agents_memory/             # workflow_state.json + workflow_state.py
│   └── copilot-instructions.md    # AI coding agent instructions
│
├── artifacts/                     # Pipeline outputs (gitignored)
├── logs/                          # Execution logs (gitignored)
└── runs.db                        # SQLite database (gitignored, ~1600 runs)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone https://github.com/oliversl1vka/itemsety-qwen-finetuning.git
cd itemsety-qwen-finetuning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure OpenAI

```bash
cp openai.env.template openai.env
# Edit openai.env with your API key
```

### 3. Run Pipeline (Apriori + LLM extraction)

```bash
# Single dataset
python pipeline.py --data data/datasets_v2/ds_0001_7x12.csv --min-support 3 --llm-full --llm-model gpt-4.1-mini

# Full batch (500 datasets)
python pipeline.py --data-dir data/datasets_v2 --min-support 3 --max-size 3 --llm-full --llm-model gpt-4.1-mini
```

### 4. Prepare Training Data (3-phase)

```bash
# Phase 1: Generate SFT-CoT examples (uses Apriori ground truth + synthetic reasoning)
python src/training/generate_cot_sft_data.py --db runs.db --output data/sft_cot_v2.json

# Phase 2: Export DPO pairs (Apriori=chosen, real LLM failures=rejected)
python src/training/export_real_dpo_data.py --db runs.db --output data/dpo_real_v2.json

# Build HuggingFace dataset (SFT + DPO + GRPO configs)
python src/training/build_hf_dataset_v2.py \
  --sft data/sft_cot_v2.json \
  --dpo data/dpo_real_v2.json \
  --output data/hf_dataset_v2

# Push to HuggingFace Hub (each version → own repo, never overwrite old)
python src/training/upload_dataset_to_hf.py --dataset-path data/hf_dataset_v3 --repo OliverSlivka/itemset-extraction-v3
```

### 5. Train Model (on Jupyter server)

Upload `notebooks/training_3phase_7b.ipynb` + dataset to your GPU server.

**3-phase training:**
- **Phase 1 — SFT-CoT** (1 epoch): Learn structured `<think>` reasoning + JSON output
- **Phase 2 — DPO** (2 epochs): Prefer Apriori ground truth over real LLM failures
- **Phase 3 — GRPO** (1 epoch): Optimize with Apriori F1 reward signal

### 6. Evaluate

```bash
python src/evaluation/eval_finetuned_model.py --model-path OliverSlivka/qwen2.5-7b-itemset-extractor
```

## 📊 Training Data Summary

| Phase | Examples | Source |
|-------|----------|--------|
| SFT-CoT | 348 (314 train / 34 val) | Validated Apriori runs + synthetic CoT reasoning |
| DPO | 606 pairs (546 train / 60 val) | Apriori (chosen) vs real LLM failures (rejected) |
| GRPO | 314 (same datasets as SFT) | Apriori ground truth as reward signal |

**Database:** 1600+ pipeline runs across 4 LLM models (gpt-4.1-mini, gpt-4.1-nano, gpt-4o, o4-mini)

## 📊 Key Metrics

| Metric | Target | Status |
|--------|--------|--------|
| F1 Score (vs Apriori) | ≥ 0.80 | Training pending |
| JSON Parse Rate | ≥ 0.90 | Training pending |
| Hallucination Rate | ≤ 0.05 | Training pending |
| Inference Time | ≤ 60s | Training pending |

## 🔧 Core Components

### Pipeline (`pipeline.py`)
- Loads CSV datasets (auto-detects format: wide/long/single-column)
- Runs Apriori algorithm for deterministic ground truth
- Calls OpenAI for LLM extraction (GPT-4.1-mini, GPT-4o, o4-mini, etc.)
- Validates outputs (13 invariants)
- Persists results to SQLite (`runs.db`)

### Training (`src/training/`)
- **`training_utils.py`** — Shared utilities: compact system prompt (~150 tokens), CoT generator, CSV loader
- **`generate_cot_sft_data.py`** — Generates SFT examples with `<think>` chain-of-thought reasoning
- **`export_real_dpo_data.py`** — Exports DPO pairs using real LLM failures (not synthetic corruptions)
- **`build_hf_dataset_v2.py`** — Builds HuggingFace dataset with 3 configs (sft/dpo/grpo)
- **`training_3phase_7b.ipynb`** — Complete training notebook: SFT → DPO → GRPO for Qwen2.5-7B

### Evaluation (`src/evaluation/`)
- Generates fixed evaluation datasets (versioned, never change between models)
- Computes P/R/F1 vs Apriori ground truth
- LLM Council advisor (multi-model analysis via OpenRouter)

## 🤖 Agent System

This project uses a multi-agent orchestration system with 9 agents. See [AGENTS.md](AGENTS.md) for details.

## 📚 Documentation

- [AGENTS.md](AGENTS.md) — Multi-agent workflow guide
- [docs/guides/](docs/guides/) — How-to guides
- [docs/reports/](docs/reports/) — Experiment reports
- [obsidian-brain/](obsidian-brain/) — Persistent knowledge vault

## 📄 License

Apache 2.0

## 👤 Author

Oliver Slivka - [@oliversl1vka](https://github.com/oliversl1vka)
