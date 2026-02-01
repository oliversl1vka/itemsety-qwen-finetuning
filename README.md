# Frequent Itemset Extraction + Qwen Fine-Tuning Pipeline

A complete ML pipeline for extracting frequent itemsets from CSV datasets using Apriori algorithm + Azure OpenAI, then fine-tuning Qwen models to perform the same task without Apriori.

## 🎯 Project Goal

Train a small LLM (Qwen2.5-3B) to extract frequent itemsets from transactional data, replacing the traditional Apriori algorithm with a learned model.

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
│   │   ├── run_sft_full.py        # Production training (439 examples)
│   │   ├── run_sft_test.py        # Test training (50 examples)
│   │   ├── create_training_data_v2.py
│   │   ├── export_training_data.py
│   │   ├── create_hf_dataset.py
│   │   └── upload_dataset_to_hf.py
│   ├── evaluation/                # Model evaluation
│   │   └── eval_finetuned_model.py
│   ├── data_generation/           # Dataset generation
│   │   ├── generate_datasets_v2.py
│   │   └── generate_eval_datasets_v2.py
│   └── utils/                     # Utility scripts
│       ├── visualization.py
│       ├── compute_stats.py
│       └── analyze_and_filter_datasets.py
│
├── data/                          # All data files
│   ├── datasets_v2/               # Generated CSV datasets (500)
│   ├── training_v2/               # Training examples (JSONL)
│   ├── hf_dataset_v2/             # HuggingFace format dataset
│   └── hf_dataset_v1/             # Legacy dataset (archived)
│
├── docs/                          # Documentation
│   ├── guides/                    # How-to guides
│   │   ├── FINETUNING_README.md
│   │   ├── TRAINING_QUICKSTART.md
│   │   └── DEPLOYMENT_GUIDE.md
│   ├── reports/                   # Experiment reports
│   │   ├── EVALUATION_REPORT.md
│   │   └── TRAINING_STATUS.md
│   └── archive/                   # Historical docs
│
├── scripts/                       # Operational scripts
│   ├── deployment/                # HF Space & deployment
│   │   ├── app.py                 # Gradio app
│   │   ├── deploy_to_hf_space.ps1
│   │   └── README_SPACE.md
│   ├── colab/                     # Google Colab scripts
│   │   └── COLAB_*.py
│   └── db_maintenance/            # Database utilities
│       ├── db_editor.py
│       └── check_*.py
│
├── notebooks/                     # Jupyter notebooks
│   ├── qwen_finetuning_server.ipynb
│   └── sft_trl_lora_qlora.ipynb
│
├── agents/                        # Agent system definitions
├── agents_log/                    # Agent activity logs
├── agents_memory/                 # Agent persistent memory
│
├── archive/                       # Archived/legacy files
│   ├── legacy_scripts/            # Old script versions
│   ├── experiments/               # Test results & experiments
│   └── resources/                 # Reference materials
│
├── artifacts/                     # Pipeline outputs (gitignored)
├── logs/                          # Execution logs (gitignored)
└── runs.db                        # SQLite database (gitignored)
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone https://github.com/oliversl1vka/itemsety-qwen-finetuning.git
cd itemsety-qwen-finetuning
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI

```bash
cp azure.env.template azure.env
# Edit azure.env with your credentials
```

### 3. Run Pipeline

```bash
# Single dataset
python pipeline.py --data data/datasets_v2/ds_0001_5x53.csv --min-support 3 --llm-full

# Batch processing
python pipeline.py --data-dir data/datasets_v2 --min-support 3 --max-size 3 --llm-full
```

### 4. Train Model

```bash
# Test run (50 examples, ~15 min)
python src/training/run_sft_test.py

# Production training (439 examples, ~60 min)
python src/training/run_sft_full.py
```

### 5. Evaluate

```bash
python src/evaluation/eval_finetuned_model.py --model-path OliverSlivka/qwen2.5-3b-itemset-extractor
```

## 📊 Key Metrics

| Metric | Target | Current |
|--------|--------|---------|
| F1 Score | ≥ 0.80 | 0.65 |
| JSON Parse Rate | ≥ 0.90 | 0.95 |
| Inference Time | ≤ 60s | 45s |

## 🔧 Core Components

### Pipeline (`pipeline.py`)
- Loads CSV datasets (auto-detects format)
- Runs Apriori algorithm for ground truth
- Calls Azure OpenAI for LLM extraction
- Validates outputs (13 invariants)
- Persists results to SQLite

### Training (`src/training/`)
- Exports validated runs as training data
- Creates HuggingFace datasets
- Fine-tunes Qwen2.5-3B with LoRA/QLoRA
- Pushes models to HuggingFace Hub

### Evaluation (`src/evaluation/`)
- Generates evaluation datasets
- Computes P/R/F1 vs Apriori ground truth
- Produces detailed reports

## 📚 Documentation

- [Fine-tuning Guide](docs/guides/FINETUNING_README.md)
- [Training Quickstart](docs/guides/TRAINING_QUICKSTART.md)
- [Deployment Guide](docs/guides/DEPLOYMENT_GUIDE.md)
- [Evaluation Report](docs/reports/EVALUATION_REPORT.md)

## 🤖 Agent System

This project uses a multi-agent orchestration system. See [AGENTS.md](AGENTS.md) for details.

## 📄 License

Apache 2.0

## 👤 Author

Oliver Slivka - [@oliversl1vka](https://github.com/oliversl1vka)
