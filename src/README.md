# Source Code (`src/`)

Core Python modules organized by function.

## Structure

```
src/
├── training/           # Model fine-tuning
├── evaluation/         # Model evaluation
├── data_generation/    # Dataset generation
└── utils/              # Utility functions
```

## Modules

### training/
Fine-tuning scripts for Qwen models:
- `run_sft_full.py` - Production training (439 examples, 3 epochs)
- `run_sft_test.py` - Quick test run (50 examples)
- `create_training_data_v2.py` - Generate training examples from validated runs
- `export_training_data.py` - Export from SQLite to JSON
- `create_hf_dataset.py` - Convert to HuggingFace format
- `upload_dataset_to_hf.py` - Push dataset to Hub

### evaluation/
Model performance evaluation:
- `eval_finetuned_model.py` - Compute P/R/F1 metrics vs Apriori

### data_generation/
Dataset creation:
- `generate_datasets_v2.py` - Generate 500 training datasets
- `generate_eval_datasets_v2.py` - Generate evaluation datasets

### utils/
Helper functions:
- `visualization.py` - Matplotlib charts and dashboards
- `compute_stats.py` - Dataset statistics
- `analyze_and_filter_datasets.py` - Dataset quality analysis
- `inspect_training_data.py` - Training data inspection
- `test_dataset_loading.py` - Dataset loading tests
