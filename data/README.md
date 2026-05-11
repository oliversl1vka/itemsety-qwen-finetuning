# Data Directory

This repository is a code/template release and does not include generated datasets, training exports, HuggingFace Arrow shards, or local experiment databases.

Populate this directory by running the project scripts or by adding your own research data locally. Generated contents are ignored by git by default.

Typical generated paths:

```text
data/datasets_v2/          # CSV datasets for pipeline runs
data/eval_datasets_v2/     # held-out evaluation CSVs and metadata
data/sft_cot_v3.json       # SFT training export generated from runs.db
data/dpo_real_v2.json      # DPO preference export generated from runs.db
data/hf_dataset_v3/        # HuggingFace datasets saved with datasets.save_to_disk
```

Use the commands in the README or docs to regenerate these artifacts for your own experiment.
