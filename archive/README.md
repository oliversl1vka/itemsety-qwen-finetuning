# Archive (`archive/`)

Historical files kept for reference. Not actively maintained.

## Structure

```
archive/
├── legacy_scripts/    # Old Python script versions
├── experiments/       # Test results and experiments
└── resources/         # Reference materials and research
```

## Contents

### legacy_scripts/
Superseded script versions:
- `dataset_generation.py` → Replaced by `generate_datasets_v2.py`
- `train_qwen_sft.py` → Replaced by `run_sft_full.py`
- `generate_eval_datasets.py` → Replaced by `generate_eval_datasets_v2.py`
- Various presentation and enhancement scripts

### experiments/
Historical experiment outputs:
- `model_test_results/` - Early model inference tests
- `hf_space_testrun2/` - HF Space test files
- `dataset_analysis_report.json` - Analysis results

### resources/
Research materials and references:
- Agent documentation examples
- External code references
- Tutorial notebooks

## Note

These files are preserved for historical context but should not be used in production. Check `src/` for current implementations.
