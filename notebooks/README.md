# Notebooks (`notebooks/`)

Jupyter notebooks for experiments and server-based training.

## Contents

| Notebook | Description |
|----------|-------------|
| `qwen_finetuning_server.ipynb` | Server-based fine-tuning workflow |
| `sft_trl_lora_qlora.ipynb` | SFT training with TRL library |
| `grpo_rnj_1_instruct.ipynb` | GRPO training experiments |
| `TRL_SFT_Nemotron_3_Nano_30B_A3B_A100.ipynb` | Nemotron training reference |

## Usage

```bash
# Start Jupyter
jupyter notebook

# Or JupyterLab
jupyter lab
```

## Notes

- Notebooks are for experimentation and one-off runs
- Production training should use `src/training/run_sft_full.py`
- Some notebooks may have outdated paths after repo reorganization
