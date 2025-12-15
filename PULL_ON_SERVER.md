# Aktualizácia Repository na Serveri

## Stav
✅ Notebook `qwen_finetuning_server.ipynb` bol aktualizovaný a pushnutý na GitHub (commit c9874b2)

## Príkazy na Serveri

```bash
# 1. Prejdi do repository
cd ~/itemsety-qwen-finetuning

# 2. Pullni najnovšie zmeny
git pull origin main

# 3. Overenie - notebook má byť 23KB
ls -lh qwen_finetuning_server.ipynb

# 4. Overenie obsahu - má obsahovať 26 buniek
grep -c "VSCode.Cell" qwen_finetuning_server.ipynb
# Očakávaný výstup: 26

# 5. Otvor notebook v Jupyter
# V Jupyter interface naviguj na qwen_finetuning_server.ipynb
```

## Čo Notebook Obsahuje (26 Buniek)

1. **Header** - Popis projektu a warnings
2. **System Check** - CPU/RAM/GPU info
3. **Install Packages** - `pip install bitsandbytes transformers trl peft...`
4. **Dataset Path** - Nastavenie `DATASET_PATH = "./hf_dataset_enhanced"`
5. **Load Dataset** - `load_from_disk()`
6. **Inspect Data** - Preview prvého príkladu
7. **Load Model (4-bit)** - BitsAndBytesConfig + AutoModelForCausalLM
8. **Configure LoRA** - prepare_model_for_kbit_training() + get_peft_model()
9. **Training Args** - bf16=True, optim="paged_adamw_8bit"
10. **Initialize Trainer** - SFTTrainer setup
11. **Start Training** - trainer.train()
12. **Save Model** - trainer.save_model()
13. **Test Inference** - Generate on validation example
14. **Memory Plot** - Visualize RAM/GPU usage

## Spustenie Tréningu

```python
# Postupne spúšťaj bunky (Shift+Enter) od bunky 1 po 26
# Kritické bunky:
# - Bunka 3: System check (overí GPU)
# - Bunka 5: Install packages (zaberie ~2 min)
# - Bunka 7: Load dataset (overí hf_dataset_enhanced/)
# - Bunka 11: Load model 4-bit (~1 min download)
# - Bunka 13: Configure LoRA (~10 sec)
# - Bunka 19: START TRAINING (~30-60 min)
```

## Troubleshooting

### Ak notebook stále vyzerá prázdny:
```bash
# Hard refresh v Jupyter:
# Ctrl+Shift+R (alebo Cmd+Shift+R na Mac)

# Alebo reštartuj Jupyter kernel:
# Kernel → Restart & Clear Output
```

### Overenie Git pull:
```bash
git log --oneline -5
# Mali by si vidieť:
# c9874b2 fix: Update qwen_finetuning_server.ipynb with complete 4-bit QLoRA cells
# e318178 docs: Add comprehensive server upload and training guide
# ee4d41d feat: Add enhanced dataset with real-world context and 4-bit QLoRA configuration
```

### Ak pull zlyháva:
```bash
# Zruš lokálne zmeny (ak si robil nejaké)
git reset --hard origin/main

# Alebo fresh clone:
cd ~
rm -rf itemsety-qwen-finetuning
git clone https://<YOUR_PAT>@github.com/oliversl1vka/itemsety-qwen-finetuning.git
```

## Potvrdenie Úspechu

Po `git pull` by si mal vidieť:
```
Updating e318178..c9874b2
Fast-forward
 qwen_finetuning_server.ipynb | 680 ++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 679 insertions(+), 1 deletion(-)
```

A notebook by mal mať:
- **26 buniek** (markdown + code)
- **23KB veľkosť**
- **Kompletný obsah** (od System Check po Memory Plot)
