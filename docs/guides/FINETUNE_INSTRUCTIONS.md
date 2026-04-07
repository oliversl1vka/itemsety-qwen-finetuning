# Fine-Tuning Qwen2.5-3B s HuggingFace Skills + Gemini CLI

## Stav projektu

✅ **Hotové:**
1. Dataset nahratý na HuggingFace Hub:
   - **v3 (current):** https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v3
   - **v2 (frozen):** https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2
   - 439 tréningových príkladov
   - 49 validačných príkladov
   - 2 konfigurácie: `default` (čistý JSON) a `chain_of_thought` (s reasoning)

2. Gemini CLI nainštalované s HuggingFace Skills extension

3. HuggingFace autentifikácia hotová (user: OliverSlivka)

## Spustenie Fine-Tuningu

### Krok 1: Otvor nový PowerShell terminál

### Krok 2: Nastav HF_TOKEN
```powershell
$env:HF_TOKEN = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
```

### Krok 3: Spusti Gemini CLI
```powershell
gemini
```

### Krok 4: V Gemini CLI zadaj tento prompt:

```
Fine-tune Qwen/Qwen2.5-3B-Instruct on OliverSlivka/itemset-extraction-v3 dataset for instruction following using SFT.

Dataset format:
- Has "messages" column with ChatML format (system/user/assistant)
- Train split: 439 examples
- Validation split: 49 examples

Configuration:
- Use LoRA for efficient training
- 2-3 epochs
- Push model to OliverSlivka/qwen2.5-3b-itemset-extractor

Do a quick test run on 50 examples first to verify everything works.
```

### Krok 5: Review a potvrď konfiguráciu

Gemini ti ukáže:
- Hardware (pravdepodobne a10g-small alebo a10g-large)
- Estimated cost (~$5-15)
- Estimated time

### Krok 6: Po úspešnom teste spusti full training

```
Run full SFT training on the complete dataset with the same configuration.
```

## Alternatíva: Manuálny Fine-Tuning Script

Ak Gemini CLI nefunguje, môžeme použiť priamy HuggingFace training script.

## Očakávané výsledky

- Model: `OliverSlivka/qwen2.5-3b-itemset-extractor`
- Baseline (V1 s 0.5B): 6.7% valid JSON
- Cieľ (V2 s 3B): 80-90% valid JSON

## Troubleshooting

Ak Gemini CLI nefunguje s HF Skills:
1. Skontroluj či je extension nainštalovaná: `gemini extensions list`
2. Re-inštaluj: `gemini extensions install https://github.com/huggingface/skills.git --consent`
3. Overte HF token: `.\.venv\Scripts\huggingface-cli.exe whoami`
