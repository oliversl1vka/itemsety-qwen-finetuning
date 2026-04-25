# DEPLOYMENT GUIDE - HuggingFace Space Setup

## Súčasný stav

✅ Pripravené súbory:
- `app.py` - Gradio interface s @spaces.GPU decorator (2-hour duration)
- `run_sft_test.py` - Test training script (50 examples, 12 steps)
- `requirements.txt` - Aktualizované dependencies
- `README_SPACE.md` - Space documentation

## Deployment steps

### 1. Clone HF Space (ak ešte nie je)

```powershell
cd C:\Users\slivk\Desktop\itemsety_real_training
git clone https://huggingface.co/spaces/OliverSlivka/testrun2 hf_space
cd hf_space
```

### 2. Skopíruj súbory

```powershell
# Z main projektu do Space
Copy-Item ..\itemsety\app.py .
Copy-Item ..\itemsety\run_sft_test.py .
Copy-Item ..\itemsety\requirements.txt .
Copy-Item ..\itemsety\README_SPACE.md .\README.md -Force
```

### 3. Commit a push

```powershell
git add app.py run_sft_test.py requirements.txt README.md
git commit -m "feat: Add GPU training with Gradio interface"
git push
```

### 4. Sleduj Space rebuild

Space sa automaticky rebuild-ne a spustí na: https://huggingface.co/spaces/OliverSlivka/testrun2

## Test Run

1. Otvor Space v browse
ri
2. Klikni "Submit"
3. Sleduj logs (streaming output)
4. Čakaj ~10-20 minút (Zero GPU je limitovaný)

## Production Training

Po úspešnom teste vytvor **production script** s:
- Full dataset (439 examples)
- 2-3 epochs
- Push to `OliverSlivka/qwen2.5-3b-itemset-extractor`

Použiť príkaz v Gemini:
```
Create a production training script run_sft_full.py based on run_sft_test.py but:
1. Use full training dataset (439 examples)
2. Train for 3 epochs
3. Use Qwen/Qwen2.5-3B-Instruct model
4. Push to OliverSlivka/qwen2.5-3b-itemset-extractor
5. Save checkpoints every 100 steps
```

## Známe problémy

- ✅ PyTorch CUDA: HF Space má CUDA support
- ✅ bitsandbytes: HF Space má pre-installed
- ✅ Python 3.13: HF Space používa 3.10/3.11
- ⚠️ Zero GPU timeout: Max 2 hours (nastavené v @spaces.GPU decorator)
- ⚠️ Model size: 3B model môže byť too big pre Zero GPU (test s 0.5B najprv)

## Alternatívy

Ak Zero GPU nestačí:
1. **Paid GPU**: Upgrade Space na A10G/A100 ($0.60-1.10/hour)
2. **Colab Pro**: T4 GPU, vlastný notebook
3. **Lokálny server**: Ak máš vlastnú GPU
