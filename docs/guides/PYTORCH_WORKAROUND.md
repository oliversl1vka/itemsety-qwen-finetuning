# PyTorch Training Workaround for Python 3.14

## Problem

Python 3.14 is too new - PyTorch doesn't have an official release yet. This blocks:
- Model training (`run_sft_full.py`)
- Model evaluation (`eval_finetuned_model.py`)
- Model deployment (requires model to exist)

## Solution Options

### Option 1: Google Colab (Recommended) ✅

**Pros:** Free GPU, pre-installed PyTorch, no setup required  
**Cons:** Session limits, requires internet

**Steps:**
1. Upload `scripts/colab/sft_trl_lora_qlora.ipynb` to Google Colab
2. Upload your training data:
   - `data/training_v2/all_training_examples.json`
   - Or upload entire `data/hf_dataset_v2/` folder
3. Run the notebook cells in order
4. Download the fine-tuned model
5. Upload to HuggingFace Hub using the notebook's final cell

**Colab Notebook:** `scripts/colab/sft_trl_lora_qlora.ipynb`

---

### Option 2: Docker Container 🐳

**Pros:** Consistent environment, local GPU support  
**Cons:** Requires Docker installation, more complex setup

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version for testing, use GPU version in production)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy project files
COPY . .

# Run training
CMD ["python3", "src/training/run_sft_full.py"]
```

**Build and run:**
```bash
# Build image
docker build -t itemset-trainer .

# Run training
docker run -v $(pwd)/data:/app/data -v $(pwd)/runs.db:/app/runs.db itemset-trainer

# For GPU support, add --gpus all flag
docker run --gpus all -v $(pwd)/data:/app/data itemset-trainer
```

---

### Option 3: Python 3.11 Virtual Environment (Local)

**Pros:** Full local control, no cloud dependency  
**Cons:** Requires managing multiple Python versions

**macOS Setup:**
```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Create virtual environment
python3.11 -m venv .venv-py311

# Activate
source .venv-py311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (M1/M2 Mac)
pip install torch torchvision torchaudio

# Run training
python src/training/run_sft_full.py
```

**Linux Setup:**
```bash
# Install Python 3.11 via pyenv
pyenv install 3.11.7
pyenv local 3.11.7

# Create virtual environment
python -m venv .venv-py311
source .venv-py311/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# Run training
python src/training/run_sft_full.py
```

---

### Option 4: Conda Environment (Alternative)

**Pros:** Handles Python version + dependencies automatically  
**Cons:** Larger disk footprint, slower installation

**Setup:**
```bash
# Install Miniconda (if not already installed)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Create environment with Python 3.11
conda create -n itemset-py311 python=3.11

# Activate
conda activate itemset-py311

# Install dependencies
pip install -r requirements.txt

# Install PyTorch via conda (recommended)
conda install pytorch torchvision torchaudio -c pytorch

# Run training
python src/training/run_sft_full.py
```

---

## Comparison Matrix

| Option | Setup Time | Cost | GPU Support | Persistence | Best For |
|--------|-----------|------|-------------|-------------|----------|
| **Google Colab** | 5 min | Free | ✅ Yes (T4) | Session-based | Quick experiments |
| **Docker** | 15 min | Free | ✅ Yes (local) | Permanent | CI/CD, production |
| **Venv (Py3.11)** | 10 min | Free | ✅ Yes (local) | Permanent | Local development |
| **Conda** | 20 min | Free | ✅ Yes (local) | Permanent | Multi-environment |

---

## Recommended Workflow

### Phase 1: Development (Use Python 3.14)
```bash
# Dataset generation, pipeline, export - all work on Python 3.14
python3 .github/agents/orchestrator.py run \
  --workflow full-training \
  --skip-upload \
  --skip-training \
  --skip-eval \
  --llm-model gpt-4.1-mini
```

This completes:
1. ✅ Dataset generation (500 datasets)
2. ✅ Pipeline execution (Apriori + LLM)
3. ✅ Export training data
4. ✅ Create HF dataset
5. ✅ Visualization

### Phase 2: Training (Switch to Python 3.11 or Colab)
```bash
# Upload data to Colab OR switch to Python 3.11 environment
# Then run:
python3 src/training/run_sft_full.py
```

This completes:
6. ✅ Model training (40-60 min on GPU)

### Phase 3: Evaluation (Python 3.11 or Colab)
```bash
python3 src/evaluation/eval_finetuned_model.py \
  --model-path OliverSlivka/qwen2.5-3b-itemset-extractor \
  --count 50
```

This completes:
7. ✅ Model evaluation

### Phase 4: Deployment (Can return to Python 3.14)
```bash
# Deployment scripts don't need PyTorch
./scripts/deployment/deploy_to_hf_space.ps1
```

---

## Quick Start: Google Colab (5 Minutes)

1. **Open Colab:** https://colab.research.google.com/
2. **Upload notebook:** `scripts/colab/sft_trl_lora_qlora.ipynb`
3. **Upload data files:**
   ```python
   # In Colab cell:
   from google.colab import files
   uploaded = files.upload()  # Select data/hf_dataset_v2/
   ```
4. **Run all cells** (Runtime → Run all)
5. **Wait 40-60 minutes** (free T4 GPU)
6. **Download model** or push directly to HuggingFace Hub

---

## Troubleshooting

### Error: "No module named 'torch'"
- **Cause:** PyTorch not installed or wrong Python version
- **Fix:** Use Option 1 (Colab) or Option 3 (Python 3.11 venv)

### Error: "CUDA out of memory"
- **Cause:** GPU memory insufficient
- **Fix:** Reduce `per_device_train_batch_size` to 1, enable `gradient_checkpointing`

### Error: "Illegal instruction (core dumped)"
- **Cause:** PyTorch built for wrong CPU architecture
- **Fix:** Reinstall PyTorch: `pip install --force-reinstall torch`

---

## When Will Python 3.14 Support Be Available?

**PyTorch Roadmap:**
- Python 3.14 released: October 2024
- PyTorch support typically lags: 3-6 months
- **Expected:** March-June 2025

**Current Workaround Timeline:**
- Use Python 3.11 until PyTorch 3.14 support arrives
- Monitor: https://github.com/pytorch/pytorch/issues

---

## Conclusion

**Recommended Approach:**
1. **Develop on Python 3.14** (dataset generation, pipeline, export all work)
2. **Train on Google Colab** (free GPU, zero setup)
3. **Deploy from Python 3.14** (no PyTorch needed for deployment scripts)

This gives you the best of both worlds: cutting-edge Python for development, stable environment for training.

---

**Last Updated:** 2026-02-03  
**Python 3.14 Status:** PyTorch unavailable  
**Recommended:** Google Colab or Python 3.11 venv  
**Timeline:** 3-6 months until PyTorch 3.14 support
