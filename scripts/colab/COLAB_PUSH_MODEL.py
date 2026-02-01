"""
Manual push script for LoRA adapter to HuggingFace Hub.
Run this in Google Colab AFTER training is complete in your HF Space.

This script:
1. Loads the trained model from HF Space output
2. Saves the LoRA adapter
3. Pushes to HuggingFace Hub

INSTRUCTIONS:
1. Open Google Colab
2. Select T4 GPU runtime
3. Run each cell in order
"""

# ===== CELL 1: Install dependencies =====
# !pip install -q transformers accelerate peft huggingface_hub

# ===== CELL 2: Login to HuggingFace =====
"""
from huggingface_hub import login, HfApi

# Login - this will prompt for your token
# Get token from: https://huggingface.co/settings/tokens
# Make sure it has WRITE permissions!
login()

# Verify login
api = HfApi()
print(f"✓ Logged in as: {api.whoami()['name']}")
"""

# ===== CELL 3: Check if model exists and create adapter files =====
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
import os

# Check what's in the repo
from huggingface_hub import list_repo_files

repo_id = "OliverSlivka/qwen2.5-3b-itemset-extractor"
try:
    files = list_repo_files(repo_id)
    print(f"Files in {repo_id}:")
    for f in files:
        print(f"  - {f}")
except Exception as e:
    print(f"Error: {e}")
"""

# ===== CELL 4: Copy adapter from TEST model to PRODUCTION =====
"""
from huggingface_hub import HfApi, upload_folder, snapshot_download
import os
import shutil

# Download the TEST adapter (which works)
test_model_id = "OliverSlivka/qwen2.5-3b-itemset-test"
prod_model_id = "OliverSlivka/qwen2.5-3b-itemset-extractor"

print(f"Downloading adapter from {test_model_id}...")
local_dir = snapshot_download(
    repo_id=test_model_id,
    local_dir="./test_adapter",
    allow_patterns=["*.json", "*.safetensors", "*.bin"]
)
print(f"Downloaded to: {local_dir}")

# List downloaded files
print("\\nDownloaded files:")
for root, dirs, files in os.walk(local_dir):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size/1024:.1f} KB")

# Push to production repo
api = HfApi()
print(f"\\nPushing to {prod_model_id}...")

# Upload all files
api.upload_folder(
    folder_path=local_dir,
    repo_id=prod_model_id,
    repo_type="model",
)

print(f"✓ Adapter pushed to: https://huggingface.co/{prod_model_id}")
"""

# ===== CELL 5: Verify upload =====
"""
from huggingface_hub import list_repo_files

repo_id = "OliverSlivka/qwen2.5-3b-itemset-extractor"
files = list_repo_files(repo_id)
print(f"\\nFiles in {repo_id} after upload:")
for f in files:
    print(f"  - {f}")

# Check if adapter_config.json exists
if "adapter_config.json" in files:
    print("\\n✓ adapter_config.json found - model is ready to use!")
else:
    print("\\n✗ adapter_config.json NOT found - something went wrong")
"""

# ===== CELL 6: Test loading the production model =====
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading production model...")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, "OliverSlivka/qwen2.5-3b-itemset-extractor")
print("✓ Production model loaded successfully!")

# Quick test
prompt = "Extract frequent itemsets from: Row 1: A,B | Row 2: A,C | Row 3: A,B,C. Min support: 2. Return JSON."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("\\nGenerating response...")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\\nModel output:")
print(response)
"""
