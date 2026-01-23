# Test Production Model on Google Colab
# 
# INSTRUCTIONS:
# 1. Go to: https://colab.research.google.com
# 2. Create new notebook
# 3. Runtime → Change runtime type → T4 GPU
# 4. Copy and run each cell below

# === CELL 1: Install dependencies ===
# !pip install -q transformers accelerate bitsandbytes peft torch

# === CELL 2: Load model ===
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_MODEL = "OliverSlivka/qwen2.5-3b-itemset-extractor"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading base model (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
print("✓ Model ready!")
"""

# === CELL 3: Test inference ===
"""
TEST_PROMPT = '''You are an expert data mining assistant.

TASK: Extract frequent itemsets from this dataset.

DATASET:
Row 1: Bread, Milk, Eggs
Row 2: Bread, Butter
Row 3: Milk, Eggs, Butter
Row 4: Bread, Milk, Eggs, Butter
Row 5: Bread, Milk

MINIMUM SUPPORT: 2

Return a JSON array with itemset, support, and rows for each frequent itemset.'''

messages = [{"role": "user", "content": TEST_PROMPT}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=True,
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\\n" + "="*50)
print("MODEL OUTPUT:")
print("="*50)
print(response)

# Validate JSON
import re
json_match = re.search(r'\\[.*\\]', response, re.DOTALL)
if json_match:
    try:
        parsed = json.loads(json_match.group())
        print("\\n✓ Valid JSON! Found", len(parsed), "itemsets")
    except:
        print("\\n✗ JSON parse error")
else:
    print("\\n✗ No JSON found")
"""
