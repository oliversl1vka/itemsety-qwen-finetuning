#!/usr/bin/env python3
"""
Manual push script - run this in your HF Space to push the trained model.
Add this file to your Space and run it separately.

Usage in Space terminal or add a button in app.py:
    python push_model.py
"""

import os
import sys
from pathlib import Path

def push_trained_model():
    """Push the trained model to HuggingFace Hub"""
    
    print("="*60)
    print("🚀 Manual Model Push to HuggingFace Hub")
    print("="*60)
    
    # Check HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment!")
        print("   Set it in Space secrets: https://huggingface.co/spaces/OliverSlivka/testrun2/settings")
        return False
    
    print(f"✓ HF_TOKEN found (length: {len(hf_token)} chars)")
    
    # Check for trained model
    output_dir = "OliverSlivka/qwen2.5-3b-itemset-extractor"
    local_output = Path("./output") if Path("./output").exists() else Path(output_dir)
    
    # Look for checkpoint directories
    checkpoint_dirs = list(Path(".").glob("**/checkpoint-*"))
    adapter_files = list(Path(".").glob("**/adapter_config.json"))
    
    print(f"\n📁 Searching for trained model...")
    print(f"   Checkpoint dirs found: {len(checkpoint_dirs)}")
    print(f"   Adapter configs found: {len(adapter_files)}")
    
    for cp in checkpoint_dirs[:5]:
        print(f"   - {cp}")
    for af in adapter_files[:5]:
        print(f"   - {af}")
    
    if not adapter_files and not checkpoint_dirs:
        print("\n❌ No trained model found!")
        print("   The model may have been cleared from memory.")
        print("   You need to run training again.")
        return False
    
    # Import required libraries
    print("\n📦 Loading libraries...")
    try:
        from huggingface_hub import HfApi, login
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        print(f"❌ Missing library: {e}")
        print("   Run: pip install huggingface_hub peft transformers torch")
        return False
    
    # Login with token
    print("\n🔑 Logging in to HuggingFace...")
    try:
        login(token=hf_token)
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        print("   Check that your token has WRITE permissions!")
        return False
    
    # Find the best checkpoint or final adapter
    adapter_path = None
    
    # Priority: final adapter > latest checkpoint
    if adapter_files:
        # Use the one NOT in a checkpoint folder (final)
        for af in adapter_files:
            if "checkpoint" not in str(af.parent):
                adapter_path = af.parent
                break
        if not adapter_path:
            adapter_path = adapter_files[-1].parent  # Use latest checkpoint
    elif checkpoint_dirs:
        # Use latest checkpoint
        checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[-1]) if x.name.split("-")[-1].isdigit() else 0)
        adapter_path = checkpoint_dirs[-1]
    
    print(f"\n📂 Using adapter from: {adapter_path}")
    
    # List files to be uploaded
    print("\n📄 Files to upload:")
    for f in adapter_path.iterdir():
        size = f.stat().st_size if f.is_file() else 0
        print(f"   {f.name}: {size/1024:.1f} KB")
    
    # Push to Hub
    target_repo = "OliverSlivka/qwen2.5-3b-itemset-extractor"
    print(f"\n⬆️ Pushing to {target_repo}...")
    
    try:
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=target_repo,
            repo_type="model",
        )
        print(f"\n✅ SUCCESS! Model pushed to:")
        print(f"   https://huggingface.co/{target_repo}")
        return True
    except Exception as e:
        print(f"\n❌ Push failed: {e}")
        print("\n   Possible causes:")
        print("   1. Token doesn't have WRITE permission")
        print("   2. You don't have write access to this repo")
        print("   3. Network error")
        return False


def list_all_files():
    """Debug: List all files in current directory"""
    print("\n📁 All files in Space:")
    for root, dirs, files in os.walk(".", topdown=True):
        # Skip hidden and large dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', '.git']]
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:20]:  # Limit files shown
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath)
                print(f"{subindent}{file}: {size/1024:.1f} KB")
            except:
                print(f"{subindent}{file}")
        if len(files) > 20:
            print(f"{subindent}... and {len(files)-20} more files")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_all_files()
    else:
        list_all_files()  # Always show files first
        print("\n" + "="*60 + "\n")
        success = push_trained_model()
        sys.exit(0 if success else 1)
