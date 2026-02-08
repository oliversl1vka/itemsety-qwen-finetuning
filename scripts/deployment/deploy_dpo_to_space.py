#!/usr/bin/env python3
"""
Deploy DPO training setup to HuggingFace Space.

Updates the Space with:
- DPO-enabled Gradio app
- Updated README
- Requirements with DPO dependencies
- Training scripts (run_dpo_training.py)
"""

import os
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
from dotenv import load_dotenv

# Load environment variables
load_dotenv("hf.env")

# Configuration
SPACE_ID = "OliverSlivka/testrun2"
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    """Deploy DPO setup to HF Space."""
    
    if not HF_TOKEN:
        print("❌ HF_TOKEN not found. Please set it in hf.env")
        return 1
    
    print("=" * 60)
    print("🚀 DEPLOYING DPO TRAINING TO HUGGINGFACE SPACE")
    print("=" * 60)
    print(f"\nSpace: {SPACE_ID}")
    print(f"URL: https://huggingface.co/spaces/{SPACE_ID}\n")
    
    # Files to upload
    operations = [
        # Main app
        CommitOperationAdd(
            path_in_repo="app.py",
            path_or_fileobj="scripts/deployment/app_dpo.py"
        ),
        
        # README
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj="scripts/deployment/hf_space_README.md"
        ),
        
        # Requirements
        CommitOperationAdd(
            path_in_repo="requirements.txt",
            path_or_fileobj="scripts/deployment/hf_space_requirements.txt"
        ),
        
        # Training scripts
        CommitOperationAdd(
            path_in_repo="src/training/run_dpo_training.py",
            path_or_fileobj="src/training/run_dpo_training.py"
        ),
        CommitOperationAdd(
            path_in_repo="src/training/run_sft_full.py",
            path_or_fileobj="src/training/run_sft_full.py"
        ),
        CommitOperationAdd(
            path_in_repo="src/training/run_sft_test.py",
            path_or_fileobj="src/training/run_sft_test.py"
        ),
    ]
    
    print("📦 Files to upload:")
    for op in operations:
        print(f"   - {op.path_in_repo}")
    print()
    
    # Create API client
    api = HfApi(token=HF_TOKEN)
    
    # Upload files
    print("📤 Uploading to HuggingFace Space...")
    try:
        commit_info = api.create_commit(
            repo_id=SPACE_ID,
            repo_type="space",
            operations=operations,
            commit_message="Deploy DPO training setup: app, README, requirements, training scripts"
        )
        
        print(f"✅ Successfully deployed!")
        print(f"\n🔗 Space URL: https://huggingface.co/spaces/{SPACE_ID}")
        print(f"📝 Commit: {commit_info.commit_url}")
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print(f"1. Go to: https://huggingface.co/spaces/{SPACE_ID}")
        print("2. Wait for Space to build (~2-3 minutes)")
        print("3. Set Space secrets (Settings > Variables and secrets):")
        print("   - HF_TOKEN = your_huggingface_token")
        print("4. Select 'DPO' method and 'test' mode")
        print("5. Click 'Submit' to start training")
        print("\n⏱️  Expected time:")
        print("   - DPO Test: ~15-20 minutes")
        print("   - DPO Full: ~60-90 minutes")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
