"""
Delete the qwen-itemsety-qlora model from Hugging Face Hub.
This model was our first fine-tuning experiment and is no longer needed.
"""

from huggingface_hub import HfApi, list_repo_refs
import os

# Initialize API
api = HfApi()

# Model to delete
model_id = "OliverSlivka/qwen-itemsety-qlora"

print("="*80)
print("DELETE MODEL FROM HUGGING FACE HUB")
print("="*80)
print(f"\nModel to delete: {model_id}")

# Check if model exists
try:
    refs = list_repo_refs(model_id)
    print(f"✅ Model found on Hub")
    print(f"   Branches: {[ref.name for ref in refs.branches]}")
    
    # Confirm deletion
    print("\n⚠️  WARNING: This will permanently delete the model from Hub!")
    print("   You can still keep local files in ~/.cache/huggingface/")
    
    response = input("\nType 'DELETE' to confirm: ")
    
    if response.strip() == "DELETE":
        print("\n🗑️  Deleting repository...")
        api.delete_repo(repo_id=model_id, repo_type="model")
        print(f"✅ Successfully deleted {model_id}")
        
        print("\n📝 Next steps:")
        print("   - Local cache files remain in ~/.cache/huggingface/")
        print("   - Future fine-tuning will use new model names")
        print("   - Test results preserved in model_test_results/")
    else:
        print("\n❌ Deletion cancelled")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPossible reasons:")
    print("   - Model doesn't exist")
    print("   - Authentication required (set HF_TOKEN env var)")
    print("   - No permission to delete")
