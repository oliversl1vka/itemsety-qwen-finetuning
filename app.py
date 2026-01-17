import gradio as gr
import spaces
import subprocess

# Note: @spaces.GPU removed - using persistent paid GPU instead
def run_training(training_mode):
    """
    Run training with GPU support via @spaces.GPU decorator.
    
    Args:
        training_mode: "test" for quick 50-example test, "full" for production 439-example training
    """
    
    # Upgrade libraries
    upgrade_command = "pip install --upgrade torch transformers trl peft accelerate bitsandbytes"
    yield f"🚀 Upgrading libraries...\n{upgrade_command}\n\n"
    
    process_upgrade = subprocess.Popen(
        upgrade_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )
    
    output_upgrade = ""
    for line in iter(process_upgrade.stdout.readline, ''):
        output_upgrade += line
        yield output_upgrade
    
    process_upgrade.stdout.close()
    process_upgrade.wait()
    
    yield output_upgrade + "✅ Libraries upgraded.\n\n"

    if training_mode == "test":
        command = "python run_sft_test.py"
        description = "🧪 TEST RUN: 50 examples, Qwen2.5-3B (4-bit LoRA)"
    else:
        command = "python run_sft_full.py"
        description = "🚀 PRODUCTION: 439 examples, 3 epochs, Qwen2.5-3B"
    
    yield f"{description}\n\n{'='*60}\n\n"
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=True,
    )
    
    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        yield output
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code == 0:
        yield output + "\n\n" + "="*60 + "\n✅ Training finished successfully!\n" + "="*60
    else:
        yield output + "\n\n" + "="*60 + f"\n❌ Training failed with return code {return_code}!\n" + "="*60

# Create Gradio interface
demo = gr.Interface(
    fn=run_training,
    inputs=gr.Radio(
        choices=["test", "full"],
        value="test",
        label="Training Mode",
        info="Test: Quick validation (50 examples). Full: Production training (439 examples, ~30-60 min)"
    ),
    outputs=gr.Textbox(
        lines=30,
        label="Training Log",
        show_copy_button=True
    ),
    title="🚀 Qwen2.5 Fine-Tuning for Itemset Extraction",
    description="""
    Fine-tune Qwen2.5 on the [itemset-extraction-v2](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2) dataset.
    
    **Test Mode**: Qwen2.5-3B, 50 examples, ~10-15 minutes  
    **Full Mode**: Qwen2.5-3B, 439 examples, ~40-60 minutes (3 epochs)
    
    ⚠️ **Zero GPU Limit**: 2 hours max. Both modes use 4-bit quantization to fit in Zero GPU memory.
    """,
    article="""
    ## Output Models
    
    - **Test**: `OliverSlivka/qwen2.5-3b-itemset-test` (pushed to Hub for validation)
    - **Full**: `OliverSlivka/qwen2.5-3b-itemset-extractor` (final production model)
    
    ## Dataset
    
    Training data: [itemset-extraction-v2](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2)  
    Project: [itemsety_real_training](https://github.com/OliverSlivka/itemsety)
    """
)

if __name__ == "__main__":
    demo.launch()
