import gradio as gr
import spaces
import subprocess
import os

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


def push_model_to_hub():
    """Manually push trained model to HuggingFace Hub"""
    
    yield "🚀 Starting manual push to HuggingFace Hub...\n\n"
    
    process = subprocess.Popen(
        "python push_model.py",
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
        yield output + "\n\n✅ Model pushed successfully!"
    else:
        yield output + f"\n\n❌ Push failed with return code {return_code}"


# Create Gradio interface with Blocks for multiple functions
with gr.Blocks(title="🚀 Qwen2.5 Fine-Tuning") as demo:
    gr.Markdown("""
    # 🚀 Qwen2.5 Fine-Tuning for Itemset Extraction
    
    Fine-tune Qwen2.5 on the [itemset-extraction-v2](https://huggingface.co/datasets/OliverSlivka/itemset-extraction-v2) dataset.
    """)
    
    with gr.Tab("🎯 Training"):
        training_mode = gr.Radio(
            choices=["test", "full"],
            value="test",
            label="Training Mode",
            info="Test: 50 examples (~15 min). Full: 439 examples (~2 hours)"
        )
        train_btn = gr.Button("▶️ Start Training", variant="primary")
        train_output = gr.Textbox(lines=25, label="Training Log", show_copy_button=True)
        train_btn.click(run_training, inputs=training_mode, outputs=train_output)
    
    with gr.Tab("⬆️ Push Model"):
        gr.Markdown("""
        ### Manual Push to HuggingFace Hub
        
        Use this if training completed but the model wasn't pushed automatically.
        Make sure your `HF_TOKEN` secret has **WRITE** permissions!
        """)
        push_btn = gr.Button("⬆️ Push Model to Hub", variant="secondary")
        push_output = gr.Textbox(lines=20, label="Push Log", show_copy_button=True)
        push_btn.click(push_model_to_hub, outputs=push_output)
    
    gr.Markdown("""
    ## Output Models
    - **Test**: `OliverSlivka/qwen2.5-3b-itemset-test`
    - **Full**: `OliverSlivka/qwen2.5-3b-itemset-extractor`
    """)

if __name__ == "__main__":
    demo.launch()
