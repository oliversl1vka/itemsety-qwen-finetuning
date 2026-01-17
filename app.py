import gradio as gr
import spaces
import subprocess
import os

@spaces.GPU(duration=7200) # Set a 2-hour duration
def run_training_test():
    command = "python run_sft_test.py"
    
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
        yield output + "\n\nTest training finished successfully!"
    else:
        yield output + f"\n\nTest training failed with return code {return_code}!"

@spaces.GPU(duration=7200) # Set a 2-hour duration
def run_training_full():
    command = "python run_sft_full.py"
    
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
        yield output + "\n\nFull training finished successfully!"
    else:
        yield output + f"\n\nFull training failed with return code {return_code}!"

with gr.Blocks() as demo:
    gr.Markdown("# Model Fine-Tuning with Qwen2.5-3B-Instruct")
    gr.Markdown("Choose between a quick test run or a full training run for your model.")
    
    with gr.Tab("Test Training Run"):
        gr.Markdown("This will run a quick test training on a small subset of your dataset.")
        test_button = gr.Button("Start Test Training")
        test_output = gr.Textbox(lines=30, label="Test Training Log")
        test_button.click(run_training_test, inputs=None, outputs=test_output)
        
    with gr.Tab("Full Training Run"):
        gr.Markdown("This will run the full training process on your entire dataset.")
        full_button = gr.Button("Start Full Training")
        full_output = gr.Textbox(lines=30, label="Full Training Log")
        full_button.click(run_training_full, inputs=None, outputs=full_output)

demo.launch()