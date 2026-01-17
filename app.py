import gradio as gr
import spaces
import subprocess
import os

# This script assumes it's running in a Hugging Face Space environment
# where the dependencies are installed from requirements.txt.
# The venv activation is not needed if the script is launched
# by the space's runtime with the correct python interpreter.

@spaces.GPU(duration=7200) # Set a 2-hour duration
def run_training():
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
        yield output + "\n\nTraining finished successfully!"
    else:
        yield output + f"\n\nTraining failed with return code {return_code}!"

demo = gr.Interface(
    fn=run_training, 
    inputs=None, 
    outputs=gr.Textbox(lines=30, label="Training Log"), 
    title="Model Fine-Tuning",
    description="Click the 'Submit' button to start the fine-tuning process. The logs will be streamed below."
)
demo.launch()

