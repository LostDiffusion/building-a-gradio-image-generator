# Trying to integrate with the Hugging face library
import gradio as gr
import numpy as np
import io
from PIL import Image
import requests

API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.content

image_bytes = query({
	"inputs": "Astronaut riding a horse",
})

app_title = "<center><h1>StableDiffusion Demo</h1></center>"
app_description = "<center>A basic demo of a diffusion model. And Gradio.</model>"
prompt_text = "Describe the image you want generated. SD and SDXL models should use danboru style tagging, Flux should use descriptive text."
start_image_prompt = "A beautiful cat girl standing on a beach."
hf_token_text = "Hugging Face Token"
hf_token_placeholder_text = "Not needed unless you want to use Hugging Face Pro features or access gated repos, such as Flux.1 [dev]."
submit_btn_text = "Generate"

def sepia(input_img):
    sepia_filter = np.array([
        [0.393,0.769, 0.189],
        [0.349,0.686, 0.168],
        [0.272,0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    # return gr.Image(type="filepath", shape=...)
    return sepia_img

def inference(*args):
    return "image"

with gr.Blocks() as demo:
    gr.Markdown(app_title)    
    gr.Markdown(app_description)
    prompt = gr.Textbox(label=prompt_text, value=start_image_prompt, lines=3, max_lines=20)
    hf_token = gr.Textbox(label=hf_token_text, placeholder=hf_token_placeholder_text, type="password")
    with gr.Group():
        with gr.Row():
            upload = gr.Image()
            editor = gr.ImageEditor()
        with gr.Row():
            mask = gr.ImageMask()

    with gr.Group():
        with gr.Row():
            submit_btn = gr.Button(value="Submit",variant="primary", size=None)
            clear_btn = gr.Button(value="Clear",variant ="secondary", size=None)

    with gr.Row() as output_row:
        generated_image = gr.Image()

    submit_btn.click(
        fn=query,
        inputs=[prompt, hf_token],
        outputs=[generated_image],
        generated_image = Image.open(io.BytesIO(image_bytes))
    )

demo.launch()

#gr.Slider(value=2, minimum=1, maximum=10, step=1)
#gr.load("models/black-forest-labs/FLUX.1-schnell")
#https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev
#https://huggingface.co/spaces/yanze/PuLID-FLUX
#theme='NoCrypt/miku'
