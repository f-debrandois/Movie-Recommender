import gradio as gr
from PIL import Image
import requests
import io
import numpy as np
import argparse
import matplotlib.pyplot as plt




def recognize_genre(image):
    if image == None:
        return "No image"
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    response = requests.post("http://model_api:5000/predict", data=img_binary.getvalue())
    predicted_label = response.json()["prediction"]
    return predicted_label

if __name__=='__main__':


    gr.Interface(fn=recognize_genre, 
                inputs="image",
                outputs='label',
                live=True,
                description="Insert an image to see the model's prediction.",
                ).launch(debug=True, share=True, server_name="0.0.0.0", server_port=7860)