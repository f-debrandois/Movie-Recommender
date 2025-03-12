import gradio as gr
from PIL import Image
import requests
import io
import numpy as np
import argparse
import torch
from model import MNISTNet
import matplotlib.pyplot as plt




def recognize_digit(image):
    # Convert to PIL Image necessary if using the API method
    print(image.keys())
    image = image['composite']    
    image = image[:,:,3]
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    response = requests.post("http://model_api:5000/predict", data=img_binary.getvalue())
    predicted_label = response.json()["prediction"]
    return predicted_label

if __name__=='__main__':
    print('HIIIIIIIIIIIIIIIIIIIII')

    gr.Interface(fn=recognize_digit, 
                inputs="sketchpad", 
                outputs='label',
                live=True,
                description="Draw a number on the sketchpad to see the model's prediction.",
                ).launch(debug=True, share=False)