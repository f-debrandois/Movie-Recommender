import gradio as gr
from PIL import Image
import requests
import io
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='weight.pth')
args = parser.parse_args()
model_path = args.model_path


def recognize_genre(image):
    if image is None:
        return "No image"
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    response = requests.post("http://model_api:5000/predict", data=img_binary.getvalue())
    predicted_label = response.json()["prediction"]
    return predicted_label

def recommender_model(image):
    model_genre = torchvision.models.resnet18()
    model_genre.fc = torch.nn.Linear(in_features=512, out_features=10)
    model_genre.load_state_dict(torch.load(model_path, map_location=device))
    model_reco = torch.nn.Sequential(model_genre.conv1, model_genre.bn1, model_genre.relu, model_genre.maxpool, 
                            model_genre.layer1, model_genre.layer2, model_genre.layer3, model_genre.layer4,
                            model_genre.avgpool, torch.nn.Flatten())
    model_reco.to(device)
    model_reco.eval()

    # DEFINE THE TRANSFORM
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return model_reco, transform

def recommender(image):
    # afficher ls dans le dossier
    import os
    files = os.listdir('.')
    print("Files in the current directory:", files)

    if image is None:
        return "No image"
        
    model_reco, transform = recommender_model(image)
    image = Image.fromarray(image.astype('uint8'))
    # img_binary = io.BytesIO()
    # img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(image).to(device)
    tensor = tensor.unsqueeze(0)

    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model_reco(tensor)

    # Send the embeddings to the recommender API
    response = requests.post("http://model_api:5000/recommender", json={"features" : outputs[0].detach().cpu().numpy().tolist()})
    predicted_label = response.json()["recommendations"]
    if not predicted_label:
        return "No recommendations found"
    return predicted_label

def handle_action(image, choice):
    if choice == "Prédire le genre":
        genre = recognize_genre(image)
        # return f"Genre prédit : {genre}"
        return gr.update(value=genre, visible=True), gr.update(value=[], visible=False)
    elif choice == "Recommander des films":
        films = recommender(image)
        # return "\n".join(f"- {film}" for film in films)
        return gr.update(value="", visible=False), gr.update(value=films, visible=True)

if __name__=='__main__':


    # gr.Interface(fn=recognize_genre, 
    #             inputs="image",
    #             outputs='label',
    #             live=True,
    #             description="Insert an image to see the model's prediction.",
    #             ).launch(debug=True, share=True, server_name="0.0.0.0", server_port=7860)


    with gr.Blocks() as demo:
        gr.HTML("""
        <style>
            .small-image img {
                width: 250px !important;
                height: auto !important;
            }
        </style>
        """)

        gr.Markdown("## 🎥 Analyse de Poster de Film")

        with gr.Row():
            image_input = gr.Image(type="numpy")

            action_choice = gr.Radio(
                choices=["Prédire le genre", "Recommander des films"],
                label="Choisissez une action",
                value="Prédire le genre"
            )

        genre_output = gr.Textbox(label="Résultat", lines=5, visible=False)
        reco_gallery = gr.Gallery(label="Films recommandés", visible=False, columns=3, elem_classes="small-image")

        run_button = gr.Button("Exécuter")
        run_button.click(fn=handle_action, inputs=[image_input, action_choice], outputs=[genre_output, reco_gallery])

        

    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)