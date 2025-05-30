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
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='app_utils/weight.pth')
args = parser.parse_args()
model_path = args.model_path


def recognize_genre(image):
    '''
    Function to recognize the genre of a movie poster using a pre-trained model.
    '''
    if image is None:
        return "No image"
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    response = requests.post("http://model_api:5000/predict", data=img_binary.getvalue())
    predicted_label = response.json()["prediction"]
    return predicted_label

def recommender_model(image):
    '''
    Function to load the recommender model and define the transformation for the input image.
    '''
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
    '''
    Function to recommend movies based on the input image.
    '''
    if image is None:
        return "No image"
        
    model_reco, transform = recommender_model(image)
    image = Image.fromarray(image.astype('uint8'))

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

def plot_recommender(plot, embedding_method):
    '''
    Function to recommend movies based on the input plot.
    '''
    if plot is None:
        return "No plot"
    
    if embedding_method == "Bag of words":
        # load vectorizer
        vectorizer = joblib.load('app_utils/vectorizer_bow.pkl')
        # Transform the plot using the vectorizer
        plot_vectorized = vectorizer.transform([plot]).toarray().tolist()[0]
        response = requests.post("http://model_api:5000/plot_recommender_bow", json={"features" : plot_vectorized})

    # elif embedding_method == "glove":
    #     # load glove model

    #     response = requests.post("http://model_api:5000/plot_recommender_glove", json={"features" : plot})

    movies_recommended = response.json()["recommendations"]
    return gr.update(value="\n".join(movies_recommended), visible=True) if movies_recommended else gr.update(value="No recommendations found", visible=True)


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

        with gr.Tab("Classification"):
            image_input = gr.Image(type="numpy")

            action_choice = gr.Radio(
                choices=["Prédire le genre", "Recommander des films"],
                label="Choisissez une action",
                value="Prédire le genre"
            )

            genre_output = gr.Textbox(label="Résultat", lines=1, visible=False)
            reco_gallery = gr.Gallery(label="Films recommandés", visible=False, columns=3, elem_classes="small-image")

            run_button = gr.Button("Exécuter")
            run_button.click(fn=handle_action, inputs=[image_input, action_choice], outputs=[genre_output, reco_gallery])

        with gr.Tab("Synopsis"):
            gr.Markdown("### Recommander des films en fonction du synopsis")
            plot_input = gr.Textbox(label="Entrez le synopsis du film", placeholder="Tapez le synopsis ici...", lines=5)

            action_choice_plot = gr.Radio(
                choices=["Bag of words", "Glove"],
                label="Choisissez la méthode d'embedding",
                value="Bag of words"
            )

            plot_reco_output = gr.Textbox(label="Résultat", lines=5, visible=False)

            run_button = gr.Button("Exécuter")
            run_button.click(fn=plot_recommender, inputs=[plot_input, action_choice_plot], outputs=plot_reco_output)

        

    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True, share=True)