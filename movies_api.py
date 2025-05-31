import argparse
import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
import io
import torchvision
from annoy import AnnoyIndex
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Function to predict the genre of a movie based on the image.
    '''
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))

    # Transform the PIL image
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model_genre(tensor)
        _, predicted = torch.max(outputs, 1)
    label = dict_genres[predicted.item()]
    return jsonify({"prediction": label})

@app.route('/recommender', methods=['POST'])
def recommender():
    '''
    Function to recommend movies based on the embedding using the Annoy index.
    '''
    dict_embedding = request.get_json()
    embedding = dict_embedding["features"]
    # Find in the Annoy index
    indices = annoy_index_poster.get_nns_by_vector(embedding, 5)
    recommendations = movies_path.iloc[indices,0].to_list()
    return jsonify({"recommendations": recommendations})

def plot_recommender(embedding, annoy_index):
    '''
    Function to recommend plots based on the embedding using the specified Annoy index.
    '''
    # Find in the Annoy index
    indices = annoy_index.get_nns_by_vector(embedding, 5)
    recommendations = plots_path.iloc[indices,0].to_list()
    return jsonify({"recommendations": recommendations})

@app.route('/plot_recommender_bow', methods=['POST'])
def plot_recommender_bow():
    '''
    Function to recommend plots based on the embedding using the bag of words Annoy index.
    '''
    dict_embedding = request.get_json()
    embedding = dict_embedding["features"]
    print("Received embedding for plot recommendation:", embedding)
    # Find in the Annoy index
    return plot_recommender(embedding, annoy_index_bow)


@app.route('/plot_recommender_glove', methods=['POST'])
def plot_recommender_glove():
    '''
    Function to recommend plots based on the embedding using the glove Annoy index.
    '''
    dict_embedding = request.get_json()
    embedding = dict_embedding["features"]
    # Find in the Annoy index
    return plot_recommender(embedding, annoy_index_glove)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='weight.pth')
    args = parser.parse_args()
    model_path = args.model_path

    dict_genres = {0: 'Horreur', 1: 'Animation', 2: 'Action', 3: 'Fantasy', 4: 'Science-Fiction', 5: 'Thriller', 6: 'Drama', 7: 'Romance', 8: 'Comedy', 9: 'Documentary'}
    num_classes = len(dict_genres)

    # LOAD THE MODEL
    model_genre = torchvision.models.resnet18()
    model_genre.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
    model_genre.to(device)
    model_genre.load_state_dict(torch.load(model_path, map_location=device))
    model_genre.eval()

    # Transform for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Annoy index for poster recommendations
    annoy_index_poster = AnnoyIndex(512, 'angular')
    annoy_index_poster.load('annoy_index_poster.ann')
    movies_path = pd.read_csv('movie_paths.csv')

    # Load Annoy index for plot recommendations with bag of words
    annoy_index_bow = AnnoyIndex(53363, 'angular')
    annoy_index_bow.load('annoy_index_bag_of_words.ann')
    plots_path = pd.read_csv('plot_titles.csv')

    # # Load Annoy index for plot recommendations wiith glove
    # annoy_index_bow = AnnoyIndex(?, 'angular')
    # annoy_index_bow.load('annoy_index_glove.ann')
    # plots_path = pd.read_csv('plot_titles.csv')

    app.run(host='0.0.0.0', port=5000, debug=True)