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

# Load the model
model_genre.load_state_dict(torch.load(model_path, map_location=device))
model_genre.eval()

# DEFINE THE TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Annoy index for recommendations
annoy_index = AnnoyIndex(512, 'angular')
# Load the Annoy index
annoy_index.load('annoy_index_poster.ann')
# load df path movies
movies_path = pd.read_csv('movie_paths.csv')

@app.route('/predict', methods=['POST'])
def predict():
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
    dict_embedding = request.get_json()
    embedding = dict_embedding["features"]
    # Find in the Annoy index
    indices = annoy_index.get_nns_by_vector(embedding, 5)
    recommendations = movies_path.iloc[indices,0].to_list()
    return jsonify({"recommendations": recommendations})

    
        

# @app.route('/batch_predict', methods=['POST'])
# def batch_predict():
#     # Get the image data from the request
#     images_binary = request.files.getlist("images[]")

#     tensors = []

#     for img_binary in images_binary:
#         img_pil = Image.open(img_binary.stream)
#         tensor = transform(img_pil)
#         tensors.append(tensor)

#     # Stack tensors to form a batch tensor
#     batch_tensor = torch.stack(tensors, dim=0)

#     # Make prediction
#     with torch.no_grad():
#         outputs = model(batch_tensor.to(device))
#         _, predictions = outputs.max(1)

#     return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)