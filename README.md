# Movie-Recommender


## lancer les dockers
### build et execute
sudo docker-compose up --build
### stop the docker
sudo docker-compose stop
sudo docker-compose down
### clean
sudo docker builder prune

## logs d'un docker
sudo docker-compose logs -f model_api


### forcer l'arrêt des dockers
sudo systemctl restart docker.socket docker.service

## Accéder à l'API
