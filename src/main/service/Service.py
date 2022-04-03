from flask import Flask
from flask_restful import Api

from os import environ

from src.main.resources.Ping import Ping
from src.main.resources.Dogs import DogEmbeddingGenerator
from src.main.resources.Cats import CatsEmbeddingGenerator

BASE_MODEL_PATH = "/app/src/main/model"

#TODO: add caching for embeddings cause they take some time to process

app = Flask(__name__)
api = Api(app, prefix='/api/v0')

##################### Application routes #####################

api.add_resource(Ping, "/", methods=['GET'])

api.add_resource(CatsEmbeddingGenerator, "/cats/embedding", methods=['POST'])
api.add_resource(DogEmbeddingGenerator, "/dogs/embedding", methods=['POST'])

##################### Run app #####################

#TODO: first request takes about 1 minute (presumably because model hasn't been loaded yet).
# Figure out how to reduce this time. Maybe load model somewhere else. Maybe call it on startup, before
# any requests are made to force it to load.
if __name__ == '__main__':
    port = environ['LOCAL_PORT'] if (environ.get('PORT') is None) else environ['PORT']
    app.run(debug=True, host='0.0.0.0', port=port)
