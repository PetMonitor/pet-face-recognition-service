from flask_restful import Resource
from pymongo import MongoClient

client = MongoClient()
db = client.pet_monitor_db

class CatsEmbeddingGenerator(Resource):

    def __init__(self):
        super(CatsEmbeddingGenerator, self).__init__()


    """
    Expects a list of base64 encoded images of cat's faces.
    Expects the body to be a json with the following shape:
    Expects the body to be a json with the following shape:
    {
        "cats": {
            "images" : [
                { 
                    photoId: uuid1,
                    photo: base64encodedPhoto1, 
                },
                ..., 
                { 
                    photoId: uuidN,
                    photo: base64encodedPhotoN, 
                }, 
            ]
        }
    }
    """
    def post():
        raise NotImplementedError()