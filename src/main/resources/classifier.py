
from calendar import c
from flask_restful import Resource
from pymongo import MongoClient

client = MongoClient()
db = client.pet_monitor_db

class PetClassifier(Resource):

    def __init__(self):
        super(PetClassifier, self).__init__()

    def post(self):
        return []