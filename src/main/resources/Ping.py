from flask_restful import Resource
from pymongo import MongoClient

client = MongoClient()
db = client.pet_monitor_db

class Ping(Resource):

    def __init__(self):
        super(Ping, self).__init__()

    def get():
        return "Uneasy lies the head that wears the crown"