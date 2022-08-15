from flask_restful import Resource

class Ping(Resource):

    def __init__(self):
        super(Ping, self).__init__()

    def get():
        return "Uneasy lies the head that wears the crown"