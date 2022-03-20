import sys, os
from flask import jsonify
from flask_restful import request, Resource

import numpy as np
from src.main.utils import Utils
import src.main.utils.Models as Models
#from sklearn.preprocessing import Normalizer

MAX_THREADS = 10
TASK_TIMEOUT_SECONDS = 20
IMG_SIZE = (160,160)

class DogEmbeddingGenerator(Resource):

    def __init__(self):
        super(DogEmbeddingGenerator, self).__init__()

    """
    Expects a list of base64 encoded images of dog's faces.
    Expects the body to be a json with the following shape:
    {
        "dogs": {
            [
                { 
                    uuid: uuid1,
                    photo: base64encodedPhoto1, 
                },
                ..., 
                { 
                    uuid: uuidN,
                    photo: base64encodedPhotoN, 
                }, 
            ]
        }
    }
    """
    def post(self):
        try:
            images = request.json['dogs']
            response = {}
            for img in images:
                response[img["uuid"]] = self.predictAndSaveEmbedding(img)

            return jsonify({ "embeddings": response })
            
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("ERROR\n")
            print(exc_type, fname, exc_tb.tb_lineno)
            return {}

    def predictAndSaveEmbedding(self, image):
        try:
            print("Process and decode base 64", flush=True)
            img = Utils.process_and_decode_base64_image(image["photo"])

            img_arr = np.ndarray((1, IMG_SIZE[0], IMG_SIZE[1], 3))
            #TODO: image should be pre-processed



            img_arr[0] = img
            print("Processed and decoded images {}".format(img_arr.shape), flush=True)

            # scale RGB values to interval [0,1]
            face_pixels = (img_arr[0] / 255.).astype('float32')
            scaled_img = np.expand_dims(face_pixels, axis=0)
            embedding =  Models.dogsModel.predict(scaled_img)[0]

            #TODO: embedding should be normalized
            #in_encoder = Normalizer(norm='l2')
            #embedding = in_encoder.transform(embedding)
            return embedding.tolist()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("ERROR\n")
            print(exc_type, fname, exc_tb.tb_lineno)
            return { "exception": str(e) }