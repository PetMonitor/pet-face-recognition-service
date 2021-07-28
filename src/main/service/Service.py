from flask import Flask, jsonify, redirect, request
import json
from src.main.utils import Utils
from src.main.drive import FileManager, Authenticator
import tensorflow.keras.models as models

DOGS_MODEL_PATH = "src/main/model/dogs/dog_facenet.h5"
#TODO: change this to fetch a file with the *.h5 extension
#dogs_model = models.load_model("src/main/model/dogs/3.7.2021.h5", custom_objects={'triplet':Utils.triplet,'triplet_acc':Utils.triplet_acc})

#TODO: add caching for embeddings cause they take some time to process

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Uneasy lies the head that wears the crown"

"""
Expects a base64 encoded image of a cat's face.
Returns a 128 dimension embedding that represents that image.
"""
@app.route("/cats/embedding", methods=['POST'])
def get_cat_embedding():
    return ""


"""
Expects a base64 encoded image of a dog's face.
Expects the body to be a json with the following shape:
{
    "dogs": {
        "images" : [ base64encodedPhoto1, ..., base64encodedPhotoN ]
    }
}
Returns a 128 dimension embedding for each of the provided images.
"""
@app.route("/dogs/embedding", methods=['POST'])
def get_dog_embedding():
    dogs_model = models.load_model(DOGS_MODEL_PATH, custom_objects={'triplet':Utils.triplet,'triplet_acc':Utils.triplet_acc})
    
    images = Utils.process_and_decode_base64_images(request.json['dogs']['images'])
    prediction = dogs_model.predict(images)
    print("Predicted embedding for dog image: {}".format(prediction))
    return jsonify(prediction.tolist())

def download_models():
    # get first file id for file with extension .h5
    modelFileIds = FileManager.get_file_ids_for_file_extension('.h5')
    if len(modelFileIds) == 0:
        raise Exception("Unable to find files with extension .h5")
    
    # For the time being, access the one and only element in modelFileIds
    # which should be the dogs model
    # TODO: adapt this to also download the model for cats
    FileManager.download_file(DOGS_MODEL_PATH, modelFileIds[0])
    



if __name__ == '__main__':
    download_models()
    app.run(debug=True, host='0.0.0.0')