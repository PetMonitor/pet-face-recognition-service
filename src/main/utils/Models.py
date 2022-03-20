import tensorflow

from os.path import exists

from src.main.utils import Constants
from src.main.drive import FileManager
from src.main.utils import Constants
from src.main.utils import Utils


global dogsModel


##################### Initial service setup #####################

def download_models():
    if exists(Constants.DOGS_FACENET_MODEL_PATH):
        return
    # get first file id for file with extension .h5
    modelFileId = FileManager.get_file_id_for_filename(Constants.DOGS_FACENET_MODEL_FILENAME)
    FileManager.download_file(Constants.DOGS_FACENET_MODEL_PATH, modelFileId)


download_models()

print("Loading model {}...".format(str(Constants.DOGS_FACENET_MODEL_PATH)))
dogsModel = tensorflow.keras.models.load_model(Constants.DOGS_FACENET_MODEL_PATH, custom_objects={'triplet':Utils.triplet,'triplet_acc':Utils.triplet_acc})
print("Model loaded")