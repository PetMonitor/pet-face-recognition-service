import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("/content/drive/My Drive/4.4.2021.h5",custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})

IMG_PATH = "/content/drive/My Drive/DOG_DATASET/data/val/86/86.0.jpg"
prediction = model.predict(load_images([IMG_PATH]))

print("Embedding size {}".format(len(prediction[0])))
print("Predicted embedding {}".format(prediction))