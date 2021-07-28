import numpy as np 
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.io import decode_base64
from tensorflow.nn import relu
from tensorflow.image import decode_image, resize, ResizeMethod

ALPHA = 0.3
SIZE = (160,160,3)

URL_SAFE_REPLACEMENTS = { "/": "_", "+": "-"}


def triplet(y_true, y_pred):
	# Triplet loss function.
	# Returns the quantity to be minimized during training.
    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    negative = y_pred[2::3]
 
 	# Distance between anchor/positive and anchor/negative
    dst_anchor_positive = K.sum(K.square(anchor - positive), -1)
    dst_anchor_negative = K.sum(K.square(anchor - negative), -1)

    # minimize distance between anchor and positive embeddings
    # pow(||f(a) - f(p)||,2) + ALPHA <= pow(||f(a) - f(n)||,2)
    # return max(0, dst(ap) - dst(an) + margin)
    return K.sum(relu(dst_anchor_positive - dst_anchor_negative + ALPHA))

def triplet_acc(y_true, y_pred):
    anchor = y_pred[0::3]
    positive = y_pred[1::3]
    negative = y_pred[2::3]
    
    ap = K.sum(K.square(anchor - positive), -1)
    an = K.sum(K.square(anchor - negative), -1)
    
    return K.less(ap + ALPHA, an)


def load_images(filenames):
    h,w,c = SIZE
    images = np.empty((len(filenames), h, w, c))
    for i, f in enumerate(filenames):
      images[i] = np.array(image.load_img(f, target_size = SIZE))/255.0
    return images

def transform_to_url_safe(img_str):
    url_safe_base64_img = img_str
    try:
        for unsafe_c, safe_c in URL_SAFE_REPLACEMENTS.items():
            url_safe_base64_img = url_safe_base64_img.replace(unsafe_c, safe_c)
    except Exception as ex:
        print("Error converting base64 encoded img to url safe format: {}".format(ex))

    if url_safe_base64_img:
        return url_safe_base64_img
    return ""

def process_and_decode_base64_images(image_strings, shape=SIZE):
    h,w,c = SIZE
    images = np.empty((len(image_strings), h, w, c))
    #try:
    for i, img_str in enumerate(image_strings):
        img = decode_base64(transform_to_url_safe(img_str))
        img = decode_image(img, channels=3)
        img = resize(img, [shape[0],shape[1]], method=ResizeMethod.BILINEAR)
        images[i] = img
    return images
    #except Exception as ex:
    #    print("Error loading base64 image: {}".format(ex))
    return np.empty(0)

def predict_generator(filenames, batch_size=32):
    for i in range(0,len(filenames),batch_size):
        images_batch = load_images(filenames[i:i+batch_size])
        yield images_batch

def save_history(histories, filename):
    # Save history (metrics obtained during training)
    loss = np.empty(0)
    val_loss = np.empty(0)
    acc = np.empty(0)
    val_acc = np.empty(0)

    for history in histories:
        loss = np.append(loss, history.history['loss'])
        val_loss = np.append(val_loss, history.history['val_loss'])
        acc = np.append(acc, history.history['triplet_acc'])
        val_acc = np.append(val_acc, history.history['val_triplet_acc'])

    history_ = np.array([loss, val_loss, acc, val_acc])
    np.save(filename, history_)


    