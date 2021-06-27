!pip install Keras
!pip install Pillow
!pip install tensorflow


from google.colab import drive
drive.mount('/content/drive')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import pickle
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from math import isnan

SIZE=(160,160,3)
ALPHA=0.3

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=8, zoom_range=0.1, fill_mode='nearest', channel_shift_range = 0.1)

def apply_transform(images, datagen):
    for x in datagen.flow(images, batch_size=len(images), shuffle=False):
        return x

def define_triplets_batch(filenames,labels,nbof_triplet = 21 * 3):
    triplet_train = []
    y_triplet = np.empty(nbof_triplet)
    classes = np.unique(labels)
    for i in range(0,nbof_triplet,3):
        # Pick a class and chose two pictures from this class
        classAP = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classAP)
        keep_classAP = filenames[keep]
        keep_classAP_idx = labels[keep]
        idx_image1 = np.random.randint(len(keep_classAP))
        idx_image2 = np.random.randint(len(keep_classAP))
        while idx_image1 == idx_image2:
            idx_image2 = np.random.randint(len(keep_classAP))

        triplet_train += [keep_classAP[idx_image1]]
        triplet_train += [keep_classAP[idx_image2]]
        y_triplet[i] = keep_classAP_idx[idx_image1]
        y_triplet[i+1] = keep_classAP_idx[idx_image2]
        # Pick a class for the negative picture
        classN = classes[np.random.randint(len(classes))]
        while classN==classAP:
            classN = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classN)
        keep_classN = filenames[keep]
        keep_classN_idx = labels[keep]
        idx_image3 = np.random.randint(len(keep_classN))
        triplet_train += [keep_classN[idx_image3]]
        y_triplet[i+2] = keep_classN_idx[idx_image3]
        
    return triplet_train, y_triplet

def define_hard_triplets_batch(filenames,labels,predict,nbof_triplet=21*3, use_neg=True, use_pos=True):
    # Check if we have the right number of triplets
    assert nbof_triplet%3 == 0
    
    _,idx_classes = np.unique(labels,return_index=True)
    classes = labels[np.sort(idx_classes)]
    
    triplets = []
    y_triplets = np.empty(nbof_triplet)
    
    for i in range(0,nbof_triplet,3):
        # Chooses the first class randomly
        keep = np.equal(labels,classes[np.random.randint(len(classes))])
        keep_filenames = filenames[keep]
        keep_labels = labels[keep]
        
        # Chooses the first image among this class randomly
        idx_image1 = np.random.randint(len(keep_labels))
        
        
        # Computes the distance between the chosen image and the rest of the class
        if use_pos:
            dist_class = np.sum(np.square(predict[keep]-predict[keep][idx_image1]),axis=-1)

            idx_image2 = np.argmax(dist_class)
        else:
            idx_image2 = np.random.randint(len(keep_labels))
            i = 0
            while idx_image1==idx_image2:
                idx_image2 = np.random.randint(len(keep_labels))
                # Just to prevent endless loop:
                i += 1
                if i == 1000:
                    print("[Error: define_hard_triplets_batch] Endless loop.")
                    break
        
        triplets += [keep_filenames[idx_image1]]
        y_triplets[i] = keep_labels[idx_image1]
        triplets += [keep_filenames[idx_image2]]
        y_triplets[i+1] = keep_labels[idx_image2]
        
        
        # Computes the distance between the chosen image and the rest of the other classes
        not_keep = np.logical_not(keep)
        
        if use_neg:
            dist_other = np.sum(np.square(predict[not_keep]-predict[keep][idx_image1]),axis=-1)
            idx_image3 = np.argmin(dist_other) 
        else:
            idx_image3 = np.random.randint(len(filenames[not_keep]))
            
        triplets += [filenames[not_keep][idx_image3]]
        y_triplets[i+2] = labels[not_keep][idx_image3]

    #return triplets, y_triplets
    return np.array(triplets), y_triplets

def define_adaptive_hard_triplets_batch(filenames,labels,predict,nbof_triplet=21*3, use_neg=True, use_pos=True):
    # Check if we have the right number of triplets
    assert nbof_triplet%3 == 0
    
    _,idx_classes = np.unique(labels,return_index=True)
    classes = labels[np.sort(idx_classes)]
    
    triplets = []
    y_triplets = np.empty(nbof_triplet)
    pred_triplets = np.empty((nbof_triplet,predict.shape[-1]))
    
    for i in range(0,nbof_triplet,3):
        # Chooses the first class randomly
        keep = np.equal(labels,classes[np.random.randint(len(classes))])
        keep_filenames = filenames[keep]
        keep_labels = labels[keep]
        
        # Chooses the first image among this class randomly
        idx_image1 = np.random.randint(len(keep_labels))
        
        # Computes the distance between the chosen image and the rest of the class
        if use_pos:
            dist_class = np.sum(np.square(predict[keep]-predict[keep][idx_image1]),axis=-1)

            idx_image2 = np.argmax(dist_class)
        else:
            idx_image2 = np.random.randint(len(keep_labels))
            j = 0
            while idx_image1==idx_image2:
                idx_image2 = np.random.randint(len(keep_labels))
                # Just to prevent endless loop:
                j += 1
                if j == 1000:
                    print("[Error: define_hard_triplets_batch] Endless loop.")
                    break
        
        triplets += [keep_filenames[idx_image1]]
        y_triplets[i] = keep_labels[idx_image1]
        pred_triplets[i] = predict[keep][idx_image1]
        triplets += [keep_filenames[idx_image2]]
        y_triplets[i+1] = keep_labels[idx_image2]
        pred_triplets[i+1] = predict[keep][idx_image2]
        
        # Computes the distance between the chosen image and the rest of the other classes
        not_keep = np.logical_not(keep)
        
        if use_neg:
            dist_other = np.sum(np.square(predict[not_keep]-predict[keep][idx_image1]),axis=-1)
            idx_image3 = np.argmin(dist_other) 
        else:
            idx_image3 = np.random.randint(len(filenames[not_keep]))
            
        triplets += [filenames[not_keep][idx_image3]]
        y_triplets[i+2] = labels[not_keep][idx_image3]
        pred_triplets[i+2] = predict[not_keep][idx_image3]

    return np.array(triplets), y_triplets, pred_triplets

def load_images(filenames):
    h,w,c=SIZE
    images=np.empty((len(filenames),h,w,c))
    for i,f in enumerate(filenames):
      images[i]=np.array(image.load_img(f, target_size=SIZE))/255.0
      #images[i]=skimage.io.imread(f)/255.0
    return images

def image_generator(filenames, labels, batch_size=63, use_aug=True, datagen=datagen):
    while True:
        f_triplet, y_triplet = define_triplets_batch(filenames, labels, batch_size)
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
        yield (i_triplet, y_triplet)

def hard_image_generator(filenames, labels, predict, batch_size=63, use_neg=True, use_pos=True, use_aug=True, datagen=datagen):
    while True:
        f_triplet, y_triplet = define_hard_triplets_batch(filenames, labels, predict, batch_size, use_neg=use_neg, use_pos=use_pos)
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
        yield (i_triplet, y_triplet)

def predict_generator(filenames, batch_size=32):
    for i in range(0,len(filenames),batch_size):
        images_batch = load_images(filenames[i:i+batch_size])
        yield images_batch

def online_hard_image_generator(filenames, labels, model, batch_size=63, nbof_subclasses=10, use_neg=True, use_pos=True, use_aug=True, datagen=datagen):
    while True:
        # Select a certain amount of subclasses
        classes = np.unique(labels)
        subclasses = np.random.choice(classes,size=nbof_subclasses,replace=False)
        
        keep_classes = np.equal(labels,subclasses[0])
        for i in range(1,len(subclasses)):
            keep_classes = np.logical_or(keep_classes,np.equal(labels,subclasses[i]))
        subfilenames = filenames[keep_classes]
        sublabels = labels[keep_classes]
        predict = model.predict_generator(predict_generator(subfilenames, 32),
                                          steps=np.ceil(len(subfilenames)/32))
        
        f_triplet, y_triplet = define_hard_triplets_batch(subfilenames, sublabels, predict, batch_size, use_neg=use_neg, use_pos=use_pos)
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
        yield (i_triplet, y_triplet)

def online_adaptive_hard_image_generator(filenames, labels, model, loss, batch_size=63, nbof_subclasses =10, use_aug=True,datagen=datagen):
    hard_triplet_ratio = 0
    nbof_hard_triplets = 0
    while True:
        # Select a certain amount of subclasses
        classes = np.unique(labels)
        # In order to limit the number of computation for prediction,
        # we will not computes nbof_subclasses predictions for the hard triplets generation,
        # but int(nbof_subclasses*hard_triplet_ratio)+2, which means that the higher the
        # accuracy is the more prediction are going to be computed.
        subclasses = np.random.choice(classes,size=int(nbof_subclasses*hard_triplet_ratio)+2,replace=False)
        
        keep_classes = np.equal(labels,subclasses[0])
        for i in range(1,len(subclasses)):
            keep_classes = np.logical_or(keep_classes,np.equal(labels,subclasses[i]))
        subfilenames = filenames[keep_classes]
        sublabels = labels[keep_classes]
        predict = model.predict_generator(predict_generator(subfilenames, 32), steps=int(np.ceil(len(subfilenames)/32)))
        
        f_triplet_hard, y_triplet_hard, predict_hard = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, nbof_hard_triplets*3, use_neg=True, use_pos=True)
        f_triplet_soft, y_triplet_soft, predict_soft = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, batch_size-nbof_hard_triplets*3, use_neg=False, use_pos=False)

        f_triplet = np.append(f_triplet_hard,f_triplet_soft)
        y_triplet = np.append(y_triplet_hard,y_triplet_soft)

        predict = np.append(predict_hard, predict_soft, axis=0)
        
        # Proportion of hard triplets in the generated batch
        #hard_triplet_ratio = max(0,1.2/(1+np.exp(-10*acc+5.3))-0.19)
        hard_triplet_ratio = np.exp(-loss * 10 / batch_size)

        if isnan(hard_triplet_ratio):
            hard_triplet_ratio = 0
        nbof_hard_triplets = int(batch_size//3 * hard_triplet_ratio)
        
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
            
        # Potential modif for different losses: re-labels the dataset from 0 to nbof_subclasses
        # dict_subclass = {subclasses[i]:i for i in range(nbof_subclasses)}
        # ridx_y_triplet = [dict_subclass[y_triplet[i]] for i in range(len(y_triplet))]
        
        yield (i_triplet, y_triplet)


def triplet(y_true,y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
 
    ap = K.sum(K.square(a-p), -1)
    an = K.sum(K.square(a-n), -1)

    return K.sum(tf.nn.relu(ap - an + ALPHA))

def triplet_acc(y_true,y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p), -1)
    an = K.sum(K.square(a-n), -1)
    
    return K.less(ap + ALPHA, an)







 PATH="/content/drive/My Drive/DOG_DATASET/data_preprocessed/train/" # Path to the directory of the saved dataset
PATH_SAVE="/content/drive/My Drive/DOG_DATASET/"    # Path to the directory where the history will be stored
PATH_MODEL="/content/drive/My Drive/DOG_DATASET/model/"    # Path to the directory where the model will be stored
PRETRAINED_MODEL_PATH="/content/drive/My Drive/facenet_keras.h5"

SIZE=(160,160,3)                               # Size of the input images
TEST_SPLIT = 0.1                                       # Train/test ratio

LOAD_NET=False                                     # Load a network from a saved model? If True NET_NAME and START_EPOCH have to be precised
NET_NAME="2021.03.28.dogfacenet"                   # Network saved name
START_EPOCH=0                                         # Start the training at a specified epoch
NBOF_EPOCHS=100                                     # Number of epoch to train the network (original value = 250)
HIGH_LEVEL=True                                      # Use high level training ('fit' keras method)
STEPS_PER_EPOCH=300                                   # Number of steps per epoch
VALIDATION_STEPS=30                                   # Number of steps per validation

#----------------------------------------------------------------------------


filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(PATH):
    if len(files)>1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames,files)
        labels = np.append(labels,np.ones(len(files))*idx)
        idx += 1

print('Done.')

print('Total number of imported pictures: {:d}'.format(len(labels)))

nbof_classes = len(np.unique(labels))
print('Total number of classes: {:d}'.format(nbof_classes))

#----------------------------------------------------------------------------
# Split the dataset.

nbof_test = int(TEST_SPLIT*nbof_classes)

keep_test = np.less(labels,nbof_test)
keep_train = np.logical_not(keep_test)

filenames_test = filenames[keep_test]
labels_test = labels[keep_test]

filenames_train = filenames[keep_train]
labels_train = labels[keep_train]

#----------------------------------------------------------------------------


def train_model():

    histories=[]
    crt_loss=0.6
    crt_acc=0
    batch_size=3*10
    nbof_subclasses=40

    # Load model
    model=tf.keras.models.load_model(PRETRAINED_MODEL_PATH,custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})

    model.layers[-1].trainable=True

    model.compile(loss=triplet, optimizer='adam', metrics=[triplet_acc])

    # Bug fixed: keras models are to be initialized by a training on a single batch
    for images_batch,labels_batch in online_adaptive_hard_image_generator(filenames_train, labels_train, model, crt_acc, batch_size, nbof_subclasses=nbof_subclasses):
        h = model.train_on_batch(images_batch,labels_batch)
        break
    for i in range(START_EPOCH,START_EPOCH+NBOF_EPOCHS):
        print("Beginning epoch number: "+str(i))

        hard_triplet_ratio = np.exp(-crt_loss * 10 / batch_size)
        nbof_hard_triplets = int(batch_size//3 * hard_triplet_ratio)
        
        print("Current hard triplet ratio: " + str(hard_triplet_ratio))
        histories+=[model.fit(online_adaptive_hard_image_generator(filenames_train,labels_train,model,crt_loss,batch_size,nbof_subclasses=nbof_subclasses),steps_per_epoch=STEPS_PER_EPOCH, epochs=1, validation_data=image_generator(filenames_test,labels_test,batch_size,use_aug=False),validation_steps=VALIDATION_STEPS)]
        crt_loss=histories[-1].history['loss'][0]
        crt_acc=histories[-1].history['triplet_acc'][0]
        
        # Save model
        #model.save('{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,i))

        # Save history
        loss = np.empty(0)
        val_loss = np.empty(0)
        acc = np.empty(0)
        val_acc = np.empty(0)

        for history in histories:
            loss = np.append(loss,history.history['loss'])
            val_loss = np.append(val_loss,history.history['val_loss'])
            acc = np.append(acc,history.history['triplet_acc'])
            val_acc = np.append(val_acc,history.history['val_triplet_acc'])

        history_ = np.array([loss,val_loss,acc,val_acc])
        np.save('{:s}{:s}.{:d}.npy'.format(PATH_SAVE,NET_NAME,i),history_)
    model.save("/content/drive/My Drive/4.4.2021.h5")

train_model()