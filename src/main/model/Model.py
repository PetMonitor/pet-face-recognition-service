import tensorflow as tf

import os
import pickle
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from math import isnan

SIZE = (160,160,3)                               # Size of the input images

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=8, zoom_range=0.1, fill_mode='nearest', channel_shift_range = 0.1)

def apply_transform(images, datagen):
    for x in datagen.flow(images, batch_size=len(images), shuffle=False):
        return x

def define_triplets_batch(filenames, labels, nbof_triplet = 21 * 3):
    # Input:
    #   - filenames: path to directory with classified images
    #   - labels: class labels list
    #   - nbof_triplet: number of triplets required
    # Output:
    #   - triplet_train
    #   - y_triplet
    triplet_train = []
    y_triplet = np.empty(nbof_triplet)
    classes = np.unique(labels)
    for i in range(0, nbof_triplet, 3):
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

def image_generator(filenames, labels, batch_size=63, use_aug=True, datagen=datagen):
    while True:
        f_triplet, y_triplet = define_triplets_batch(filenames, labels, batch_size)
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
        yield (i_triplet, y_triplet)


def define_adaptive_hard_triplets_batch(filenames, labels, predict, nbof_triplet=21*3, use_neg=True, use_pos=True):
    """
    Generates hard triplet for offline selection. It will consider the whole dataset.
    This function will also return the predicted values.
    
    Args:
        -images: images from which the triplets will be created
        -labels: labels of the images
        -predict: predicted embeddings for the images by the trained model
        -alpha: threshold of the triplet loss
    Returns:
        -triplets
        -triplets_labels: labels of the triplets
        -pred_triplets: predicted embeddings of the triplets
    """

    # Check if we have the right number of triplets
    assert nbof_triplet%3 == 0
    
    # return the sorted unique elements of 'labels' and the indices that give the unique values
    _,idx_classes = np.unique(labels, return_index=True)
    classes = labels[np.sort(idx_classes)]
    
    # initialize one array for the triplets, one for the labels, one for the embeddings
    triplets = []
    triplets_labels = np.empty(nbof_triplet)
    pred_triplets = np.empty((nbof_triplet, predict.shape[-1]))
    
    for i in range(0, nbof_triplet, 3):
        # Chooses the first class randomly
        keep = np.equal(labels, classes[np.random.randint(len(classes))])
        keep_filenames = filenames[keep]
        keep_labels = labels[keep]
        
        # Chooses the first image among this class randomly
        # this will be the anchor
        idx_image1 = np.random.randint(len(keep_labels))
        
        if use_pos:
            # Computes the distance between the chosen image and the rest of the class
            # select the one with the biggest distance
            dist_class = np.sum(np.square(predict[keep]-predict[keep][idx_image1]),axis=-1)
            idx_image2 = np.argmax(dist_class)
        else:
            # Select the second image randomly
            idx_image2 = np.random.randint(len(keep_labels))
            j = 0
            while idx_image1 == idx_image2:
                idx_image2 = np.random.randint(len(keep_labels))
                # Just to prevent endless loop:
                j += 1
                if j == 1000:
                    print("[Error: define_hard_triplets_batch] Endless loop.")
                    break
        
        triplets += [keep_filenames[idx_image1]]
        triplets_labels[i] = keep_labels[idx_image1]
        pred_triplets[i] = predict[keep][idx_image1]
        triplets += [keep_filenames[idx_image2]]
        triplets_labels[i+1] = keep_labels[idx_image2]
        pred_triplets[i+1] = predict[keep][idx_image2]
        
        # Exclude the labels from the selected subclass to obtain a reduced subset
        # from which to pick the negative sample.
        not_keep = np.logical_not(keep)
        
        if use_neg:
            # Computes the distance between the anchor and the rest of the other classes
            dist_other = np.sum(np.square(predict[not_keep]-predict[keep][idx_image1]),axis=-1)
            idx_image3 = np.argmin(dist_other) 
        else:
            # Select the negative image randomly
            idx_image3 = np.random.randint(len(filenames[not_keep]))
            
        triplets += [filenames[not_keep][idx_image3]]
        triplets_labels[i+2] = labels[not_keep][idx_image3]
        pred_triplets[i+2] = predict[not_keep][idx_image3]

    # Return the selected triplets (a, p, n), their labels and embeddings
    return np.array(triplets), triplets_labels, pred_triplets


"""
 Hard Triplets: triplets that are harder for the network to train on, because dst(anchor,negative) < dst(anchor,positive)
"""
def online_adaptive_hard_image_generator(filenames,  # Absolute path of the images
    labels,                                          # Labels of the images
    model,                                           # A keras model
    loss,                                            # Current loss of the model
    batch_size      =63,                             # Batch size (has to be a multiple of 3 for dogfacenet)
    nbof_subclasses =10,                             # Number of subclasses from which the triplets will be selected
    use_aug         =True,                           # Use data augmentation
    datagen         =datagen):                       # Data augmentation parameter
    """
    Generator to select online hard triplets for training.
    Include an adaptive control on the number of hard triplets included 
    during the training (it should increase as accuracy improves).
    """
    hard_triplet_ratio = 0
    nbof_hard_triplets = 0
    while True:
        # Select all unique labels (== classes)
        classes = np.unique(labels)

        """
        In order to limit the number of computation for prediction,
        we will not computes nbof_subclasses predictions for the hard triplets generation,
        but int(nbof_subclasses*hard_triplet_ratio)+2, which means that the higher the
        accuracy is the more prediction are going to be computed.
        """ 

        # subclasses <= select nbof_subclasses*hard_triplet_ratio + 2 unique classes from 'classes' array
        subclasses = np.random.choice(classes, size=int(nbof_subclasses*hard_triplet_ratio) + 2, replace=False)
        
        # Return an array of booleans (e.g: [True True False])
        # with a True in each position where a label has matched
        # one of the subclasses:    
        keep_classes = np.equal(labels, subclasses[0])
        for i in range(1, len(subclasses)):
            keep_classes = np.logical_or(keep_classes, np.equal(labels, subclasses[i]))

        #Select filenames and labels for
        #subclasses
        subfilenames = filenames[keep_classes]
        sublabels = labels[keep_classes]
        predict = model.predict(predict_generator(subfilenames, 32), steps=int(np.ceil(len(subfilenames)/32)))
        
        # Select triplets (a, p, n) using the largest embedding distance predicted by the model to choose n and p (HARD)
        f_triplet_hard, y_triplet_hard, predict_hard = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, nbof_hard_triplets*3, use_neg=True, use_pos=True)
        # Select triplets (a, p, n) randomly (SOFT)
        f_triplet_soft, y_triplet_soft, predict_soft = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, batch_size-nbof_hard_triplets*3, use_neg=False, use_pos=False)

        f_triplet = np.append(f_triplet_hard, f_triplet_soft)
        y_triplet = np.append(y_triplet_hard, y_triplet_soft)

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
            
        
        yield (i_triplet, y_triplet)


