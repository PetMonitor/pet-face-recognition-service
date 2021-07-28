#!pip install Keras
#!pip install Pillow
#!pip install tensorflow

from src.main.utils import Utils
from src.main.model import Model


from google.colab import drive
drive.mount('/content/drive')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Load dataset to begin training model
TEST_SPLIT = 0.1                                 # Train/test ratio

START_EPOCH = 0                                  # Start the training at a specified epoch
NBOF_EPOCHS = 100                                # Number of epoch to train the network (original value = 250)

NET_NAME = "2021.03.28.dogfacenet"               # Network saved name
HIGH_LEVEL = True                                # Use high level training ('fit' keras method)
STEPS_PER_EPOCH = 300                            # Number of steps per epoch
VALIDATION_STEPS = 30                            # Number of steps per validation

PATH = "/content/drive/My Drive/DOG_DATASET/data_preprocessed/train/" 	# Path to the directory of the saved dataset
PATH_SAVE = "/content/drive/My Drive/DOG_DATASET/"    					# Path to the directory where the history will be stored
MODEL_PATH = "/content/drive/My Drive/DOG_DATASET/model/"    			# Path to the directory where the model will be stored
PRETRAINED_MODEL_PATH = "/content/drive/My Drive/facenet_keras.h5"


filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root, dirs, files in os.walk(PATH):
    if len(files) > 1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames, files)
        labels = np.append(labels, np.ones(len(files))*idx)
        idx += 1

print('Done.')

print('Total number of imported pictures: {:d}'.format(len(labels)))

nbof_classes = len(np.unique(labels))
print('Total number of classes: {:d}'.format(nbof_classes))


# Split the dataset into test and train
nbof_test = int(TEST_SPLIT*nbof_classes)
keep_test = np.less(labels,nbof_test)
keep_train = np.logical_not(keep_test)

filenames_test = filenames[keep_test]
labels_test = labels[keep_test]
filenames_train = filenames[keep_train]
labels_train = labels[keep_train]

#################################################################################################

def train_model():
	# Loads a previously trained facenet model, unfreezes the last layer
	# and re-trains it using pet face data set.
    histories = []
    current_loss = 0.6
    current_accuracy = 0
    batch_size = 3*10
    nbof_subclasses = 40

    # Load pretrained facenet model
    model = tf.keras.models.load_model(PRETRAINED_MODEL_PATH, custom_objects = {'triplet':triplet, 'triplet_acc':triplet_acc})

    # Unfreeze last layer for training
    model.layers[-1].trainable = True

    # Define 'triplet' as loss function, 'triplet_acc' as accuracy function
    # and set 'adam' optimizer.
    model.compile(loss = triplet, optimizer = 'adam', metrics = [triplet_acc])

    # Bug fixed: keras models are to be initialized by a training on a single batch
    for images_batch, labels_batch in online_adaptive_hard_image_generator(filenames_train, labels_train, model, current_accuracy, batch_size, nbof_subclasses=nbof_subclasses):
        h = model.train_on_batch(images_batch, labels_batch)
        break

    for i in range(START_EPOCH, START_EPOCH + NBOF_EPOCHS):
        print("Beginning epoch number: {:d}".format(i))

        hard_triplet_ratio = np.exp(-crt_loss * 10 / batch_size)
        nbof_hard_triplets = int(batch_size//3 * hard_triplet_ratio)
        
        print("Current hard triplet ratio: " + str(hard_triplet_ratio))
        
        hard_image_gen = online_adaptive_hard_image_generator(filenames_train,labels_train,model,crt_loss,batch_size,nbof_subclasses=nbof_subclasses)
        validation_data_gen = image_generator(filenames_test, labels_test, batch_size, use_aug=False)
        histories+=[model.fit(hard_image_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=1, validation_data=validation_data_gen, validation_steps=VALIDATION_STEPS)]
        
        crt_loss=histories[-1].history['loss'][0]
        crt_acc=histories[-1].history['triplet_acc'][0]

        save_history(histories, '{:s}{:s}.{:d}.npy'.format(PATH_SAVE, NET_NAME ,i))


    # Save newly trained model    
    model.save('{:s}{:s}.h5'.format(MODEL_PATH, NET_NAME))


train_model()