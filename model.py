from PIL import Image
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import tensorflow as tf
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, AlphaDropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, cohen_kappa_score
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# This script generates the training set (the one you will be provided with),
# and the held out set (the one we will use to test your model towards the leaderboard).
folder = '/home/ubuntu/Deep-Learning/mid-term/train'
imgs = []
labels = []
dictionary = {'red blood cell' : 0,'ring': 1,'schizont': 2,'trophozoite': 3}
for file in os.scandir(folder):
    if file.name.endswith('png'):
        img = cv2.imread(os.path.join(folder, file.name))
        resized_img = cv2.resize(img,(100,100))
        imgs.append(list(resized_img.flatten()))
        txt = file.name[:-4]+'.txt'
        a = open(os.path.join(folder, txt))
        label = a.read()
        labels.append(dictionary[label])
        a.close()
x = np.array(imgs)
y = np.array(labels).transpose()
y = to_categorical(y, num_classes=4)


# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_NEURONS = (100, 200, 100)
N_EPOCHS =1
BATCH_SIZE = 500
DROPOUT = 0.5
# %% -------------------------------------- Data Prep ------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
#y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)

# %% -------------------------------------- Training Prep ----------------------------------------------------------

model = Sequential([  # The dropout is placed right after the outputs of the hidden layers.
    Dense(N_NEURONS[0], input_dim=x.shape[1], kernel_initializer=weight_init),  # This sets some of these outputs to 0, so that
    Activation("relu"),  # a random dropout % of the hidden neurons are not used during each training step,
    AlphaDropout(DROPOUT),  # nor are they updated. The Batch Normalization normalizes the outputs from the hidden
    BatchNormalization()  # activation functions. This helps with neuron imbalance and can speed training significantly.
])  # Note this is an actual layer with some learnable parameters. It's not just min-maxing or standardizing.
# Loops over the hidden dims to add more layers
for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, activation="selu", kernel_initializer=weight_init))
    model.add(Dropout(DROPOUT, seed=SEED))
    model.add(BatchNormalization())
# Adds a final output layer with softmax to map to the 4 classes
model.add(Dense(4, activation="softmax", kernel_initializer=weight_init))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])
#es = EarlyStopping(monitor='val_loss',  patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
#mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
# %% -------------------------------------- Training Loop ----------------------------------------------------------
# Trains the MLP, while printing validation loss and metrics at each epoch
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(x_test, y_test),
          callbacks=[ModelCheckpoint("mlp_tan1300549644.hdf5", monitor="val_loss", save_best_only=True)])

# %% ------------------------------------------ Final test -------------------------------------------------------------
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))