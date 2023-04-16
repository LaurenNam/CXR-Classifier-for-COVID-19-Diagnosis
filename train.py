# %% [markdown]
# # CXR Classifier for COVID-19 --- 2. Training the Modfied VGG16 Model
# 
# This is a script to train or fine-tune a modified VGG16 model. This script allows you to load and train a modified VGG16 model. Once finished, the accuracy and loss function of the model on the training and validation data are plotted. The trained model is saved for later use.
# 
# 

# %% [markdown]
# __1. Install packages__

# %%
# packages setup
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model 
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

import itertools

import os.path

# %% [markdown]
# __2. Load the modfied model__

# %%
# load the modified model
# 'modified_models/model_tf.h5' for transfer learning
# 'modified_models/model_dl.h5' for traditional (deep) learning 
train_model = load_model('modified_models/model_tl.h5')
print("The modified VGG16 model is loaded.")
# %%
# check that the model you loaded is correct
# for transfer learning, only the parameters from the final layers are trainable (i.e., 8194 )
# for traditional learning, all parameters are trainable (i.e., 134,268,748)
# train_model.summary()

# %% [markdown]
# __3. Load training data__

# %%
# load data (the processed images) from the respective directories
train_path = 'train_pro'
valid_path = 'valid_pro'
print("The tranining data are loaded.")

# %%
# create batches of train, valid and test sets
# batch_size = no. of images drawn to the model per iteration
# all batches of images will be passed through the model after each epoch
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size = (224,224), classes= ['covid', 'non-covid'], batch_size = 10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size = (224,224), classes= ['covid', 'non-covid'], batch_size = 10)

# %%
# define plot to plot images with their respective labels 
# def plots(ims, figsize = (12,6), rows = 1, interp = False, titles = None):
#   if type(ims[0]) is np.ndarray:
#        ims = np.array(ims).astype(np.uint8)
#        if (ims.shape[-1] != 3):
#            ims = ims.tranpose((0,2,3,1))
#    f = plt.figure(figsize = figsize)
#    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
#    for i in range(len(ims)):
#        sp = f.add_subplot(rows, cols, i+1)
#        sp.axis('Off')
#        if titles is not None:
#            sp.set_title(titles[i], fontsize = 16)
#        plt.imshow(ims[i], interpolation = None if interp else 'none')

# %%
# draw a batch of training data 
# plot that batch of training data with their respective labels
# imgs, labels = next(train_batches)
# plots(imgs, titles = labels)

# %% [markdown]
# __4. Compiled and trained the modified model__

# %%
# compile the model
# the optimizer is Adam, with the learning rate at 0.0001
# the loss used is cross entropy loss
train_model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
print("The model is compiled and ready for training.")

# %%
# call GPU
# specify the GPU no. you want to use
CUDA_VISIBLE_DEVICES=2

# %%
# fit the data to train
# the no. of epoch is 50 by default
# the verbosity is 2 (super-verbose) by default
# details of training are stored in a variable called history
history = train_model.fit(x = train_batches, validation_data = valid_batches, epochs = 50, verbose = 2)

# %% [markdown]
# __5. Plot the training accuracy and loss function__

# %%
# Plot training and validation accuracy against epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# %%
# plot training and validation loss vs epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Fucntion')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# %%
# check that the weights changed after training if needed
# omit unless necessary
# model.get_weights()

# %% [markdown]
# __6. Save the trained model__

# %%
# save the fine-tuned model after training
# the architecture, weights, training configuration will be saved
if os.path.isfile('fine-tuned_models/model_trained.h5') is False:
    train_model.save('fine-tuned models/model_trained.h5')


