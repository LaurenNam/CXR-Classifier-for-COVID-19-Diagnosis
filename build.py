# %% [markdown]
# # CXR Classifier for COVID-19 --- 1. Modifying the VGG16 Model
# 
# This is a script to modify a pre-trained VGG16 model. 
# 
# This script allows you to load a pre-trained VGG16 model from tensorflow.keras, build a sequential model from it. The final classification layer of the model is changed from a 1000-unit classifier to a 2-unit classifier to cater the need of this task (i.e., COVID-19 diagnosis or classification)
# 
# If transfer learning is adopted, the weights of all layers before the final layer are freezed, so that they are not re-trained during fine-tuning. In other words, parameters within these layers are frozen and non-trainable. 
# 
# If traditional learning (i.e., deep learning) is adopted, all weights across all layers of the model can be retrained. All parameters remain trainable. 

# %% [markdown]
# ### Modified models
# 
# There are two modfiied VGG16 models available for usage. 
# 
# __'modified_models/model_tl.h5'__ is for transfer learning where all parameters before the final classification layer are frozen and non-trainable.
# 
# 
# __'modified_models/model_dl.h5'__ is for deep learning of the entire model wherea all parameters across the model can be re-trained. 

# %% [markdown]
# __1. Install packages__

# %%
# packages setup
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

import itertools

import os.path

# %% [markdown]
# __2. Load the pre-trained model__

# %%
# download the VGG16 model from tensorflow.keras
vgg16_model = tf.keras.applications.vgg16.VGG16()

# %%
# check the architecture of VGG16
# VGG16 should have 23 layers
# VGG16 should have a total of 138,357,544 parameters, which are all trainable
# VGG16 should be able to predict 1000 classes
# vgg16_model.summary()

# %%
# check the type of models of which VGG16 belongs to
# type(vgg16_model)

# %% [markdown]
# __3. Modify the pre-trained model__

# %%
# transform VGG16 from a Model to a Sequential model
model = Sequential()

# create an input layer as it may be removed when the VGG16 model was loaded into a Sequential Model
input_shape = (224, 224, 3)
input_layer = Input(shape = input_shape)
model.add(input_layer)

# # loop through every layer of VGG16 (except for the last layer) and add that to the newly created Sequential model
for layer in vgg16_model.layers[:-1]:
    model.add(layer)


# check that all layers of VGG16 were created in the Sequential model
# the no. of layers and parameters should be consistent between the VGG16 Model and VGG16 Sequential model
# model.summary()

# %%
# check that the input shape is correct
print(" A new sequential model is built. The input shape of the model is:", model.input_shape)

# %%
# freeze all layers 
# the weights of these layers will remain unchanged during fine-tuning
# omit this step if the model is retrained from scratch

for layer in model.layers:
    layer.trainable = False

# check the new architecture
# all trainable parameters should become non-trainable 
# the no. of trainable parameter should be 0
# the no. of trainable parameter should be 134,260,544
# model.summary()

# %%
# add a new Dense layer as the final layer 
# the Dense layer only classify data into 2 classes
# Softmax is used as the activation function
model.add(Dense(2, activation = 'softmax'))

# check the new architecture
# the outptut shape of the last layer should be (None, 2)
# the no. of trainable parameters should update from 0 to 8,194, which are all from the last layer
print("Check the architecture of the modified VGG16 model.")
model.summary()

# %% [markdown]
# __4. Save the modified model__

# %%
# save the modified after training
# the architecture, weights, training configuration will be saved
import os.path
if os.path.isfile('modified_models/model_name.h5') is False:
    model.save('modified_models/model_name.h5')
print("The modified VGG16 model is saved.")
# two models are modified and stored in modifed_models
# modified_models/model_tl.h5 <--- all parameters before the final layer are non-trainable
# modified_models/model_dl.h5 <--- all parameters across the model are trainable


