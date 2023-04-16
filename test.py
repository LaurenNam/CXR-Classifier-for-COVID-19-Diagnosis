# %% [markdown]
# # CXR Classifier for COVID-19 --- 3. Making Predictions
# 
# This is a script to test a trained VGG16 model. 
# 
# This script allows you to load a trained VGG16 model, pass data to it to make classification predictions. Ech trained VGG16 model can classify incoming CXR images into two classes -- 'covid' which refers to COVID-19 pneumonia CXR images, and 'non-covid' which includes CXR images of non-COVID-19 pneumonia and healthy individuals. 
# 
# The classification results will be mapped into a confusion matrix. Other evaluation metrics, including accuracy, sensitivity, specificity, precision, recall, f1-score (from tensorflow.keras) are also used. 

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

# %% [markdown]
# __2. Load the trained model__

# %%
# load the model you want to tested 
# the model is saved as test_model
from tensorflow.keras.models import load_model
test_model = load_model('model_7/save_model/model_7.h5')
print("A model is loaded.")
# test_model.summary()

# %% [markdown]
# __2. Load testing data__

# %%
# load the testing data
test_path = 'test_pro'
print("The testing set was loaded")
# %%
# create batches of testing sets
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size = (224,224), classes= ['covid', 'non-covid'], batch_size = 10, shuffle = False)


# %%
# make predictions of the testing data 
predictions = test_model.predict(x = test_batches, verbose = 0)
print("Testing is done.")

# %%
# set the lables for confusion matrix
# y_true = true labels
# y_pred = predicted lables
cm = confusion_matrix(y_true = test_batches.classes, y_pred = np.argmax(predictions, axis = -1))

# %%
# define the confution matrix
# turn on normalization by "normalize = Ture" if needed
def plot_confusion_matrix(cm, classes, normalize = False, 
                          title = 'Confusion Matrix', cmap = plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],
                horizontalalignment = 'center',
                color = 'white' if cm [i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# %%
# plot the confusion matrix
cm_plot_labels = ['covid','non-covid']
plot_confusion_matrix(cm = cm, classes = cm_plot_labels, title = 'Confusion Matrix')
print(plot_confusion_matrix)
# %%
# Compute the accuracy, sensitivity and specificity from the confusion matrix
tp, fn, fp, tn = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print the accuracy, sensitivity and specificity, and other classfiication matrix
print("True positive:", tp)
print("False positive:", fp)
print("True negative:", tn)
print("False negative:", fn)
print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
report = classification_report(y_true = test_batches.classes, y_pred = np.argmax(predictions, axis = -1), target_names = test_batches.class_indices)
print(report)


