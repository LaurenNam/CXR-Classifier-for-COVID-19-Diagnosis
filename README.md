# CXR Classifier for COVID-19

### Objective
This project aims to train a model for classifying chest X-ray (CXR) images of patients into two classes, which are 'covid' and 'non-covid'. The first class refers to CXR images of COVID-19 pneumonia patients, whereas the second class includes CXR images of both non-COVID-19 pneumonia patients and healthy individuals. 


### Methodology
Pre-trained VGG16 model is leveraged to enable the classification tasks through transfer learning. VGG16 is a classification model pre-trained on ImageNet, and used for classifying images into 1000 classes. In this project, the model is modified into a binary classifier to classy images into 'covid' and 'non-covid' class.


### Dataset
Public datasets from various sources are used for fine-tuning the model. 
1. [COVID-10 image data collection] (https://github.com/ieee8023/covid-chestxray-dataset)
2. [Chest X-Ray Images] (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
3. [Figure 1 COVID-19 Chest X-ray Dataset Initiative] (https://github.com/agchung/Figure1-COVID-chestxray-dataset)
4. [COVID-19 Radiography Database] (https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database?select=COVID-19_Radiography_Dataset)

### Code implementation
The codes for this project can be splitted into four parts:
1. Image pre-processing 
2. Modifying a pre-trained model
3. Training the modified model
4. Making predictions using the trained model

These codes are written in jupyter notebook (.ipynb) format, which is the recommended format for code execution. 
The codes are also exported into python (.py) format for those who prefer raw python scripts. Each python file has a corresponidng shell (.sh) script for execution. 

### Usage
__Pre-requisites__
All codes are written in python.Below are the pre-requisites for running the python scripts for modifying, training, and testing the model. 
```
CUDA 11.1
tensorflow == 2.4.0 
python == 3.8.2
numpy == 1.19.2 
matplotlib == 3.6.0
scikit-learn == 1.2.2
```
__To run the codes__
```bash img-pro.sh``` for image pre-processing 
```bash build.sh``` for modifying the pre-trained model
```bash train.sh``` for training the modified model
```bash test.sh``` for testing the trained model

### Pre-trained models
