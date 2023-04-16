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
1. Image pre-processing ```'img-pro.ipynb'```
2. Modifying a pre-trained model ```'build.ipynb'```
3. Training the modified model ```'train.ipynb'```
4. Making predictions using the trained model ```'test.ipynb'```

These codes are written in jupyter notebook (.ipynb) format, which is the recommended format for code execution. 
The codes are also exported into python (.py) format for those who prefer raw python scripts. A shell file ```exe.sh``` is used for code execution. 



### Usage
__Pre-requisites__
All codes are written in python.
Below is the pre-requisites for running the python scripts for image pre-processing.
``` 
numpy == 1.18.5
```

Below are the pre-requisites for running the python scripts for modifying, training, and testing the model. 
```
CUDA 11.1
tensorflow == 2.4.0 
python == 3.8.2
numpy == 1.19.2 
matplotlib == 3.6.0
scikit-learn == 1.2.2
```

__To run the codes__
Type command ```bash exe.sh ``` in the console for all four steps of the project. You may disable any of the steps by adding # before its respective command line. 



### Models
Below provvides 9 models for running the codes. You may access all models via https://hkustconnect-my.sharepoint.com/:f:/g/personal/wktsuiac_connect_ust_hk/EnkWnRYYyVVDksSR45dqRXoBSzK1cM6_YaVEhnhGQxTF7g?e=sxcQld.

- __Modified VGG16 models__ 2 modified VGG16 models were built and available for fine-tuning by transfer learning, or direct traditional learning. You may access to the models in ```modified_models/``` 
- __Fine-tunned VGG16 models__ 7 fine-tuned VGG16 models trained at different training strategies are available for testing. Model 2 and 7 are the default model and optimal model for the project respectively. You may access to the models in ```fine-tuned_models/``` 



### Available datasets
Below provides a set of training and validation data for training a model in the default setting. Two sets of testing data are providied for making predictions. You may access to the datasets via ![alt text](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wktsuiac_connect_ust_hk/EgmlqxlInN9Er7zJC69v1mIBCE5nJSO181OrjvWxoUi9Ow?e=ymC6VE).

- __Training data__ A set of training data of 1,250 CXR images is provided in ```train_raw/```
- __Validation data__ A set of training data of 150 CXR images is provided in ```train_raw/```
- __Testing data__ Two sets of testing data of 560 CXR is provided in ```test_pro_1/``` and ```test_pro_2/``` 
