# %% [markdown]
# # Image Pre-processing
# This is a script for pre-processing the images before training or testing. All images are resized into 224*224 so that they can fit in the VGG16 model. 
# 
# Public datasets of chest X-Ray (CXR) images of COVID-19 penumonia, non-COVID-19 penumonia, and healthy cases are used for this project. They were collected from various sources, including:
# 1. COVID-19 image data collection (https://github.com/ieee8023/covid-chestxray-dataset)
# 2. Chest X-Ray Images (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
# 3. Figure 1 COVID-19 Chest X-ray Dataset Initiative (https://github.com/agchung/Figure1-COVID-chestxray-dataset)
# 4. https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database?select=COVID-19_Radiography_Dataset

# %%
# packages setup
import os
import cv2
import numpy as np

# %%
# change the directory of the input folder and output folder
# you may use the provided raw data to create a set of training and validation data
# training data -- 'train_raw/covid' and 'train_raw/non-covid' 
# validation data -- 'valid_raw/covid' and 'valid_raw/non-covid'
input_dir = 'train_raw/non-covid'
output_dir = 'train_pro/non-covid'

# create a new directory for the output folder if it is not yet created
os.makedirs(output_dir, exist_ok=True)

# %%
# loop through every image of the input folder
for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    # resize the image proportionally 
    # crop additional pixels so that the output size is 224
    if h > w:
        new_h = 224
        new_w = max(int(w * new_h / h), 224)
    else:
        new_w = 224
        new_h = max(int(h * new_w / w), 224)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    crop_y = (new_h - 224) // 2
    crop_x = (new_w - 224) // 2
    img_cropped = img_resized[crop_y:crop_y+224, crop_x:crop_x+224, :]
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img_cropped)
    print(f"Resized and saved {filename} to {output_path}")

# %%
# check how many images are stored in the output image
count = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
print(f"Number of files in {output_dir}: {count}")


# %%



